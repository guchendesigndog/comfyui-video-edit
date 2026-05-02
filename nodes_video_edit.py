from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from fractions import Fraction
import logging

log = logging.getLogger(__name__)

try:
    from comfy.utils import common_upscale
except ImportError:
    common_upscale = None

# ---- Lazy imports for ComfyUI new API ----
_InputImpl = None
_Types = None
io = None
ui = None

def _get_native_video():
    global _InputImpl, _Types, io, ui
    if _InputImpl is None:
        try:
            from comfy_api.latest import InputImpl, Types, io as _io, ui as _ui
            _InputImpl = InputImpl
            _Types = Types
            io = _io
            ui = _ui
        except Exception:
            pass
    return _InputImpl, _Types, io, ui


# ---- VideoFromTensors wrapper ----

class VideoFromTensors:
    """Wraps a torch tensor as a ComfyUI-compatible video object."""
    def __init__(self, images: torch.Tensor, frame_rate: Fraction = Fraction(30, 1)):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        self._images = images.float()
        self._frame_rate = frame_rate

    @property
    def images(self):
        return self._images

    @property
    def frame_rate(self):
        return self._frame_rate

    def get_components(self):
        _, Types, _, _ = _get_native_video()
        if Types is not None:
            return Types.VideoComponents(
                images=self._images,
                frame_rate=self._frame_rate,
            )
        class _C:
            pass
        c = _C()
        c.images = self._images
        c.frame_rate = self._frame_rate
        c.audio = None
        c.metadata = None
        c.alpha = None
        return c

    def get_dimensions(self) -> tuple[int, int]:
        h, w = self._images.shape[1], self._images.shape[2]
        return w, h

    def get_duration(self) -> float:
        return float(self._images.shape[0] / self._frame_rate)

    def get_frame_count(self) -> int:
        return int(self._images.shape[0])

    def get_frame_rate(self) -> Fraction:
        return self._frame_rate

    def save_to(self, path, format="AUTO", codec="AUTO", metadata=None):
        import av
        ext = "mp4"
        if not str(path).endswith(ext):
            path = f"{path}.{ext}"
        container = av.open(path, mode="w")
        stream = container.add_stream("libx264", rate=self._frame_rate)
        h, w = self._images.shape[1], self._images.shape[2]
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for frame in self._images:
            arr = torch.clamp(frame[..., :3] * 255, min=0, max=255).to(
                device=torch.device("cpu"), dtype=torch.uint8
            ).numpy()
            vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)
        container.mux(stream.encode())
        container.close()

    def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
        return None


# ---- Helpers ----

def _extract_tensor(video) -> tuple[torch.Tensor, Fraction]:
    if isinstance(video, torch.Tensor):
        return video, Fraction(30, 1)
    if hasattr(video, "get_components"):
        components = video.get_components()
        return components.images, components.frame_rate
    if hasattr(video, "images") and hasattr(video, "frame_rate"):
        return video.images, video.frame_rate
    raise TypeError(f"Unsupported video type: {type(video)}")


def _video_meta(video) -> tuple[int, Fraction]:
    """Get frame count and frame rate WITHOUT materializing frames."""
    if hasattr(video, "get_frame_count") and hasattr(video, "get_frame_rate"):
        return int(video.get_frame_count()), video.get_frame_rate()
    tensor, frame_rate = _extract_tensor(video)
    return tensor.shape[0], frame_rate


def _decode_frames(video, indices: list[int]) -> torch.Tensor:
    """Decode only the requested frames, avoiding full materialization."""
    if hasattr(video, "get_frame"):
        frames = []
        for i in indices:
            f = video.get_frame(i)
            img = torch.from_numpy(f.to_rgb().to_ndarray()).float() / 255.0
            frames.append(img)
        if not frames:
            return torch.zeros(0, 0, 0, 3)
        return torch.stack(frames, dim=0)
    # VideoFromFile (ComfyUI): use PyAV to decode specific frames from file
    if hasattr(video, "get_stream_source") and hasattr(video, "get_frame_count"):
        return _decode_frames_via_av(video, indices)
    # VideoFromTensors: slice directly from stored tensor (avoid get_components)
    if hasattr(video, "images"):
        return video.images[indices]
    # Fallback: materialize everything
    tensor, _ = _extract_tensor(video)
    return tensor[indices]


def _video_has_lazy_decode(video) -> bool:
    """True if video can be decoded frame-by-frame without loading all frames.
    VideoFromFile has no get_frame() but has get_stream_source() → use PyAV."""
    if hasattr(video, "get_frame"):
        return True
    # VideoFromFile: has get_stream_source() + get_frame_count(), no get_frame()
    if hasattr(video, "get_stream_source") and hasattr(video, "get_frame_count"):
        return True
    return False


def _decode_frames_via_av(video, indices: list[int]) -> torch.Tensor:
    """Decode specific frames from VideoFromFile using PyAV directly.
    Opens the file with av, iterates through packets, decodes only matching frames."""
    import av
    source = video.get_stream_source()
    if not isinstance(source, str) or not os.path.isfile(source):
        tensor, _ = _extract_tensor(video)
        return tensor[indices]

    frame_set = set(indices)
    frame_count = video.get_frame_count()
    frames_out = []
    next_idx = 0

    with av.open(source) as container:
        video_stream = container.streams.video[0]
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if next_idx >= frame_count:
                    break
                idx = next_idx
                next_idx += 1
                if idx in frame_set:
                    img = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
                    frames_out.append((idx, img))
                    if len(frames_out) == len(frame_set):
                        break
            if len(frames_out) == len(frame_set):
                break

    if not frames_out:
        return torch.zeros(0, 0, 0, 3)

    frames_out.sort(key=lambda x: x[0])
    return torch.stack([f[1] for f in frames_out], dim=0)


def _wrap_output(images: torch.Tensor, frame_rate: Fraction) -> VideoFromTensors:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.dtype != torch.float32:
        images = images.float()
    return VideoFromTensors(images, frame_rate)


# ---- Classic Nodes ----

class VideoCropRegion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                "crop_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "crop_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "crop_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "crop_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "crop"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Crop pixels from edges. Output auto-shrinks and aligns to even dimensions."

    def crop(self, video, crop_top, crop_bottom, crop_left, crop_right, chunk_size=64):
        # Get metadata lazily - never call _extract_tensor or get_components()
        total, frame_rate = _video_meta(video)
        is_lazy = _video_has_lazy_decode(video)

        if is_lazy:
            # Use get_dimensions() for VideoFromFile (lazy, no frame materialization)
            W, H = video.get_dimensions()
            B, C = 1, 3
        else:
            tensor_full, frame_rate = _extract_tensor(video)
            B, H, W, C = tensor_full.shape

        if crop_top + crop_bottom >= H:
            raise ValueError(f"Vertical crop ({crop_top + crop_bottom}) >= height ({H})")
        if crop_left + crop_right >= W:
            raise ValueError(f"Horizontal crop ({crop_left + crop_right}) >= width ({W})")

        y_start = crop_top
        y_end = H - crop_bottom
        x_start = crop_left
        x_end = W - crop_right

        if (x_end - x_start) % 2 != 0:
            x_end -= 1
        if (y_end - y_start) % 2 != 0:
            y_end -= 1

        # Pre-allocate output tensor to avoid memory spike from torch.cat
        out_h = y_end - y_start
        out_w = x_end - x_start
        result = torch.empty((total, out_h, out_w, C), dtype=torch.float32, device="cpu")

        for s in range(0, total, chunk_size):
            e = min(s + chunk_size, total)
            if is_lazy:
                indices = list(range(s, e))
                chunk = _decode_frames(video, indices)
            else:
                chunk = tensor_full[s:e]

            # Already on CPU for both paths (lazy decode → CPU, tensor slice → .cpu())
            if chunk.device.type != "cpu":
                chunk = chunk.cpu()
            result[s:e] = chunk[:, y_start:y_end, x_start:x_end, :]

        log.info(f"VideoCropRegion: {W}x{H} -> {x_end - x_start}x{y_end - y_start} "
                 f"(top={crop_top} bottom={crop_bottom} left={crop_left} right={crop_right})")

        return (_wrap_output(result, frame_rate),)


class VideoGetFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_frame"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Extract a single frame from a video at the specified frame index."

    def get_frame(self, video, frame_index):
        total, _ = _video_meta(video)
        if frame_index < 0 or frame_index >= total:
            raise ValueError(f"Frame index {frame_index} out of range [0, {total - 1}]")

        if _video_has_lazy_decode(video):
            # Use chunk decode for single frame (works with both get_frame and VideoFromFile+PyAV)
            chunk = _decode_frames(video, [frame_index])
            return (chunk,)

        tensor, _ = _extract_tensor(video)
        return (tensor[frame_index:frame_index + 1],)


class VideoGetFramesRange:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 99999, "step": 1, "tooltip": "-1 means last frame"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "get_frames"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Extract frames from start to end index (inclusive). Set end_frame to -1 for last frame."

    def get_frames(self, video, start_frame, end_frame):
        total, _ = _video_meta(video)
        if start_frame < 0 or start_frame >= total:
            raise ValueError(f"Start frame {start_frame} out of range [0, {total - 1}]")
        if end_frame == -1:
            end_frame = total - 1
        if end_frame < start_frame:
            raise ValueError(f"End frame {end_frame} must be >= start frame {start_frame}")

        if _video_has_lazy_decode(video):
            indices = list(range(start_frame, end_frame + 1))
            return (_decode_frames(video, indices),)

        tensor, _ = _extract_tensor(video)
        return (tensor[start_frame:end_frame + 1],)


class VideoCompositeLayer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                "x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "X position (0=left, 0.5=center, 1=right)"}),
                "y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Y position (0=top, 0.5=center, 1=bottom)"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01, "tooltip": "Scale: 0.5=half, 1=original, 2=double"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["normal", "add", "multiply", "screen", "overlay"], {"default": "normal"}),
                "order": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Higher = on top"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("VIDEOCOMPOSITE_LAYER",)
    RETURN_NAMES = ("layer",)
    FUNCTION = "create_layer"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Configures one video layer with position, scale and opacity."

    def create_layer(self, video, x, y, scale, opacity, blend_mode, order, mask=None):
        return ({
            "video": video,
            "x": x,
            "y": y,
            "scale": scale,
            "opacity": opacity,
            "blend_mode": blend_mode,
            "order": order,
            "mask": mask,
        },)


class VideoComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layer1": ("VIDEOCOMPOSITE_LAYER",),
            },
            "optional": {
                "layer2": ("VIDEOCOMPOSITE_LAYER",),
                "layer3": ("VIDEOCOMPOSITE_LAYER",),
                "layer4": ("VIDEOCOMPOSITE_LAYER",),
                "layer5": ("VIDEOCOMPOSITE_LAYER",),
                "layer6": ("VIDEOCOMPOSITE_LAYER",),
                "layer7": ("VIDEOCOMPOSITE_LAYER",),
                "layer8": ("VIDEOCOMPOSITE_LAYER",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "composite"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Composites up to 8 video layers onto a canvas (background). Canvas = final output dimensions."

    def composite(self, image=None, canvas=None, layer1=None, layer2=None, layer3=None, layer4=None,
                  layer5=None, layer6=None, layer7=None, layer8=None, chunk_size=64):
        # Backward compat: workfile may still send 'canvas' from old schema
        if image is None:
            image = canvas
        layers = [l for l in [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8] if l is not None]
        if not layers:
            raise ValueError("At least one layer is required")

        canvas_width = image.shape[2]
        canvas_height = image.shape[1]
        layers.sort(key=lambda l: l.get("order", 0))

        # Determine frame count and device lazily
        vid0 = layers[0]["video"]
        num_frames, frame_rate = _video_meta(vid0)
        is_lazy = _video_has_lazy_decode(vid0)

        if is_lazy:
            w, h = vid0.get_dimensions()
            device = torch.device("cpu")
            dtype = torch.float32
        else:
            device = image.device
            dtype = image.dtype

        # Precompute per-layer metadata (no frame materialization for ANY video type)
        layer_info = []
        for layer in layers:
            lv = layer["video"]
            l_lazy = _video_has_lazy_decode(lv)

            if l_lazy:
                w, h = lv.get_dimensions()
                fps_l = lv.get_frame_rate()
                lf_count = lv.get_frame_count()
            else:
                if hasattr(lv, "get_dimensions"):
                    w, h = lv.get_dimensions()
                else:
                    tensor_peek, fps_l = _extract_tensor(lv)
                    w, h = tensor_peek.shape[2], tensor_peek.shape[1]
                    lf_count = tensor_peek.shape[0]
                    fps_l = Fraction(30, 1)
                if hasattr(lv, "get_frame_rate"):
                    fps_l = lv.get_frame_rate()
                else:
                    fps_l = Fraction(30, 1)
                if hasattr(lv, "get_frame_count"):
                    lf_count = lv.get_frame_count()
                else:
                    lf_count = tensor_peek.shape[0]

            sw = max(1, int(round(w * layer["scale"])))
            sh = max(1, int(round(h * layer["scale"])))
            if sw % 2 != 0:
                sw += 1
            if sh % 2 != 0:
                sh += 1

            px = int(round(layer["x"] * canvas_width))
            py = int(round(layer["y"] * canvas_height))

            layer_info.append({
                "video": lv,
                "is_lazy": l_lazy,
                "lf_count": lf_count,
                "sw": sw, "sh": sh,
                "px": px, "py": py,
                "blend_fn": self._get_blend_fn(layer["blend_mode"]),
                "opacity": layer["opacity"],
            })

        # Process in chunks - ALL videos use _decode_frames (lazy or tensor slicing)
        out_frames = []
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            chunk_len = end - start

            if image.shape[0] == 1:
                # expand() creates overlapping views; clone() gives each chunk its own memory
                work = image.to(device=device, dtype=dtype).expand(chunk_len, -1, -1, -1).clone()
            else:
                work = image[start:end].to(device=device, dtype=dtype)

            for li in layer_info:
                sw, sh = li["sw"], li["sh"]
                px, py = li["px"], li["py"]

                # Decode/extract chunk - unified path for all video types
                indices = list(range(start, end))
                chunk_lt = _decode_frames(li["video"], indices)

                if chunk_lt.shape[0] == 0:
                    continue

                # Scale this chunk
                orig_h, orig_w = chunk_lt.shape[1], chunk_lt.shape[2]
                if sw != orig_w or sh != orig_h:
                    if common_upscale is not None:
                        chunk_lt = common_upscale(chunk_lt.movedim(-1, 1), sw, sh, "lanczos", "disabled").movedim(1, -1)
                    else:
                        chunk_lt = F.interpolate(chunk_lt.permute(0, 3, 1, 2), size=(sh, sw), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

                # Dynamic overlap: recompute after scaling so src and dst are always the same HxW
                cx0 = max(0, px)
                cy0 = max(0, py)
                cx1 = min(canvas_width, px + sw)
                cy1 = min(canvas_height, py + sh)
                sx0 = cx0 - px
                sy0 = cy0 - py
                sh_overlap = cy1 - cy0
                sw_overlap = cx1 - cx0

                if sh_overlap <= 0 or sw_overlap <= 0:
                    continue

                # src and dst slices are guaranteed the same HxW
                src = chunk_lt[:, sy0:sy0 + sh_overlap, sx0:sx0 + sw_overlap, :]
                dst = work[:, cy0:cy1, cx0:cx1, :]
                alpha = torch.full((1, 1, 1, 1), li["opacity"], device=device, dtype=dtype)

                work[:, cy0:cy1, cx0:cx1, :] = li["blend_fn"](src, dst) * alpha + dst * (1 - alpha)

            out_frames.append(work.cpu())
            del work

        result = torch.cat(out_frames, dim=0)
        return (_wrap_output(result, frame_rate),)

    @staticmethod
    def _get_blend_fn(mode):
        if mode == "add":
            return lambda src, dst: torch.clamp(src + dst, 0.0, 1.0)
        elif mode == "multiply":
            return lambda src, dst: src * dst
        elif mode == "screen":
            return lambda src, dst: 1.0 - (1.0 - src) * (1.0 - dst)
        elif mode == "overlay":
            def overlay(src, dst):
                return torch.where(dst < 0.5, 2.0 * src * dst, 1.0 - 2.0 * (1.0 - src) * (1.0 - dst))
            return overlay
        return lambda src, dst: src


class VideoGetFrameRate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_rate",)
    FUNCTION = "get_framerate"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Returns the frame rate of a video as an integer."

    def get_framerate(self, video):
        if hasattr(video, "get_frame_rate"):
            return (int(video.get_frame_rate()),)
        _, frame_rate = _extract_tensor(video)
        return (int(frame_rate),)


class VideoGetTotalFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_frames",)
    FUNCTION = "get_total_frames"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Returns the total number of frames in a video as an integer."

    def get_total_frames(self, video):
        if hasattr(video, "get_frame_count"):
            return (int(video.get_frame_count()),)
        tensor, _ = _extract_tensor(video)
        return (int(tensor.shape[0]),)


class ImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_count": ("INT", {"default": 30, "min": 1, "max": 99999, "step": 1}),
                "frame_rate": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1, "tooltip": "Frames per second of the output video"}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "convert"
    CATEGORY = "VideoEdit"
    DESCRIPTION = "Repeats a single image for the specified number of frames to create a still video."

    def convert(self, image, frame_count, frame_rate):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # Use a lazy wrapper that doesn't materialize all frames in memory.
        # Each frame is the same reference to the single image tensor.
        lazy_images = image.expand(frame_count, -1, -1, -1)
        return (_wrap_output(lazy_images, Fraction(frame_rate)),)


# ---- Classic Node Mappings ----

NODE_CLASS_MAPPINGS = {
    "VideoCropRegion": VideoCropRegion,
    "VideoGetFrame": VideoGetFrame,
    "VideoGetFramesRange": VideoGetFramesRange,
    "VideoGetFrameRate": VideoGetFrameRate,
    "VideoGetTotalFrames": VideoGetTotalFrames,
    "ImageToVideo": ImageToVideo,
    "VideoCompositeLayer": VideoCompositeLayer,
    "VideoComposite": VideoComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCropRegion": "Video Crop Region",
    "VideoGetFrame": "Video Get Frame",
    "VideoGetFramesRange": "Video Get Frames Range",
    "VideoGetFrameRate": "Video Get Frame Rate",
    "VideoGetTotalFrames": "Video Get Total Frames",
    "ImageToVideo": "Image To Video",
    "VideoCompositeLayer": "Video Composite Layer",
    "VideoComposite": "Video Composite",
}
