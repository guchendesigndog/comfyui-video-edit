# ComfyUI Video Edit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-informational)](https://www.python.org/)

Video editing nodes for ComfyUI with **lazy frame-by-frame decoding**, designed to handle large/long videos without OOM crashes.

## Features

- **Lazy frame decoding** via PyAV — decode only the frames you need, never loads the entire video into memory
- **Chunked processing** for composite and crop operations (default 64 frames per chunk)
- **Multi-layer compositing** with position, scale, opacity, and blend modes
- **Video preview** with new API nodes: Video Load Preview (native UI), Video Save (format/codec options)
- Supports all common video formats (mp4, webm, mov, mkv, etc.)

## Installation

### Option 1: Clone from GitHub

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/guchendesigndog/comfyui-video-edit.git
pip install -r comfyui-video-edit/requirements.txt
```

### Option 2: ComfyUI Manager

Search for `ComfyUI-Video-Edit` in ComfyUI Manager and install directly.

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `av` (PyAV) | >= 10.0.0 | Frame-by-frame video decoding |

> `torch` (>= 2.0.0) and `torchvision` (>= 0.15.0) are required but included with ComfyUI — no need to install separately.
>
> `av` requires FFmpeg. On Windows: `conda install -c conda-forge av` or `pip install av`.

## Nodes

### Quick Reference

```
+---------------------------+-----------------------------------------------+
| 节点名称                   | 节点功能                                        |
+---------------------------+-----------------------------------------------+
| Video Load Preview        | 加载视频文件，支持原生预览，懒加载帧解码            |
+---------------------------+-----------------------------------------------+
| Video Save                | 保存视频，支持格式和编解码选择及原生预览            |
+---------------------------+-----------------------------------------------+
| Video Get Frame           | 提取指定索引的单帧，仅解码该帧                     |
+---------------------------+-----------------------------------------------+
| Video Get Frames Range    | 提取指定范围的连续帧，懒加载解码                   |
+---------------------------+-----------------------------------------------+
| Video Get Frame Rate      | 获取视频帧率，从元数据读取，不解码                  |
+---------------------------+-----------------------------------------------+
| Video Get Total Frames    | 获取视频总帧数，从元数据读取，不解码                |
+---------------------------+-----------------------------------------------+
| Video Crop Region         | 裁剪视频四边像素，自动对齐偶数尺寸                  |
+---------------------------+-----------------------------------------------+
| Image To Video            | 将单张图像重复为指定帧数的静止视频                  |
+---------------------------+-----------------------------------------------+
| Video Composite Layer     | 配置合成层参数（位置、缩放、透明度、混合模式）        |
+---------------------------+-----------------------------------------------+
| Video Composite           | 多图层视频合成，分块处理限制内存占用                |
+---------------------------+-----------------------------------------------+
```

### Detailed Description

| Node Name | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| **Video Load Preview** | Load a video file with native preview UI. Lazy loading — frames are decoded only when requested. Supports all common video formats. | `file` (video selector + upload) | `video` (VIDEO), `total_frames` (INT), `fps` (FLOAT), `width` (INT), `height` (INT) |
| **Video Save** | Save a video to the output directory with format and codec selection, plus native preview in ComfyUI. | `video` (VIDEO), `filename_prefix` (string), `format` (auto/mp4/webm/mov/mkv), `codec` (auto/h264/h265/vp9/av1) | Video preview in ComfyUI output |
| **Video Get Frame** | Extract a single frame at a specified index. Only decodes that one frame — does not load the rest of the video. | `video` (VIDEO), `frame_index` (INT, default 0) | `image` (IMAGE) |
| **Video Get Frames Range** | Extract a contiguous range of frames from a video. Uses lazy PyAV decoding to only materialize the requested frames. | `video` (VIDEO), `start_frame` (INT), `end_frame` (INT, -1 means last frame) | `images` (IMAGE) |
| **Video Get Frame Rate** | Get the frame rate of a video as an integer. Does not decode any frames — reads from file metadata. | `video` (VIDEO) | `frame_rate` (INT) |
| **Video Get Total Frames** | Get the total number of frames in a video as an integer. Does not decode any frames — reads from file metadata. | `video` (VIDEO) | `total_frames` (INT) |
| **Video Crop Region** | Crop pixels from edges of a video (top, bottom, left, right). Auto-aligns output to even dimensions. Processes in chunks to bound memory. | `video` (VIDEO), `crop_top` (INT), `crop_bottom` (INT), `crop_left` (INT), `crop_right` (INT) | `video` (VIDEO) |
| **Image To Video** | Create a still video by repeating a single image for a specified number of frames. Uses `expand()` for zero-copy frame repetition. | `image` (IMAGE), `frame_count` (INT), `frame_rate` (INT) | `video` (VIDEO) |
| **Video Composite Layer** | Configure one video layer for compositing. Sets position, scale, opacity, blend mode, and stacking order. Only stores configuration — does not process frames. | `video` (VIDEO), `x` (FLOAT, position 0-1), `y` (FLOAT, position 0-1), `scale` (FLOAT), `opacity` (FLOAT), `blend_mode` (normal/add/multiply/screen/overlay), `order` (INT), `mask` (MASK, optional) | `layer` (VIDEOCOMPOSITE_LAYER) |
| **Video Composite** | Composite up to 8 video layers onto an image canvas. Canvas is always the bottom layer. Layers sorted by `order` (bottom to top). Processes in chunks to bound memory. | `image` (IMAGE, canvas/background), `layer1` (required), `layer2`–`layer8` (optional) | `video` (VIDEO) |

## How It Works

### Lazy Decoding

ComfyUI's built-in `VideoFromFile.get_components()` decodes **all frames** into a single tensor, causing 50+ GB OOM for long 4K videos.

This extension bypasses that by using PyAV directly:

```
VideoFromFile.get_stream_source() -> av.open() -> decode only requested frame indices
```

### Chunked Processing

`VideoComposite` and `VideoCropRegion` process in chunks (default 64 frames):

```python
for chunk in range(0, total, chunk_size):
    decode only these frames -> process -> release -> next chunk
```

This keeps memory bounded regardless of video length.

## Usage Examples

### Extract Specific Frames

```
Video Load Preview -> Video Get Frame (frame_index=42) -> Save Image
```

### Crop and Save

```
Video Load Preview -> Video Crop Region (top=50, bottom=50, left=30, right=30) -> Video Save
```

### Multi-layer Composite

```
Video Load Preview (background) -> Image To Video (canvas)
Video Load Preview (layer 1) -> Video Composite Layer (x=0.5, y=0.5, scale=0.3)
Video Load Preview (layer 2) -> Video Composite Layer (x=0.8, y=0.2, scale=0.2, opacity=0.7)
All layers -> Video Composite -> Video Save
```

## License

[MIT License](LICENSE)
