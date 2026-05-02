"""
@author: guchendesigndog
@title: ComfyUI Video Edit
@nickname: Video Edit Nodes
@description: Video editing nodes for ComfyUI with lazy frame-by-frame decoding for large video support.
"""

__version__ = "1.0.0"

from .nodes_video_edit import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# ---- New API nodes with full video preview ----
try:
    import os
    import folder_paths
    from comfy_api.latest import io, ui
    from comfy_api.latest import ComfyExtension, InputImpl, Types
    from typing_extensions import override

    class _VideoLoad(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            files = folder_paths.filter_files_content_types(files, ["video"])
            return io.Schema(
                node_id="VideoLoadPreview",
                display_name="Video Load Preview",
                category="VideoEdit",
                inputs=[
                    io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
                ],
                outputs=[
                    io.Video.Output("video"),
                    io.Int.Output("total_frames"),
                    io.Float.Output("fps"),
                    io.Int.Output("width"),
                    io.Int.Output("height"),
                ],
            )

        @classmethod
        def execute(cls, file):
            video_path = folder_paths.get_annotated_filepath(file)
            video_obj = InputImpl.VideoFromFile(video_path)
            w, h = video_obj.get_dimensions()
            total = video_obj.get_frame_count()
            fps = float(video_obj.get_frame_rate())
            return io.NodeOutput(video_obj, int(total), fps, int(w), int(h))

        @classmethod
        def validate_inputs(cls, file):
            if not folder_paths.exists_annotated_filepath(file):
                return f"Invalid video file: {file}"
            return True

        @classmethod
        def fingerprint_inputs(cls, file):
            return os.path.getmtime(folder_paths.get_annotated_filepath(file))

    class _VideoSave(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="VideoSave",
                display_name="Video Save",
                category="VideoEdit",
                inputs=[
                    io.Video.Input("video"),
                    io.String.Input("filename_prefix", default="video/ComfyUI"),
                    io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto"),
                    io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto"),
                ],
                hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, video, filename_prefix, format, codec, prompt=None, extra_pnginfo=None):
            from comfy.cli_args import args
            w, h = video.get_dimensions()
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), w, h)
            saved_metadata = None
            if not args.disable_metadata:
                metadata = {}
                if extra_pnginfo is not None:
                    metadata.update(extra_pnginfo)
                if prompt is not None:
                    metadata["prompt"] = prompt
                if metadata:
                    saved_metadata = metadata
            ext = Types.VideoContainer.get_extension(format)
            file = f"{filename}_{counter:05}_.{ext}"
            video.save_to(os.path.join(full_output_folder, file),
                          format=Types.VideoContainer(format), codec=Types.VideoCodec(codec),
                          metadata=saved_metadata)
            return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

    class _Ext(ComfyExtension):
        @override
        async def get_node_list(self):
            return [_VideoLoad, _VideoSave]

    async def comfy_entrypoint():
        return _Ext()

except Exception:
    pass  # New API not available, classic nodes only
