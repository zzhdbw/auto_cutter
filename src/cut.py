from src.log_setup import logger


import os
from moviepy.editor import VideoFileClip


def cut_video(
    input_path: str, output_folder: str, file_name: str, segments: list[dict]
):
    os.makedirs(output_folder, exist_ok=True)

    def cut_one_video(input_path, output_path, start, end):
        clip = VideoFileClip(input_path)
        subclip = clip.subclip(start, end)
        subclip.write_videofile(
            output_path, codec="libx264", audio_codec="aac", logger=None
        )

    for segment in segments:
        start = segment['start']
        end = segment['end']
        cut_one_video(
            input_path,
            os.path.join(output_folder, f"{file_name}-{start}-{end}.mp4"),
            start,
            end,
        )
        with open(
            os.path.join(output_folder, f"{file_name}-{start}-{end}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(segment['text'])

        logger.info(
            f"gen {file_name}-{start}-{end}.mp4, video_length: {end-start:.3f}s, text: {segment['text']}"
        )
