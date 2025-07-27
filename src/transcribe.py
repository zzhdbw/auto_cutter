import os
import time
import srt
import datetime

# import opencc

from src.asr import SenceVoiceBackend
from src.vad import SileroVADBackend
from src import utils
from src.cut import cut_video

from src.log_setup import logger
from tqdm import tqdm

# cc = opencc.OpenCC("t2s")

model_map = {
    "sensevoice": SenceVoiceBackend,
    "silero_vad": SileroVADBackend,
}


class Transcribe:
    def __init__(
        self,
        asr_model: str = "sensevoice",
        asr_args: dict = {},
        vad_model: str = "fsmn-vad",
        vad_args: dict = {},
        sampling_rate: int = 16000,
    ):
        if asr_model not in model_map:
            raise ValueError(f"Invalid asr model: {asr_model}")
        if vad_model not in model_map:
            raise ValueError(f"Invalid vad model: {vad_model}")

        self.sampling_rate = sampling_rate
        self.asr_model = model_map[asr_model](**asr_args)
        self.vad_model = model_map[vad_model](**vad_args)

        logger.info(f"模型初始化完成")

    def run(self, inputs: str, output: str, is_force: bool = False):
        for input in tqdm(inputs):
            logger.info(f"开始转录音频: {input}")
            name, _ = os.path.splitext(input)

            audio = utils.load_audio(input, sr=self.sampling_rate)
            audio_duration = len(audio) / self.sampling_rate
            logger.info(f"音频加载完成，时长: {audio_duration:.2f}秒")

            speech_array_indices = self.vad_model.run([audio])[0]

            # 去除短段落
            speeches = utils.remove_short_segments(
                speech_array_indices, 1 * self.sampling_rate
            )

            # 扩展段落
            speeches = utils.expand_segments(
                speeches,
                0.5 * self.sampling_rate,
                0.0 * self.sampling_rate,
                audio.shape[0],
            )

            # 合并相邻段落
            speeches = utils.merge_adjacent_segments(speeches, 0.5 * self.sampling_rate)
            # print(speeches)
            # print(audio)
            # exit()
            transcribe_results = self.asr_model.run_by_indices([audio], [speeches])[0]

            # 格式化结果,时间戳转化为秒
            transcribe_results = [
                {
                    "start": t['origin_timestamp']["start"] / self.sampling_rate,
                    "end": t['origin_timestamp']["end"] / self.sampling_rate,
                    "text": t["text"],
                }
                for t in transcribe_results
            ]

            cut_video(
                input,
                output,
                name.split("/")[-1],
                transcribe_results,
            )
            logger.info(f"转录音频: {input}完毕")


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--input",
        type=str,
        default="data/第二批-20250725/",
    )
    parse.add_argument(
        "--output",
        type=str,
        default="output",
    )
    args = parse.parse_args()

    # 列出input目录下全部mp4文件的路径
    input_files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".mp4")
    ]

    transcribe = Transcribe(
        asr_model="sensevoice",
        vad_model="silero_vad",
    )
    transcribe.run(inputs=input_files, output=args.output)
