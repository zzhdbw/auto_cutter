from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import numpy as np
from tqdm import tqdm
import jieba
from abc import ABC, abstractmethod
from typing import Literal

jieba.setLogLevel(60)


class BaseASRBackend(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, audio: list[np.ndarray], language: str = "auto") -> list[str]:
        pass


class SenceVoiceBackend(BaseASRBackend):
    def __init__(self, model_dir: str = "iic/SenseVoiceSmall", device: str = "cuda:0"):
        self.model = None
        self._load_model(model_dir, device)
        self.event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷", "😡", "😔", "😊"}

    def _load_model(self, model_dir: str, device: str):
        self.model = AutoModel(
            model=model_dir,
            # vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model="ct-punc",  # < | zh | > < | NEUTRAL | > < | S pe ech | > < | wo itn | >我从小就羡慕你。
            device=device,
            disable_update=True,
            disable_log=True,
            disable_pbar=True,
            log_level="ERROR",
        )

    def _clear_text(self, text: str) -> str:
        t = rich_transcription_postprocess(text.replace(" ", ""))
        for event in self.event_set:
            if event in t:
                t = t.replace(event, "")
        return t

    def run(
        self,
        audio: list[str] | list[np.ndarray],
        language: Literal[
            "auto",
            "zn",
            "en",
            "yue",
            "ja",
            "ko",
        ] = "auto",
        batch_size: int = 6,
    ) -> list[str]:

        res = []
        for i in range(0, len(audio), batch_size):
            batch_audio = audio[i : i + batch_size]

            batch_r = self.model.generate(
                input=batch_audio,
                cache={},
                language=language,  # "zn", "en", "yue", "ja", "ko", "nospeech"
            )

            clean_res = [self._clear_text(r["text"]) for r in batch_r]
            res.extend(clean_res)

        return res

    def run_by_indices(
        self,
        audios: list[np.ndarray],
        speech_array_indices: list[list[dict]],
        language: Literal["auto", "zn", "en", "yue", "ja", "ko"] = "auto",
    ) -> list[list[dict]]:

        res = []
        for audio, audio_segs in zip(audios, speech_array_indices):
            audio_segs_indices = [
                audio[int(seg["start"]) : int(seg["end"])] for seg in audio_segs
            ]

            text_list = self.run(
                audio_segs_indices,
                language=language,
            )

            r = []
            for seg, text in zip(audio_segs, text_list):
                temp = {"text": text, "origin_timestamp": seg}

                r.append(temp)

            res.append(r)

        return res


if __name__ == "__main__":
    from src.utils import load_audio
    from src.vad import SileroVADBackend

    sence_voice_backend = SenceVoiceBackend(model_dir="iic/SenseVoiceSmall")
    vad = SileroVADBackend()
    audio = load_audio("data/第二批-20250725/63845843-1-208.mp4")
    speech_array_indices = vad.run([audio])

    res = sence_voice_backend.run_by_indices(
        [audio],
        speech_array_indices,
    )

    print(res)
