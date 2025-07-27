from abc import ABC, abstractmethod
import torch
import numpy as np
from src.log_setup import logger


class BaseVADBackend(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(
        self, audio: list[np.ndarray], sampling_rate: int = 16000
    ) -> list[list[dict]]:

        raise NotImplementedError


class SileroVADBackend(BaseVADBackend):
    def __init__(
        self,
        model_dir: str = "snakers4/silero-vad",
        model="silero_vad",
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device = device

        if "cuda" in self.device and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning(f"CUDA不可用，切换到CPU")

        self._load_model(model_dir, model)

    def _load_model(self, model_dir: str, model: str) -> None:

        logger.info(f"开始加载VAD模型: {model} from {model_dir}")
        try:
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

            self.vad_model, funcs = torch.hub.load(
                repo_or_dir=model_dir, model=model, trust_repo=True
            )
            self.vad_model.to(self.device)
            self.detect_speech = funcs[0]

            logger.info(f"VAD模型加载成功: {model} on {self.device}")

        except Exception as e:
            logger.error(f"加载VAD模型失败: {e}")
            raise RuntimeError(f"加载VAD模型失败: {e}")

    def run(
        self, audio: list[np.ndarray], sampling_rate: int = 16000
    ) -> list[list[dict]]:
        if not audio:
            return []

        res = []
        for a in audio:
            try:
                r = self.detect_speech(
                    torch.tensor(a, dtype=torch.float32).to(self.device),
                    self.vad_model,
                    sampling_rate=sampling_rate,
                )
                res.append(r)
            except Exception as e:
                logger.error(f"处理音频段时出错: {e}")
                res.append([])

        return res


if __name__ == "__main__":
    from src.utils import load_audio

    audio = load_audio("data/第二批-20250725/63845843-1-208.mp4")
    audio2 = load_audio("data/第二批-20250725/213936315-1-192.mp4")
    vad = SileroVADBackend()
    res = vad.run([audio, audio2])
    print(res)
