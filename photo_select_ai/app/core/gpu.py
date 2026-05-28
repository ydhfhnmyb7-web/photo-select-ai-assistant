from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass
class AIEnvironment:
    torch_installed: bool
    cv2_installed: bool
    clip_installed: bool
    cuda_available: bool
    gpu_name: str
    source: str
    message: str
    torch_version: str = ""
    cuda_version: str = ""

    @property
    def ai_ready(self) -> bool:
        return self.cv2_installed


def detect_ai_environment(use_gpu: bool = True) -> AIEnvironment:
    torch_installed = find_spec("torch") is not None
    cv2_installed = find_spec("cv2") is not None
    clip_installed = find_spec("open_clip") is not None or find_spec("open_clip_torch") is not None

    cuda_available = False
    gpu_name = ""
    torch_version = ""
    cuda_version = ""
    if torch_installed:
        try:
            import torch

            torch_version = str(torch.__version__)
            cuda_version = str(getattr(torch.version, "cuda", "") or "")
            cuda_available = bool(use_gpu and torch.cuda.is_available())
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_available = False

    if not cv2_installed:
        source = "fallback"
        message = "AI依赖缺失：未安装 opencv-python，AI功能已禁用"
    elif not torch_installed:
        source = "fallback/no_torch"
        message = "CPU模式：torch未安装，使用OpenCV轻量分析"
    elif cuda_available:
        source = f"GPU: CUDA {gpu_name}"
        message = f"GPU: CUDA · {gpu_name} · torch {torch_version} · CUDA {cuda_version or '-'}"
    else:
        source = "CPU"
        message = f"CPU模式：CUDA不可用 · torch {torch_version}"

    return AIEnvironment(
        torch_installed=torch_installed,
        cv2_installed=cv2_installed,
        clip_installed=clip_installed,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        source=source,
        message=message,
        torch_version=torch_version,
        cuda_version=cuda_version,
    )
