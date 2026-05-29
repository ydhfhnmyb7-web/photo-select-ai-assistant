from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec


@dataclass
class RuntimeInfo:
    torch_installed: bool = False
    torch_cuda_available: bool = False
    torch_version: str = ""
    torch_cuda_version: str = ""
    onnxruntime_installed: bool = False
    onnx_providers: list[str] = field(default_factory=list)
    onnx_cuda_available: bool = False
    onnx_dml_available: bool = False
    gpu_name: str = ""
    preferred_backend: str = "fallback"
    message: str = ""


def detect_model_runtime(use_gpu: bool = True) -> RuntimeInfo:
    """Detect optional model runtimes without making them mandatory imports."""
    info = RuntimeInfo(
        torch_installed=find_spec("torch") is not None,
        onnxruntime_installed=find_spec("onnxruntime") is not None,
    )

    if info.torch_installed:
        try:
            import torch

            info.torch_version = str(torch.__version__)
            info.torch_cuda_version = str(getattr(torch.version, "cuda", "") or "")
            info.torch_cuda_available = bool(use_gpu and torch.cuda.is_available())
            if info.torch_cuda_available:
                info.gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            info.message = f"torch runtime check failed: {exc}"
            info.torch_cuda_available = False

    if info.onnxruntime_installed:
        try:
            import onnxruntime as ort

            info.onnx_providers = [str(provider) for provider in ort.get_available_providers()]
            info.onnx_cuda_available = bool(use_gpu and "CUDAExecutionProvider" in info.onnx_providers)
            info.onnx_dml_available = bool(use_gpu and "DmlExecutionProvider" in info.onnx_providers)
        except Exception as exc:
            detail = f"onnxruntime check failed: {exc}"
            info.message = f"{info.message}; {detail}" if info.message else detail

    if info.torch_cuda_available:
        info.preferred_backend = "torch_cuda"
    elif info.onnx_cuda_available:
        info.preferred_backend = "onnx_cuda"
    elif info.onnx_dml_available:
        info.preferred_backend = "onnx_dml"
    elif info.torch_installed:
        info.preferred_backend = "torch_cpu"
    elif info.onnxruntime_installed:
        info.preferred_backend = "onnx_cpu"
    else:
        info.preferred_backend = "fallback"

    if not info.message:
        if info.preferred_backend == "fallback":
            info.message = "No GPU model runtime available; fallback extractor will be used."
        elif info.gpu_name:
            info.message = f"Model runtime: {info.preferred_backend} on {info.gpu_name}"
        else:
            info.message = f"Model runtime: {info.preferred_backend}"
    return info
