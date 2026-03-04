"""
디바이스 감지 유틸리티.

CUDA → CPU 순으로 자동 감지한다.
MPS(Apple Silicon)는 ColBERT/SPLADE 등에서 OOM 위험이 있어 기본적으로 제외한다.
"""


def detect_device() -> str:
    """CUDA → CPU 순으로 디바이스를 자동 감지한다.

    MPS는 대형 모델(ColBERT, SPLADE) 실행 시 OOM 위험이 있어 포함하지 않는다.
    MPS가 필요한 경우 호출부에서 명시적으로 지정해야 한다.

    Returns:
        "cuda" 또는 "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA: {name}")
            return "cuda"
    except ImportError:
        pass
    print("[Device] CPU")
    return "cpu"
