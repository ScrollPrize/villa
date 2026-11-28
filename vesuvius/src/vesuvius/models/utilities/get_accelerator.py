import torch

def get_accelerator(include_xpu=False):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    elif include_xpu and hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print("Using XPU device (Intel)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device