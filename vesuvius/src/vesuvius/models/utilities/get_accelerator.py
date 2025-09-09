import torch

def get_accelerator():
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device