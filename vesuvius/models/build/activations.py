import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: SwiGLU(x) = Swish(x1) * x2
    where x1, x2 are the first and second half of the input channels
    
    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        # Split input in half along the channel dimension
        x1, x2 = x.chunk(2, dim=self.dim)
        # Apply Swish (SiLU) to first half with clamping for stability
        # Clamp prevents numerical overflow that can cause NaN
        gate = F.silu(x1.clamp(min=-10, max=10))
        return gate * x2


class SwiGLUBlock(nn.Module):
    """
    Linear(d → 2×hidden) → SwiGLU → Linear(hidden → d)
    """
    def __init__(self, channels, conv_op, bias=True, expansion_factor=1.5):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        # Ensure hidden_channels is even for the split in SwiGLU
        hidden_channels = 2 * (hidden_channels // 2)
        
        # Expand to 2×hidden channels (for gate and value pathways)
        # bias=True helps with stability
        self.w_gate = conv_op(channels, hidden_channels, kernel_size=1, bias=bias)
        # SwiGLU activation splits channels in half
        self.swiglu = SwiGLU(dim=1)
        # Project back from hidden/2 to original channels
        self.w_out = conv_op(hidden_channels // 2, channels, kernel_size=1, bias=bias)
        
        # Initialize weights specifically for SwiGLU
        self._initialize_swiglu_weights()
        
    def _initialize_swiglu_weights(self):
        """Custom initialization for SwiGLU to prevent NaN issues"""
        with torch.no_grad():
            # Conservative initialization for gate pathway
            # Use smaller gain to prevent explosion
            nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
            if hasattr(self.w_gate, 'bias') and self.w_gate.bias is not None:
                nn.init.zeros_(self.w_gate.bias)
            
            # Output projection initialized with small values
            nn.init.xavier_uniform_(self.w_out.weight, gain=0.5)
            if hasattr(self.w_out, 'bias') and self.w_out.bias is not None:
                nn.init.zeros_(self.w_out.bias)
        
    def forward(self, x):
        # Expand → SwiGLU → Project
        x = self.w_gate(x)
        x = self.swiglu(x)  # Output has hidden_channels // 2
        x = self.w_out(x)
        return x


class GLU(nn.Module):
    """
    Gated Linear Unit: GLU(x) = x1 * sigmoid(x2)
    where x1, x2 are the first and second half of the input channels
    
    Reference: https://arxiv.org/abs/1612.08083
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        # Split input in half along the channel dimension
        x1, x2 = x.chunk(2, dim=self.dim)
        # Apply sigmoid to second half, multiply by first half
        return x1 * torch.sigmoid(x2)


class GLUBlock(nn.Module):
    """
    Linear(d → 2×hidden) → GLU → Linear(hidden → d)
    Similar to SwiGLUBlock but with GLU activation.
    """
    def __init__(self, channels, conv_op, bias=False, expansion_factor=8/3):
        super().__init__()
        # expansion factor similar to SwiGLU
        hidden_channels = int(channels * expansion_factor)
        # Ensure hidden_channels is even for the split in GLU
        hidden_channels = 2 * (hidden_channels // 2)
        
        # Expand to 2×hidden channels (for gate and value pathways)
        self.w_gate = conv_op(channels, hidden_channels, kernel_size=1, bias=bias)
        # GLU activation splits channels in half
        self.glu = GLU(dim=1)
        # Project back from hidden/2 to original channels
        self.w_out = conv_op(hidden_channels // 2, channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        # Expand → GLU → Project
        x = self.w_gate(x)
        x = self.glu(x)  # Output has hidden_channels // 2
        x = self.w_out(x)
        return x
