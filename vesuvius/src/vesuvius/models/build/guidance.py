from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenBook3D(nn.Module):
    """Convert a 3D feature grid into a scalar spatial guide mask."""

    def __init__(
        self,
        *,
        n_tokens: int,
        embed_dim: int,
        dropout: float = 0.0,
        ema_decay: float | None = None,
        use_ema: bool = False,
        prototype_weighting: str = "mean",
        weight_mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.book = nn.Parameter(torch.randn(int(n_tokens), int(embed_dim)))
        nn.init.normal_(self.book, mean=0.0, std=float(embed_dim) ** -0.5)
        self.dropout = float(dropout)
        self.ema_decay = ema_decay
        self.use_ema = bool(use_ema)
        if ema_decay is not None:
            self.register_buffer("book_ema", self.book.detach().clone())
        else:
            self.book_ema = None

        self.prototype_weighting = str(prototype_weighting).strip().lower()
        if self.prototype_weighting not in {"mean", "token_mlp"}:
            raise ValueError(
                "prototype_weighting must be one of {'mean', 'token_mlp'}"
            )
        self.prototype_weight_mlp = None
        self.weight_mlp_hidden = None
        if self.prototype_weighting == "token_mlp":
            hidden = int(weight_mlp_hidden or embed_dim)
            if hidden <= 0:
                raise ValueError(f"weight_mlp_hidden must be > 0, got {hidden}")
            self.weight_mlp_hidden = hidden
            self.prototype_weight_mlp = nn.Sequential(
                nn.Linear(int(embed_dim), hidden),
                nn.GELU(),
                nn.Linear(hidden, int(n_tokens)),
            )

    @torch.no_grad()
    def _ema_update(self) -> None:
        if self.ema_decay is None or self.book_ema is None:
            return
        decay = float(self.ema_decay)
        self.book_ema.mul_(decay).add_(self.book.detach(), alpha=1.0 - decay)

    def forward(
        self,
        x: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"TokenBook3D expects [B, C, D, H, W], got {tuple(x.shape)}")

        if self.training:
            self._ema_update()

        batch_size, _channels, depth, height, width = x.shape
        if self.training or not self.use_ema or self.book_ema is None:
            book = self.book
        else:
            book = self.book_ema

        tokens = self.input_conv(x).flatten(2).transpose(1, 2)
        tokens = F.normalize(tokens, dim=-1)
        prototypes = F.normalize(book, dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

        similarities = torch.matmul(tokens, prototypes.transpose(1, 2))
        if self.dropout > 0.0:
            similarities = F.dropout(similarities, p=self.dropout, training=self.training)

        if self.prototype_weighting == "token_mlp":
            prototype_weights = F.softmax(self.prototype_weight_mlp(tokens), dim=-1)
            similarities = (similarities * prototype_weights).sum(dim=-1)
        else:
            similarities = similarities.mean(dim=-1)

        if token_mask is not None:
            token_mask = token_mask.to(device=similarities.device, dtype=similarities.dtype)
            similarities = similarities * token_mask

        guide = similarities.reshape(batch_size, 1, depth, height, width)
        guide = guide * 0.5 + 0.5
        return guide.clamp_(0.0, 1.0)
