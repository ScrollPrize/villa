"""General candidate-path scoring from image evidence, context and geometry."""
import torch
from torch import nn
import torch.nn.functional as F


def passage_layers(input_width, hidden, heads):
    """Projection, path decoder and score head; callers own the layer names."""
    return (
        nn.Sequential(nn.Linear(input_width, hidden), nn.SiLU()),
        nn.TransformerEncoderLayer(hidden, heads, 2*hidden, dropout=0.,
                                   activation='gelu', batch_first=True, norm_first=True),
        nn.Sequential(nn.Linear(2*hidden+1, hidden), nn.SiLU(), nn.Linear(hidden, 1)),
    )


def passage_logits(projection, decoder, score, evidence, context, candidates, frontier):
    """Candidate-specific attention followed by prefix mean/max evidence pooling."""
    b, k, n, _ = candidates.shape
    geometry = candidates.detach()
    delta = torch.diff(torch.cat((frontier[:, None, None].expand(-1, k, 1, -1), geometry), 2), dim=2)
    token = projection(torch.cat((evidence, context.expand(-1, k, -1, -1),
                                  (geometry-frontier[:, None, None])/64, delta/4), -1))
    token = decoder(token.reshape(b*k, n, -1)).reshape(b, k, n, -1)
    count = torch.arange(1, n+1, device=token.device)[None, None, :, None]
    pooled = torch.cat((token.cumsum(2)/count, token.cummax(2).values,
                        (count/n).expand(b, k, -1, -1)), -1)
    return score(pooled).squeeze(-1).float()


def sample_path_features(features, points, crop, stride=1):
    """Sample actual encoder lattice centers, including odd crop dimensions."""
    index = points.float()/crop.spacing
    index = index+index.new_tensor(((crop.width-1)/2, (crop.width-1)/2, crop.behind))
    shape = index.new_tensor(tuple(reversed(features.shape[-3:])))
    grid = 2*index/stride/(shape-1).clamp_min(1)-1
    supported = torch.isfinite(grid).all(-1) & (grid.abs() <= 1).all(-1)
    values = F.grid_sample(features.float(), grid[:, :, None, None], align_corners=True)
    # Append support in the sampler's channel-major layout, then transpose.
    # Concatenating after the transpose lets Inductor miscompile the downstream
    # stencil reshape as contiguous, scrambling evidence and reading padding.
    values = values[:, :, :, 0, 0]
    return torch.cat((values, supported[:, None].float()), 1).transpose(1, 2)


class PassageScorer(nn.Module):
    """Score every prefix of arbitrary candidate paths independently.

    Evidence is [B,K,N,E]; context is [B,1|K,N,C]; paths are [B,K,N,3].
    Callers supply image/history features and the frontier [B,3]. No model,
    image encoder, horizon, candidate generator, or annotation is prescribed.
    """
    def __init__(self, evidence_width, context_width, hidden, heads):
        super().__init__()
        self.path_projection, self.path_decoder, self.passage_score = passage_layers(
            evidence_width+context_width+6, hidden, heads)

    def forward(self, evidence, context, candidates, frontier):
        return passage_logits(self.path_projection, self.path_decoder, self.passage_score,
                              evidence, context, candidates, frontier)
