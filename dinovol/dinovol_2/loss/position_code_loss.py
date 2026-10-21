import torch


def position_code_share(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Share of patch-token variance that is the same function of grid position across views.

    patch_tokens: [views, tokens, channels], all views on the same token grid.

    Crops and their orientations are random, so real content has no consistent mean at a
    given grid position. A component that is consistent across views is a code for the
    token's place in the crop, which the network can read from the crop borders (zero
    padding) and which makes iBOT targets predictable from position alone.

    The per-position mean of a finite batch also carries content noise of expected size
    (within-position variance) / views; it is subtracted, so the estimate is unbiased and
    the term does not reward shrinking content variance. The normalizer is not
    differentiated, so the term cannot be lowered by inflating the token scale.
    """
    tokens = patch_tokens.float()
    views = tokens.shape[0]
    if views < 2:
        return tokens.new_zeros(())
    position_mean = tokens.mean(0)
    grand_mean = position_mean.mean(0)
    within = (tokens - position_mean).square().sum(-1).mean() * (views / (views - 1))
    code = (position_mean - grand_mean).square().sum(-1).mean() - within / views
    total = (tokens - grand_mean).square().sum(-1).mean().detach()
    return code / total.clamp_min(1e-6)
