def compute_3dunet_feature_map_shapes(
    patch_size: tuple,
    f_maps: list,
    pool_kernel_size: int = 2
):
    """
    Compute approximate (channels, D, H, W) for each encoder and decoder level
    in a typical 3D U-Net with a down/upscale factor = pool_kernel_size.

    Parameters
    ----------
    patch_size : tuple of (D, H, W)
        Spatial size of the input volume.
    f_maps : list of int
        Channels per level, e.g. [32, 64, 128, 256, ...].
    pool_kernel_size : int
        Factor by which we downsample in each encoder level (and upsample in decoder).

    Returns
    -------
    enc_shapes : list of (channels, D, H, W)
        Shapes for encoder outputs at each level (top to bottom).
    dec_shapes : list of (channels, D, H, W)
        Shapes for decoder outputs at each level (bottom to top).
    """
    D, H, W = patch_size

    # Encoder shapes
    enc_shapes = []
    curr_d, curr_h, curr_w = D, H, W
    for c in f_maps:
        enc_shapes.append((c, curr_d, curr_h, curr_w))
        # after this level, downsample by pool_kernel_size
        curr_d //= pool_kernel_size
        curr_h //= pool_kernel_size
        curr_w //= pool_kernel_size

    # Decoder shapes
    # Typically we do len(f_maps) - 1 "decoder steps"
    dec_shapes = []
    for i in reversed(range(len(f_maps) - 1)):
        curr_d *= pool_kernel_size
        curr_h *= pool_kernel_size
        curr_w *= pool_kernel_size
        dec_shapes.append((f_maps[i], curr_d, curr_h, curr_w))

    return enc_shapes, dec_shapes


def estimate_vram(
        patch_size: tuple,
        in_channels: int,
        out_channels_per_task: list,
        f_maps: list,
        num_tasks: int,
        batch_size: int = 1,
        dtype_size: int = 4,  # 4=FP32, 2=FP16, etc.
        pool_kernel_size: int = 2,
        trainable_params: int = 0,
        optimizer_multiplier: float = 4.0,
        norm_type: str = 'batch',
        num_groups: int = 8,
        use_se: bool = True,
        residual: bool = True,
        verbose: bool = True
):
    """
    Enhanced VRAM estimate accounting for SE blocks and residual connections.
    """
    d, h, w = patch_size

    # 1) Parameter memory with optimizer states
    param_mem = (trainable_params * dtype_size)
    optimizer_mem = param_mem * optimizer_multiplier

    # 2) Feature maps memory calculation
    enc_shapes, dec_shapes = compute_3dunet_feature_map_shapes(
        (d, h, w),
        f_maps,
        pool_kernel_size=pool_kernel_size
    )

    # Helper to calculate memory for one feature map
    def calc_feature_map_mem(c, dd, hh, ww):
        base_mem = batch_size * c * dd * hh * ww * dtype_size

        # Account for residual connections (need to store identity)
        if residual:
            base_mem *= 1.5

        # Account for SE operations
        if use_se:
            # Spatial SE needs additional spatial attention maps
            se_mem = batch_size * dd * hh * ww * dtype_size
            base_mem += se_mem

        return base_mem

    # Encoder memory (shared)
    enc_mem = 0
    for (c, dd, hh, ww) in enc_shapes:
        layer_mem = calc_feature_map_mem(c, dd, hh, ww)
        enc_mem += layer_mem
        if verbose:
            print(f"Encoder layer {c}ch: shape={dd}x{hh}x{ww}, mem={layer_mem / (1024 ** 2):.1f}MB")

    # Decoder memory (per task)
    dec_mem = 0
    for (c, dd, hh, ww) in dec_shapes:
        layer_mem = calc_feature_map_mem(c, dd, hh, ww)
        dec_mem += layer_mem
        if verbose:
            print(
                f"Decoder layer {c}ch: shape={dd}x{hh}x{ww}, mem={layer_mem / (1024 ** 2):.1f}MB (x{num_tasks} tasks)")
    dec_mem *= num_tasks

    # 3) Normalization memory
    norm_mem = 0
    if norm_type == 'group':
        num_norm_layers = sum(2 for _ in f_maps)  # 2 convs per level
        norm_intermediate_mem = (
                                        batch_size *
                                        sum(c * dd * hh * ww for c, dd, hh, ww in enc_shapes + dec_shapes) *
                                        dtype_size / num_groups
                                ) * 2  # Factor of 2 for mean and variance computations
        norm_mem = norm_intermediate_mem

    # 4) Input and output memory
    input_mem = batch_size * in_channels * d * h * w * dtype_size
    total_out_channels = sum(out_channels_per_task)
    output_mem = batch_size * total_out_channels * d * h * w * dtype_size

    # 5) Additional memory factors
    activation_factor = 3.0  # Higher factor to account for gradients and optimizer states
    fm_mem = (enc_mem + dec_mem) * activation_factor

    # Sum all components
    total_mem = (
            optimizer_mem +
            norm_mem +
            input_mem +
            output_mem +
            fm_mem
    )

    # Higher overhead factors for complex architectures
    pytorch_overhead = 1.8
    cuda_fragmentation = 1.4
    total_mem *= pytorch_overhead * cuda_fragmentation

    if verbose:
        print("\nMemory Breakdown (MB):")
        print(f"Parameter + Optimizer: {optimizer_mem / (1024 ** 2):.1f}")
        print(f"Input: {input_mem / (1024 ** 2):.1f}")
        print(f"Output: {output_mem / (1024 ** 2):.1f}")
        print(f"Encoder Features: {enc_mem / (1024 ** 2):.1f}")
        print(f"Decoder Features (all tasks): {dec_mem / (1024 ** 2):.1f}")
        print(f"Normalization: {norm_mem / (1024 ** 2):.1f}")
        print(f"Feature Maps Total (with activation factor {activation_factor}x): {fm_mem / (1024 ** 2):.1f}")
        print(f"\nRaw Total: {total_mem / (1024 ** 2):.1f}")
        print(f"With PyTorch overhead (1.8x) and CUDA fragmentation (1.4x)")

    return total_mem

