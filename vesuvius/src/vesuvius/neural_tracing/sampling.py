import torch
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm


@torch.inference_mode()
def sample_ddim(unet, conditioning, train_scheduler, shape, generator, eta=0., num_inference_steps=50, use_clipped_model_output=None, cfg_scale=1.):
    r"""
    unet (`torch.nn.Module`):
        The unet to use for denoising.
    conditioning (`torch.Tensor`):
        The conditioning to concat with noise.
    train_scheduler (`diffusers.SchedulerMixin`):
        The scheduler used for training.
    shape (`tuple`):
        Shape of noise/samples, including batch dimension.
    generator (`torch.Generator`, *optional*):
        A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
        generation deterministic.
    eta (`float`, *optional*, defaults to 0.0):
        Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only
        applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0`
        corresponds to DDIM and `1` corresponds to DDPM.
    num_inference_steps (`int`, *optional*, defaults to 50):
        The number of denoising steps. More denoising steps usually lead to a higher quality image at the
        expense of slower inference.
    use_clipped_model_output (`bool`, *optional*, defaults to `None`):
        If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
        downstream to the scheduler (use `None` for schedulers which don't support this argument).
    """

    if isinstance(generator, list) and len(generator) != shape[0]:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {shape[0]}. Make sure the batch size matches the length of the generators."
        )

    image = randn_tensor(shape, generator=generator, device=conditioning.device, dtype=conditioning.dtype)

    scheduler = DDIMScheduler.from_config(train_scheduler.config)
    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps, desc='sampling'):
        timesteps = torch.full((image.shape[0],), t, device=conditioning.device, dtype=torch.long)
        inputs = torch.cat([conditioning, image], dim=1)
        model_output = unet(inputs, timesteps)
        if cfg_scale != 1.:
            input_uncond = torch.cat([torch.zeros_like(conditioning), image], dim=1)
            model_output_uncond = unet(input_uncond, timesteps)
            model_output += (model_output - model_output_uncond) * (cfg_scale - 1.)
        image = scheduler.step(
            model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
        ).prev_sample

    return image
