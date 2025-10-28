
from vesuvius_unet3d import Vesuvius3dUnetModel
from youssef_mae import Vesuvius3dViTModel


def make_model(config):

    if config['model_type'] == 'unet':
        return Vesuvius3dUnetModel(in_channels=5, out_channels=config['step_count'] * 2, config=config)
    elif config['model_type'] == 'vit':
        return Vesuvius3dViTModel(
            mae_ckpt_path=config['model_config'].get('mae_ckpt_path', None),
            in_channels=5,
            out_channels=config['step_count'] * 2,
            input_size=config['crop_size'],
            patch_size=8,  # TODO: infer automatically from volume_scale and pretraining crop size
        )
    else:
        raise RuntimeError('unexpected model_type, should be unet or vit')
