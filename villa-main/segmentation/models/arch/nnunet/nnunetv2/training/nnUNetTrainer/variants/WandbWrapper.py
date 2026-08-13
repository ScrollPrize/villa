import wandb
from datetime import datetime

class WandbWrapper():
    def __init__(self, use_wandb=True, config=None):
        self.use_wandb = use_wandb
        self.config = config if use_wandb else {}

    def init(self, local_rank=0, config=None):
        if config is None:
            config = self.config

        if self.use_wandb:
            if local_rank == 0:
                nnow = datetime.now()
                ndate_str = nnow.strftime("%Y.%m.%d")
                ntime_str = nnow.strftime("%H:%M:%S")
                wandb_name = f"{config['project_name']}_{config['architecture']}_fold_{config['fold']}_rank{local_rank}_{ndate_str}_{ntime_str}"
                return wandb.init(
                    project=config['project_name'], 
                    name=wandb_name, 
                    config=config
                )
            else:
                return wandb.init(mode="disabled")

    def log(self, *args, **kwargs):
        if self.use_wandb:
            return wandb.log(*args, **kwargs)

    def finish(self):
        if self.use_wandb:
            return wandb.finish()
