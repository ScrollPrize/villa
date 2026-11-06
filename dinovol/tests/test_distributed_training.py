"""Two-rank CPU test; uses Gloo so one GPU is sufficient for the full suite."""
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dinovol_2.training.trainer import Trainer
from test_training import tiny_config, tiny_batch


def _worker(rank, folder):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    dist.init_process_group('gloo', init_method=f'file://{folder}/rendezvous', rank=rank, world_size=2)
    try:
        config = tiny_config(folder)
        config['centering'] = 'centering'
        config['model'].update(embedding_type='deeper',deeper_embed_stage_channels=[4,8],
            deeper_embed_stage_strides=[2,2],deeper_embed_depth_per_level=[1,1])
        config['object_losses'] = dict(enabled=True,head=dict(hidden_dim=8,out_dim=8,ink_dim=4,fpn_dim=4))
        trainer = Trainer(config)
        torch.manual_seed(rank + 50)
        batch = tiny_batch()
        batch.update(object_point_coordinates=torch.rand(4,4,3)*6,
            object_point_valid=torch.ones(4,4,dtype=torch.bool),
            object_point_objects=torch.tensor([[1,1,2,2]]*2),
            object_point_ink=torch.tensor([[1,1,-1,-1]]*2),
            object_negatives=torch.tensor([[[False,False,False],[False,False,True],[False,True,False]]]*2))
        trainer.train_step([batch,batch],0)
        for parameter in trainer.method.model_module.student.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference,src=0)
            torch.testing.assert_close(parameter,reference,rtol=0,atol=0)
        # Each rank needs its own ink queues and RNG stream on resume.
        trainer.method.object_loss.ink_pos_queue.fill_(float(rank))
        path = trainer.save_checkpoint(0)
        expected_rng = torch.rand(3)
        expected_queue = trainer.method.object_loss.ink_pos_queue.clone()
        dist.barrier()
        restored = Trainer(config)
        restored.load_checkpoint(path)
        torch.testing.assert_close(torch.rand(3),expected_rng,rtol=0,atol=0)
        torch.testing.assert_close(restored.method.object_loss.ink_pos_queue,expected_queue,rtol=0,atol=0)
        restored.train_step(batch,1)
    finally:
        dist.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    def test_region_head_and_rank_local_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(_worker,args=(folder,),nprocs=2,join=True)
