import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dinovol_2.training.trainer import Trainer


class WorkerRandomDataset(Dataset):
    """Ignores the index and samples from the worker's RNGs, like SSLZarrDataset's crops."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return np.random.randint(2 ** 31)


def worker_draws(rank, seed=0):
    torch.manual_seed(seed)  # every process starts from the same default torch seed
    generator = Trainer._loader_generator(SimpleNamespace(rank=rank))
    loader = DataLoader(WorkerRandomDataset(), batch_size=2, num_workers=2, generator=generator)
    return [batch.tolist() for batch in loader]


class RankSeedTests(unittest.TestCase):
    def test_ranks_draw_different_samples_deterministically(self):
        rank0, rank1 = worker_draws(0), worker_draws(1)
        self.assertNotEqual(rank0, rank1)
        self.assertEqual(rank0, worker_draws(0))
        self.assertNotEqual(rank0, worker_draws(0, seed=1))  # a restored RNG state gives new seeds

    def test_generator_seed_depends_on_rank_and_global_state(self):
        seeds = []
        for rank in (0, 1, 2, 0):
            torch.manual_seed(0)
            seeds.append(Trainer._loader_generator(SimpleNamespace(rank=rank)).initial_seed())
        self.assertEqual(len(set(seeds[:3])), 3)
        self.assertEqual(seeds[0], seeds[3])


if __name__ == "__main__":
    unittest.main()
