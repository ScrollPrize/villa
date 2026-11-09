"""Two Gloo ranks with source chunks reproduce the single-process full logical-batch gradient."""
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_cuda_pipeline import write_array
from test_jem_training import tiny_config
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
from dinovol_2.methods.jem import JemMethod

GLOBAL_BATCH = 4


def _samples(method, config):
    dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True), do_augmentations=True)
    np.random.seed(0)
    return dataset, [dataset[i] for i in range(GLOBAL_BATCH)]


def _step(method, batch, *, distributed):
    def sync_context(sync=True):
        return method.model.no_sync() if distributed and not sync else nullcontext()
    method.model_module.student.zero_grad(set_to_none=True)
    output = method.forward_backward(batch, 0, autocast=nullcontext, sync_context=sync_context,
                                     backward=lambda loss: loss.backward())
    return output, {name: p.grad.detach().clone() for name, p in method.model_module.student.named_parameters()}


def _worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank, world_size=2)
    try:
        root = Path(folder)
        config = tiny_config(root, batch=GLOBAL_BATCH, chunk=1, scope="crop_type")
        config["data"]["batch_size"] = GLOBAL_BATCH // 2
        torch.manual_seed(3)
        method = JemMethod(config, device="cpu", is_distributed=True, rank=rank)
        method.successful_updates = 1
        dataset, samples = _samples(method, config)
        local = method.build_processor(dataset, "cpu")(samples[rank * 2:(rank + 1) * 2])
        output, grads = _step(method, local, distributed=True)
        reference = torch.load(root / "reference.pt")
        # DDP averages gradients; the surrogate makes that the full-batch gradient.
        for name, grad in reference["grads"].items():
            torch.testing.assert_close(grads[name], grad, rtol=2e-4, atol=1e-6, msg=name)
        # The second update takes the single-pass running-marginal path. With unchanged parameters
        # its r is the all-reduced full-batch marginal and its counts are all-reduced from the
        # validity masks, so it reproduces the full-batch gradient too.
        self_check, running_grads = _step(method, local, distributed=True)
        assert float(self_check.metrics["marginal_lag_kl"]) < 1e-6
        for name, grad in reference["grads"].items():
            torch.testing.assert_close(running_grads[name], grad, rtol=2e-4, atol=1e-6, msg=f"running {name}")
        # The global-collapse value is computed from all-reduced statistics on every rank.
        torch.testing.assert_close(output.metrics["global_collapse_loss"], reference["global"], rtol=1e-5, atol=1e-7)
        loss = output.loss.clone()
        dist.all_reduce(loss)
        torch.testing.assert_close(loss / 2, reference["loss"], rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


class JemDistributedTests(unittest.TestCase):
    def test_two_ranks_with_chunks_match_full_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_array(root / "volume", np.random.default_rng(0).integers(1, 255, (48,) * 3, dtype=np.uint8))
            config = tiny_config(root, batch=GLOBAL_BATCH)
            # Same intra-op threading as the workers, so FP32 reductions run in the same order.
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                torch.manual_seed(3)
                method = JemMethod(config, device="cpu")
                method.successful_updates = 1
                dataset, samples = _samples(method, config)
                output, grads = _step(method, method.build_processor(dataset, "cpu")(samples), distributed=False)
            finally:
                torch.set_num_threads(threads)
            torch.save({"grads": grads, "loss": output.loss, "global": output.metrics["global_collapse_loss"]},
                       root / "reference.pt")
            mp.spawn(_worker, args=(folder,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
