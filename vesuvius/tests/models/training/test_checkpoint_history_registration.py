"""0100 prepared CPU tests; execute only in the frozen future PBS context."""
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from vesuvius.models.training.save_checkpoint import manage_checkpoint_history, manage_debug_gifs
from vesuvius.models.training.train import BaseTrainer


class CheckpointHistoryRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.manual_seed(0)

    def trainer_saves(self, losses):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.mgr = SimpleNamespace(model_name='model')
        trainer.device = torch.device('cpu')
        trainer.ema_model = None
        trainer.ema_save_in_checkpoint = False
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        history, best = deque(maxlen=3), []
        for epoch, loss in enumerate(losses):
            history, best, _ = trainer._on_epoch_end(
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                train_dataset=None, ckpt_dir=str(self.root),
                model_ckpt_dir=str(self.root), checkpoint_history=history,
                best_checkpoints=best, avg_val_loss=loss,
            )
        return model, history, best

    def manager_saves(self, kind, pre_register):
        history, best = deque(maxlen=3), []
        for epoch, loss in enumerate([3.0, 2.0, 1.0]):
            if kind == 'checkpoint':
                path = self.root / f'model_epoch{epoch + 1}.pth'
            else:
                path = self.root / f'model_debug_epoch{epoch + 1}.mp4'
            # Bookkeeping fixture only; these bytes are not encoded video or a checkpoint.
            path.write_bytes(f'sentinel epoch {epoch}'.encode())
            if pre_register:
                history.append((epoch, str(path)))
            common = dict(epoch=epoch, validation_loss=loss,
                          checkpoint_dir=str(self.root), model_name='model',
                          max_recent=3, max_best=2)
            if kind == 'checkpoint':
                history, best = manage_checkpoint_history(
                    history, best, checkpoint_path=path, **common)
            else:
                history, best = manage_debug_gifs(
                    history, best, gif_path=path, **common)
        return history, best

    def test_trainer_keeps_three_recent_checkpoints(self):
        model, history, _ = self.trainer_saves([3.0, 2.0, 1.0])
        self.assertEqual([epoch for epoch, _ in history], [0, 1, 2])
        self.assertEqual({p.name for p in self.root.glob('model_epoch*.pth')},
                         {'model_epoch1.pth', 'model_epoch2.pth', 'model_epoch3.pth'})
        loaded = torch.load(self.root / 'model_epoch3.pth', map_location='cpu', weights_only=False)
        self.assertEqual(loaded['epoch'], 2)
        torch.testing.assert_close(loaded['model']['weight'], model.weight.detach())

    def test_debug_caller_keeps_three_recent_files(self):
        history, _ = self.manager_saves('debug', pre_register=True)
        self.assertEqual([epoch for epoch, _ in history], [0, 1, 2])
        self.assertEqual({p.name for p in self.root.glob('model_debug_epoch*.mp4')},
                         {'model_debug_epoch1.mp4', 'model_debug_epoch2.mp4', 'model_debug_epoch3.mp4'})

    def test_direct_checkpoint_manager_adds_each_epoch(self):
        history, _ = self.manager_saves('checkpoint', pre_register=False)
        self.assertEqual([epoch for epoch, _ in history], [0, 1, 2])
        for epoch, path in history:
            self.assertEqual(Path(path).read_bytes(), f'sentinel epoch {epoch}'.encode())

    def test_direct_debug_manager_adds_each_epoch(self):
        history, _ = self.manager_saves('debug', pre_register=False)
        self.assertEqual([epoch for epoch, _ in history], [0, 1, 2])
        for epoch, path in history:
            self.assertEqual(Path(path).read_bytes(), f'sentinel epoch {epoch}'.encode())

    def test_older_best_checkpoint_survives_recent_window(self):
        _, _, best = self.trainer_saves([0.1, 1.0, 2.0, 3.0])
        self.assertEqual([epoch for _, epoch, _ in best], [0, 1])
        for _, epoch, path in best:
            self.assertTrue(Path(path).exists())
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(loaded['epoch'], epoch)

    def test_unrelated_files_are_not_removed(self):
        unrelated = [self.root / 'other_epoch1.pth', self.root / 'other_debug_epoch1.gif',
                     self.root / 'notes.txt']
        for path in unrelated:
            path.write_bytes(b'preserve unrelated artifact')
        before = {path: path.read_bytes() for path in unrelated}
        self.manager_saves('checkpoint', pre_register=True)
        self.manager_saves('debug', pre_register=True)
        self.assertEqual({path: path.read_bytes() for path in unrelated}, before)


if __name__ == '__main__':
    unittest.main()

