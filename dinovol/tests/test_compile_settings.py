import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch._dynamo

from dinovol_2.methods.dino import DinoMethod


def method_with(compile_config):
    branch = {name: MagicMock() for name in ("backbone", "dino_head", "ibot_head")}
    stub = SimpleNamespace(compile_config=compile_config, gram_teacher_backbone=None,
                           model_module=SimpleNamespace(student=branch, teacher=branch),
                           ibot_patch_loss=MagicMock())
    return stub


class CompileSettingsTests(unittest.TestCase):
    def setUp(self):
        for name in ("optimize_ddp", "compiled_autograd"):
            self.addCleanup(setattr, torch._dynamo.config, name, getattr(torch._dynamo.config, name))

    def test_ddp_graph_splitting_is_off_unless_requested(self):
        for config, want in (({"enabled": True}, False), ({"enabled": True, "optimize_ddp": True}, True)):
            torch._dynamo.config.optimize_ddp = not want
            stub = method_with(config)
            DinoMethod._configure_model_compilation(stub)
            self.assertIs(torch._dynamo.config.optimize_ddp, want)
            stub.model_module.student["backbone"].compile.assert_called()

    def test_python_reducer_enables_compiled_autograd(self):
        torch._dynamo.config.compiled_autograd = False
        DinoMethod._configure_model_compilation(method_with({"enabled": True, "optimize_ddp": "python_reducer"}))
        self.assertEqual(torch._dynamo.config.optimize_ddp, "python_reducer")
        self.assertTrue(torch._dynamo.config.compiled_autograd)

    def test_disabled_compile_and_bad_values(self):
        torch._dynamo.config.optimize_ddp = True
        DinoMethod._configure_model_compilation(method_with({"enabled": False, "optimize_ddp": False}))
        self.assertIs(torch._dynamo.config.optimize_ddp, True)
        with self.assertRaises(ValueError):
            DinoMethod._configure_model_compilation(method_with({"enabled": True, "optimize_ddp": "no"}))


if __name__ == "__main__":
    unittest.main()
