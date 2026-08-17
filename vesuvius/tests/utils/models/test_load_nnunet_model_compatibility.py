from types import SimpleNamespace

from vesuvius.utils.models import load_nnunet_model
from vesuvius.utils.models.load_nnunet_model import (
    _build_network_from_trainer,
    initialize_network,
)


def configuration():
    return SimpleNamespace(
        network_arch_class_name='example.Network',
        network_arch_init_kwargs={'depth': 3},
        network_arch_init_kwargs_req_import=['conv_op'],
    )


def test_build_network_uses_legacy_nnunet_trainer_signature():
    calls = []
    network = object()

    class LegacyTrainer:
        @staticmethod
        def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
        ):
            calls.append((
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision,
            ))
            return network

    result = _build_network_from_trainer(
        LegacyTrainer,
        plans_manager=object(),
        configuration_manager=configuration(),
        num_input_channels=1,
        num_output_channels=2,
    )

    assert result is network
    assert calls == [(
        'example.Network', {'depth': 3}, ['conv_op'], 1, 2, False,
    )]


def test_build_network_uses_current_nnunet_trainer_signature():
    calls = []
    network = object()
    plans_manager = object()
    configuration_manager = configuration()

    class CurrentTrainer:
        @staticmethod
        def build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
        ):
            calls.append((
                plans_manager,
                configuration_manager,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision,
            ))
            return network

    result = _build_network_from_trainer(
        CurrentTrainer,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        num_input_channels=1,
        num_output_channels=2,
    )

    assert result is network
    assert calls == [(
        plans_manager, configuration_manager, 1, 2, False,
    )]


def test_initialize_network_uses_nnunet_factory(monkeypatch):
    calls = []
    network = object()

    def fake_get_network_from_plans(*args, **kwargs):
        calls.append((args, kwargs))
        return network

    monkeypatch.setattr(
        load_nnunet_model,
        'get_network_from_plans',
        fake_get_network_from_plans,
    )

    result = initialize_network(
        'example.Network',
        {'conv_op': 'torch.nn.Conv3d'},
        ['conv_op'],
        num_input_channels=1,
        num_output_channels=2,
        enable_deep_supervision=False,
    )

    assert result is network
    assert calls == [(
        (
            'example.Network',
            {'conv_op': 'torch.nn.Conv3d'},
            ['conv_op'],
            1,
            2,
        ),
        {'allow_init': True, 'deep_supervision': False},
    )]
