"""Tests for ConfigManager target name validation."""

from pathlib import Path
import tempfile
import pytest
import yaml

from vesuvius.models.configuration.config_manager import ConfigManager


class TestTargetNameValidation:
    """Test suite for target name validation in ConfigManager."""

    def test_valid_names(self):
        """Test that valid target names pass validation."""
        mgr = ConfigManager(verbose=False)
        valid_names = ['ink', 'fiber', 'papyrus', 'damage', 'my_target']

        # Should not raise ValueError
        mgr.validate_target_names(valid_names)

    @pytest.mark.parametrize("reserved_name", ['mask', 'is_unlabeled', 'plane_mask'])
    def test_reserved_names(self, reserved_name):
        """Test that reserved names raise ValueError."""
        mgr = ConfigManager(verbose=False)

        with pytest.raises(ValueError, match=f"Target name '{reserved_name}' is reserved"):
            mgr.validate_target_names([reserved_name])

    def test_mixed_names(self):
        """Test that a mix with one reserved name fails."""
        mgr = ConfigManager(verbose=False)
        mixed_names = ['ink', 'mask', 'fiber']

        with pytest.raises(ValueError, match="Target name 'mask' is reserved"):
            mgr.validate_target_names(mixed_names)

    def test_config_loading_validation(self):
        """Test validation during config loading with reserved target name."""
        config = {
            "dataset_config": {
                "targets": {
                    "mask": {"out_channels": 2}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            mgr = ConfigManager(verbose=False)
            with pytest.raises(ValueError, match="Target name 'mask' is reserved"):
                mgr.load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_config_loading_validation_model_config(self):
        """Test validation when targets are in model_config instead of dataset_config."""
        config = {
            "model_config": {
                "targets": {
                    "is_unlabeled": {"out_channels": 1}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            mgr = ConfigManager(verbose=False)
            with pytest.raises(ValueError, match="Target name 'is_unlabeled' is reserved"):
                mgr.load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_set_targets_validation(self):
        """Test validation in set_targets_and_data."""
        mgr = ConfigManager(verbose=False)
        targets_dict = {
            "is_unlabeled": {"out_channels": 1}
        }
        data_dict = {}

        with pytest.raises(ValueError, match="Target name 'is_unlabeled' is reserved"):
            mgr.set_targets_and_data(targets_dict, data_dict)

    def test_empty_target_list(self):
        """Test that empty target list doesn't raise error."""
        mgr = ConfigManager(verbose=False)
        # Should not raise ValueError
        mgr.validate_target_names([])

    def test_multiple_reserved_names(self):
        """Test error message when multiple reserved names are used."""
        mgr = ConfigManager(verbose=False)
        names_with_reserved = ['mask', 'plane_mask', 'ink']

        # Should raise for the first reserved name encountered
        with pytest.raises(ValueError, match="is reserved"):
            mgr.validate_target_names(names_with_reserved)
