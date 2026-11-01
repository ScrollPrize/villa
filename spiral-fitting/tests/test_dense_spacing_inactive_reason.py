"""dense_spacing_inactive_reason: a fit with no dense winding-spacing loss says so."""

from fit_session import dense_spacing_inactive_reason

# The inputs a fit from the public Lasagna release has to switch off.
PUBLIC = {
    'input_use_winding_inference': False,
    'input_use_outer_shell': False,
}


def test_defaults_have_a_spacing_loss():
    assert dense_spacing_inactive_reason({}) is None


def test_public_data_in_default_mode_has_none():
    reason = dense_spacing_inactive_reason(PUBLIC)
    assert reason is not None
    assert 'input_use_winding_inference and input_use_outer_shell are false' in reason
    assert "dense_spacing_mode='grad_mag'" in reason


def test_one_input_off_is_named():
    reason = dense_spacing_inactive_reason({'input_use_outer_shell': False})
    assert 'input_use_outer_shell is false' in reason
    assert 'input_use_winding_inference' not in reason


def test_gradient_magnitude_toggle_alone_does_not_help():
    # input_use_gradient_magnitude defaults to true; it is not enough.
    assert dense_spacing_inactive_reason(
        {**PUBLIC, 'input_use_gradient_magnitude': True}) is not None


def test_grad_mag_mode_is_the_fix():
    assert dense_spacing_inactive_reason(
        {**PUBLIC, 'dense_spacing_mode': 'grad_mag'}) is None


def test_grad_mag_mode_with_its_input_or_weight_off():
    assert 'input_use_gradient_magnitude is false' in dense_spacing_inactive_reason(
        {**PUBLIC, 'dense_spacing_mode': 'grad_mag',
         'input_use_gradient_magnitude': False})
    assert 'loss_weight_dense_spacing is 0' in dense_spacing_inactive_reason(
        {**PUBLIC, 'dense_spacing_mode': 'grad_mag',
         'loss_weight_dense_spacing': 0.0})


def test_invalid_mode_is_left_to_validation():
    assert dense_spacing_inactive_reason({'dense_spacing_mode': 'phase'}) is None
