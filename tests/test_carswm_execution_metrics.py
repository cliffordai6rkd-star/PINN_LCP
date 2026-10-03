import pytest
import torch
from train.carswm_execution_metrics import (execution_slice, local_interval_loss, physical_derivatives,
                                           ContactAccumulator, adjacent_prediction_metrics)
from train.nomalizer import Normalizer


def test_interval_endpoint_is_h_squared_velocity_and_same_gradient_scale():
    u = torch.randn(2, 8, 3, requires_grad=True)
    x, y = torch.randn_like(u), torch.randn_like(u)
    h = .25
    mse = lambda a, b: (a - b).square().mean()
    velocity = local_interval_loss(u, x, y, h, 'velocity', mse)
    endpoint = local_interval_loss(u, x, y, h, 'endpoint', mse)
    torch.testing.assert_close(endpoint, velocity * h ** 2)
    gv, = torch.autograd.grad(velocity, u, retain_graph=True)
    ge, = torch.autograd.grad(endpoint, u)
    torch.testing.assert_close(ge, gv * h ** 2)


def test_execution_indices_equal_nero_prefetch_and_openloop_boundaries():
    values = torch.arange(40)
    assert values[execution_slice(40, 32, 8)].tolist() == list(range(32, 40))
    assert values[execution_slice(40, 32, 8, 'openloop')].tolist() == list(range(8))
    with pytest.raises(ValueError):
        execution_slice(40, 33, 8)
    with pytest.raises(ValueError):
        execution_slice(40, -1, 8)


def test_physical_units_with_nonuniform_dt_and_denormalization():
    time = torch.tensor([[.01, .02, .04, .07]])
    q = time[:, None, :, None].square()  # q=t^2 rad, acceleration=2 rad/s^2
    acceleration = physical_derivatives(q, time, 2)
    torch.testing.assert_close(acceleration, torch.full_like(acceleration, 2.), atol=1e-5, rtol=1e-5)
    n = Normalizer({'tau': {'mean': torch.tensor([5.]), 'std': torch.tensor([2.])}})
    n.validate('gaussian', ['tau'], {'tau': 1})
    torch.testing.assert_close(n.gaussian_denormalize('tau', torch.tensor([1.])), torch.tensor([7.000001]))
    n.stats['tau']['std'][0] = float('nan')
    with pytest.raises(ValueError, match='invalid normalizer'):
        n.validate('gaussian', ['tau'], {'tau': 1})


def test_contact_uses_noise_marginal_for_nll_not_mean_conditional_nll():
    p = torch.tensor([[[[.9, .1]], [[.1, .9]]]])
    y = torch.zeros(1, 1, 1)
    acc = ContactAccumulator(2)
    acc.update(p, y, torch.tensor([[.01]]))
    metrics = acc.finalize()
    assert metrics['nll'] == pytest.approx(-torch.tensor(.5).log().item())
    assert metrics['brier'] == pytest.approx(.5)
    assert metrics['ece'] == pytest.approx(.5)
    assert metrics['macro_f1'] == pytest.approx(.5)


def test_contact_transition_first_frame_and_missing_event_reported():
    labels = torch.tensor([[[2.], [2.], [0.], [0.]]])
    p = torch.nn.functional.one_hot(labels[..., 0].long(), 3).float()[:, None]
    acc = ContactAccumulator(3)
    acc.update(p, labels, torch.tensor([[.01, .02, .03, .04]]), torch.zeros(1, 2, 1))
    result = acc.finalize()
    assert result['onset_matched'] == result['release_matched'] == 1
    assert result['onset_matched_mae_s'] == result['release_matched_mae_s'] == 0
    no_contact = torch.zeros_like(p)
    no_contact[..., 0] = 1
    acc = ContactAccumulator(3)
    acc.update(no_contact, labels, torch.tensor([[.01, .02, .03, .04]]), torch.zeros(1, 2, 1))
    assert acc.finalize()['onset_missing'] == 1
    assert 'onset_matched_mae_s' not in acc.finalize()


def test_adjacent_same_absolute_time_not_same_array_index():
    old = torch.arange(10.).reshape(1, 1, 10, 1)
    new = old + 3
    result = adjacent_prediction_metrics(old, new, 3, 2)
    assert result['overlap_q_mse_rad2'].item() == 0
    assert result['takeover_q_mse_rad2'].item() == 0
    with pytest.raises(ValueError):
        adjacent_prediction_metrics(old, new, 3, 7)
