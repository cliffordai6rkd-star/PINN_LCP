"""Normalization parity and preparation boundaries for control-loop use."""
import copy

import pytest
import torch

from model.pinn_model.latent_pretrained import convert_scale
from model.pinn_model.latent_runtime_normalizer import PreparedNormalizer


def envelope(mode):
    return {"normalize_mode": mode, "normalize_lowdim_keys": ["q", "tau"], "eps": 1e-4,
            "stats": {"q": {"mean": torch.tensor([0.2, -1.0]), "std": torch.tensor([0.0, 2.0]),
                             "min": torch.tensor([-2.0, 1.0]), "max": torch.tensor([3.0, 1.0]),
                             "q01": torch.tensor([-1.0, 0.3]), "q99": torch.tensor([1.0, 2.5])}}}


@pytest.mark.parametrize("mode", ["gaussian", "limit", "quantile"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_normalize_matches_existing_arithmetic_and_dtype(mode, dtype):
    source = envelope(mode)
    prepared = PreparedNormalizer(source, {"q": 2}, "cpu")
    values = torch.tensor([[[-2.0, 0.1], [0.5, 4.0]]], dtype=dtype)
    expected = convert_scale("q", values, source)
    actual = prepared.convert("q", values)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if mode == "quantile":
        assert actual.min() >= -1 and actual.max() <= 1


@pytest.mark.parametrize("mode", ["gaussian", "limit"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_inverse_preserves_statistics_dtype_promotion(mode, dtype):
    source = envelope(mode)
    prepared = PreparedNormalizer(source, {"q": 2}, "cpu")
    values = torch.tensor([[0.1, 0.4]], dtype=dtype)
    expected = convert_scale("q", values, source, inverse=True)
    actual = prepared.convert("q", values, inverse=True)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_no_normalization_and_excluded_stream_return_original_tensor():
    source = envelope(None)
    values = torch.ones(2, 2)
    assert PreparedNormalizer(source, {"q": 2}, "cpu").convert("q", values, inverse=True) is values
    source = envelope("gaussian")
    prepared = PreparedNormalizer(source, {"q": 2, "action": 2}, "cpu")
    assert prepared.convert("action", values) is values


def test_quantile_inverse_is_rejected():
    prepared = PreparedNormalizer(envelope("quantile"), {"q": 2}, "cpu")
    with pytest.raises(ValueError, match="cannot be inverted"):
        prepared.convert("q", torch.zeros(1, 2), inverse=True)


@pytest.mark.parametrize("reason", ["missing", "shape", "nonfinite", "negative", "eps", "bounds"])
def test_invalid_statistics_fail_during_preparation(reason):
    source = envelope("limit" if reason == "bounds" else "gaussian")
    if reason == "missing":
        del source["stats"]["q"]["mean"]
    elif reason == "shape":
        source["stats"]["q"]["mean"] = torch.zeros(1, 2)
    elif reason == "nonfinite":
        source["stats"]["q"]["std"][0] = float("nan")
    elif reason == "negative":
        source["stats"]["q"]["std"][0] = -1
    elif reason == "eps":
        source["eps"] = 0
    else:
        source["stats"]["q"]["max"][0] = -3
    with pytest.raises(ValueError):
        PreparedNormalizer(source, {"q": 2}, "cpu")


def test_prepared_snapshot_is_independent_and_validated_once(monkeypatch):
    source = envelope("gaussian")
    reference = copy.deepcopy(source)
    prepared = PreparedNormalizer(source, {"q": 2}, "cpu")
    source["stats"]["q"]["mean"].fill_(float("nan"))
    source["stats"]["q"]["std"].zero_()
    source["eps"] = 10
    source["normalize_lowdim_keys"].clear()
    def reject_validation(*args, **kwargs):
        raise AssertionError("conversion must not revalidate statistics")
    monkeypatch.setattr("model.pinn_model.latent_runtime_normalizer.validate_normalizer", reject_validation)
    values = torch.tensor([[0.4, 0.7]])
    torch.testing.assert_close(prepared.convert("q", values), convert_scale("q", values, reference), rtol=0, atol=0)
    torch.testing.assert_close(prepared.convert("q", values, inverse=True),
                               convert_scale("q", values, reference, inverse=True), rtol=0, atol=0)


def test_device_and_shape_checks_precede_arithmetic():
    prepared = PreparedNormalizer(envelope("gaussian"), {"q": 2}, "cpu")
    with pytest.raises(ValueError, match="last dimension"):
        prepared.convert("q", torch.zeros(2, 3))
    with pytest.raises(ValueError, match="prepared device"):
        prepared.convert("q", torch.empty(2, 2, device="meta"))
    with pytest.raises(ValueError, match="unknown prepared"):
        prepared.convert("action", torch.zeros(2, 2))


def test_prepared_conversion_never_calls_tensor_to(monkeypatch):
    prepared = PreparedNormalizer(envelope("gaussian"), {"q": 2}, "cpu")
    values = torch.tensor([[0.4, 0.7]], dtype=torch.float16)
    expected = prepared.convert("q", values)
    expected_inverse = prepared.convert("q", values, inverse=True)
    def reject_transfer(*args, **kwargs):
        raise AssertionError("prepared conversion must not call Tensor.to")
    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "to", reject_transfer)
        actual = prepared.convert("q", values)
        actual_inverse = prepared.convert("q", values, inverse=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_inverse, expected_inverse, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_conversion_uses_device_constants_without_tensor_to(monkeypatch):
    source = envelope("gaussian")
    prepared = PreparedNormalizer(source, {"q": 2}, "cuda")
    values = torch.tensor([[0.4, 0.7]], device="cuda")
    expected = convert_scale("q", values, source)
    def reject_transfer(*args, **kwargs):
        raise AssertionError("prepared conversion must not call Tensor.to")
    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "to", reject_transfer)
        actual = prepared.convert("q", values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
