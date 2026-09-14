import pytest
import torch

from model.pinn_model.contact_world_model import (
    ContactWorldModel,
    FlowTimeEmbedding,
    FlowDecoderBlock,
    PREDICTED_STATE_STREAMS,
    SUPPORTED_STATE_STREAMS,
)
from train.contact_world_model_loss import ContactWorldModelLoss


def config(inputs=None, outputs=None):
    result = {
        "dataloader": {"state_history_horizon": 5, "prediction_horizon": 4, "action_condition_horizon": 3, "high_fps": 100, "normalize_mode": None},
        "model": {"inputs": list(inputs or SUPPORTED_STATE_STREAMS), "joint_dim": 2, "action_dim": 2, "contact_state_count": 3, "hidden_dim": 8, "state_layers": 1, "action_layers": 1, "flow_layers": 1, "flow_attention_heads": 2, "flow_ffn_multiplier": 2, "flow_inference_steps": 2, "flow_solver": "heun", "flow_source_mode": "gaussian", "dropout": 0.0},
        "loss": {"dt": 0.01, "kinematic_consistency_weight": 0.01, "ddq_smoothness_weight": 0.01},
    }
    if outputs is not None:
        result["model"]["outputs"] = list(outputs)
    return result


def batch(cfg):
    data = cfg['dataloader']
    b, h, f, d, a = (2, data['state_history_horizon'], data['prediction_horizon'],
                      cfg['model']['joint_dim'], data['action_condition_horizon'])
    out = {key: torch.randn(b, h, d) for key in cfg["model"]["inputs"]}
    out.update({f"{key}_future": torch.randn(b, f, d) for key in PREDICTED_STATE_STREAMS})
    out.update(action=torch.randn(b, a, d), action_mask=torch.ones(b, a, dtype=torch.bool), contact_future=torch.randint(0, 3, (b, f, 1)).float())
    return out


@pytest.mark.parametrize(
    "inputs,count",
    [
        (["q", "dq", "delta_q", "tau"], 4),
        (["q", "delta_q", "tau"], 3),
        (["q", "tau"], 2),
    ],
)
def test_independent_state_encoders_and_fused_shapes(inputs, count):
    cfg = config(inputs)
    model = ContactWorldModel(cfg)
    assert len(model.state_encoders) == count
    assert tuple(model.state_encoders) == tuple(inputs)
    assert model.state_encoders[inputs[0]] is not model.state_encoders[inputs[-1]]
    first = {id(parameter) for parameter in model.state_encoders[inputs[0]].parameters()}
    assert first.isdisjoint(id(parameter) for parameter in model.state_encoders[inputs[-1]].parameters())
    output = model(batch(cfg), flow_time=0.5)
    assert output["state_tokens"].shape == (2, count * model.history_horizon, 8)
    assert output["action_tokens"].shape == (2, 3, 8)
    assert "condition_memory" not in output
    assert "state_action_tokens" not in output
    assert not hasattr(model, "state_to_action_attention")
    assert "action_time" not in output
    assert "future_time" not in output


def test_flow_block_uses_self_attention_cross_attention_and_ffn():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)
    (output["flow_velocity_pred"] - output["flow_velocity_target"]).square().mean().backward()
    block = model.flow_blocks[0]
    assert any(parameter.grad is not None for parameter in block.self_attention.parameters())
    for branch in (block.history_cross_attn, block.action_cross_attn):
        assert all(parameter.grad is not None and torch.any(parameter.grad != 0)
                   for parameter in branch.parameters())
    assert any(parameter.grad is not None for parameter in block.ffn.parameters())
    assert any(
        parameter.grad is not None
        for encoder in model.state_encoders.values()
        for parameter in encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.action_encoder.parameters())


@pytest.mark.parametrize("inputs", [["q"], ["q", "dq", "delta_q", "tau"]])
@pytest.mark.parametrize("history,downsample", [(7, False), (10, False), (50, True)])
def test_all_temporal_gru_outputs_and_modality_embeddings_reach_loss(inputs, history, downsample):
    cfg = config(inputs)
    cfg['dataloader'].update(state_history_horizon=history, action_condition_horizon=8)
    cfg['train'] = {'downsample': downsample}
    # Single-stream losses do not use cross-stream physics regularizers.
    cfg['loss'].update(kinematic_consistency_weight=0.0, ddq_smoothness_weight=0.0)
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    sequences = {}
    before_norm = []
    def capture_sequence(key):
        def hook(module, args, output):
            output[0].retain_grad()
            sequences[key] = output[0]
        return hook
    handles = [model.state_encoders[key].register_forward_hook(capture_sequence(key))
               for key in inputs]
    handles.append(model.state_token_norm.register_forward_pre_hook(
        lambda module, args: before_norm.append(args[0])))
    output = model(values, flow_time=0.5)
    for handle in handles:
        handle.remove()
    length = model.history_horizon
    for i, key in enumerate(inputs):
        assert sequences[key].shape == (2, length, model.hidden_dim)
        actual = before_norm[0][:, i*length:(i+1)*length]
        expected = sequences[key] + model.modality_embeddings[key][None, None, :]
        torch.testing.assert_close(actual, expected)
    assert output['state_tokens'].shape == (2, len(inputs)*length, model.hidden_dim)
    assert output['action_tokens'].shape == (2, 8, model.hidden_dim)
    assert output['action_padding_mask'].shape == (2, 8)
    assert not any('pool' in name for name, _ in model.named_modules())
    assert not any('pool' in name for name, _ in model.named_parameters())
    assert not hasattr(model, 'state_pooling')
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    loss.backward()
    for sequence in sequences.values():
        assert sequence.grad is not None
        assert torch.all(sequence.grad.abs().sum(dim=-1) > 0)


@pytest.mark.parametrize('value', ['last', 'attention'])
def test_removed_state_pooling_config_is_rejected(value):
    cfg = config()
    cfg['model']['state_pooling'] = value
    with pytest.raises(ValueError, match='state_pooling was removed'):
        ContactWorldModel(cfg)


@pytest.mark.parametrize('name,history_tokens', [
    ('cwm_insert_usb_50hz', 100), ('contact_world_model_npu', 100), ('cwm_all_50hz', None),
])
def test_training_configs_temporal_history_tokens(name, history_tokens):
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/train_cfg' / (name+'.yaml')).read_text())
    model = ContactWorldModel(cfg).eval()
    if history_tokens is None:
        history_tokens = len(model.inputs) * model.history_horizon
    values = {key: torch.randn(1, model.external_history_horizon, model.joint_dim)
              for key in model.inputs}
    values['action'] = torch.randn(1, 8, model.action_dim)
    encoded = model.encode_conditions(values)
    assert encoded['state_tokens'].shape == (1, history_tokens, 128)
    assert encoded['action_tokens'].shape == (1, 8, 128)
    assert 'condition_memory' not in encoded
    assert model.action_pos_embedding.num_embeddings == 8
    assert model.future_pos_embedding.num_embeddings == model.future_horizon
    seen = []
    def capture(module, args):
        seen.append((args[0].shape, args[1].shape, args[2].shape))
    handles = [block.register_forward_pre_hook(capture) for block in model.flow_blocks]
    model.flow_velocity(torch.randn(1, model.future_horizon, model.flow_dim),
                        torch.full((1, 1), 0.5), encoded)
    for handle in handles:
        handle.remove()
    assert len(seen) == model.flow_layers
    assert all(shapes == ((1, model.future_horizon, 128),
                          (1, history_tokens, 128), (1, 8, 128)) for shapes in seen)


def test_checkpoint_contract_identifies_simplified_token_architecture():
    model = ContactWorldModel(config())
    contract = model.checkpoint_contract()
    assert model.MODEL_VERSION == "carswm_v6"
    assert contract["schema_version"] == 7
    assert contract["architecture"] == {
        "condition_encoder": "modality_gru_action_gru",
        "state_token": "all_gru_temporal_outputs_modality_major",
        "flow_decoder": "self_attention_parallel_dual_cross_attention_residual_sum_ffn",
        "condition_memories": "independent_history_and_action",
        "action_position_encoding": "learned_sequence_index",
        "future_position_encoding": "learned_sequence_index",
    }
    assert contract["action"]["dataset_alignment"] == "previous"
    incompatible = dict(contract)
    incompatible["model_version"] = "carswm_v5"
    with pytest.raises(ValueError, match="contract mismatch"):
        model.validate_checkpoint_contract(incompatible)


def test_flow_targets_and_contact_are_separate():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)
    assert output["flow_velocity_pred"].shape == (2, 4, 8)
    assert output["flow_target_state"].shape[-1] == 8
    assert output["flow_source_state"].shape[-1] == 8
    assert output["contact_logits"].shape == (2, 4, 3)
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    loss.backward()
    assert "flow_contact_loss" not in metrics
    assert "contact_loss" in metrics
    assert any(parameter.grad is not None for parameter in model.contact_head.parameters())


@pytest.mark.parametrize("future,stride", [(16, False), (16, True), (32, True), (18, 3)])
def test_learned_positions_use_action_and_internal_future_indices(future, stride):
    cfg = config()
    cfg['dataloader'].update(state_history_horizon=6, prediction_horizon=future,
                             action_condition_horizon=8)
    cfg['train'] = {'downsample': stride}
    model = ContactWorldModel(cfg).eval()
    values = batch(cfg)
    seen = {}
    def capture(name):
        def hook(module, args, output):
            seen[name] = args[0].detach().clone()
        return hook
    handles = [model.action_pos_embedding.register_forward_hook(capture('action')),
               model.future_pos_embedding.register_forward_hook(capture('future'))]
    output = model(values, flow_time=0.4)
    for handle in handles:
        handle.remove()
    assert isinstance(model.action_pos_embedding, torch.nn.Embedding)
    assert isinstance(model.future_pos_embedding, torch.nn.Embedding)
    torch.testing.assert_close(seen['action'], torch.arange(8))
    torch.testing.assert_close(seen['future'], torch.arange(model.future_horizon))
    assert model.future_pos_embedding.num_embeddings == future // model.temporal_stride
    assert output['action_tokens'].shape == (2, 8, 8)
    output['flow_velocity_pred'].square().mean().backward()
    for embedding in (model.action_pos_embedding, model.future_pos_embedding):
        assert embedding.weight.grad is not None
        assert torch.all(embedding.weight.grad.abs().sum(dim=1) > 0)


def test_timestamp_metadata_jitter_cannot_change_network_outputs():
    cfg = config()
    cfg['dataloader'].update(state_history_horizon=6, prediction_horizon=16,
                             action_condition_horizon=8)
    cfg['train'] = {'downsample': True}
    model = ContactWorldModel(cfg).eval()
    values = batch(cfg)
    noise = torch.randn(2, model.future_horizon, model.flow_dim)
    first = model(values, flow_time=0.4, source_noise=noise)
    jittered = dict(values)
    for key, length in [('action_time', 8), ('future_time', 16),
                        ('action_chunk_timestamp_ns', 8), ('future_timestamp_ns', 16),
                        ('history_timestamp_ns', 6)]:
        jittered[key] = torch.randn(2, length) * 1e6
    second = model(jittered, flow_time=0.4, source_noise=noise)
    for key in ('action_tokens', 'state_tokens', 'condition_summary',
                'flow_velocity_pred', 'contact_logits'):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert 'action_time' not in model.CONDITION_KEYS
    assert 'future_time' not in model.CONDITION_KEYS
    assert 'action_time' not in second and 'future_time' not in second


def test_flow_matching_time_embedding_still_conditions_velocity():
    model = ContactWorldModel(config()).eval()
    assert isinstance(model.flow_time_embedding, FlowTimeEmbedding)
    encoded = model.encode_conditions(batch(config()))
    trajectory = torch.randn(2, model.future_horizon, model.flow_dim)
    times = torch.tensor([[0.2], [0.7]], requires_grad=True)
    seen = []
    handle = model.flow_time_embedding.register_forward_pre_hook(
        lambda module, args: seen.append(args[0]))
    velocity, _ = model.flow_velocity(trajectory, times, encoded)
    handle.remove()
    assert seen[0] is times
    changed, _ = model.flow_velocity(trajectory, 1-times, encoded)
    assert not torch.equal(velocity, changed)
    velocity.square().mean().backward()
    assert times.grad is not None and torch.all(times.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in model.flow_time_embedding.parameters())


@pytest.mark.parametrize('name', ['action_time_alignment', 'use_physical_time',
                                 'action_time_encoding', 'future_time_encoding'])
def test_removed_time_encoding_config_is_rejected(name):
    cfg = config()
    cfg['model'][name] = 'legacy'
    with pytest.raises(ValueError, match='Removed model time-encoding options'):
        ContactWorldModel(cfg)


def test_contact_feature_alignment_preserves_nominal_zoh_indices():
    cfg = config()
    cfg['dataloader'].update(prediction_horizon=16, action_condition_horizon=8)
    model = ContactWorldModel(cfg)
    encoded = {'action_tokens': torch.arange(8).float()[None, :, None]}
    aligned = model._aligned_action_features(encoded)
    # At 100 Hz state / 25 Hz action, offset=1 holds token zero until 80 ms.
    expected = torch.tensor([0]*7 + [1]*4 + [2]*4 + [3]).float()[None, :, None]
    torch.testing.assert_close(aligned, expected)


def test_student_uses_same_discrete_future_positions():
    from model.pinn_model.contact_world_model_student import ContactWorldModelStudent
    teacher = ContactWorldModel(config()).eval()
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=2).eval()
    encoded = student.encode_conditions(batch(config()))
    trajectory = torch.randn(2, student.future_horizon, student.flow_dim)
    velocity, _ = student.flow_velocity_student(
        trajectory, torch.full((2, 1), 0.3), torch.full((2, 1), 0.5), encoded)
    velocity.square().mean().backward()
    assert student.future_pos_embedding.weight.grad.abs().sum() > 0
    torch.testing.assert_close(student.future_pos_embedding.weight,
                               teacher.future_pos_embedding.weight)


def test_contact_head_depends_on_its_generated_continuous_sample():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    encoded = model.encode_conditions(values)
    continuous = torch.randn(2, 4, model.flow_dim, requires_grad=True)
    model.contact_logits(continuous, encoded).square().mean().backward()
    assert continuous.grad is not None
    assert torch.any(continuous.grad != 0)


def test_contact_head_uses_configured_phase_count():
    cfg = config()
    cfg["model"]["contact_state_count"] = 4
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values["contact_future"] = torch.randint(0, 4, values["contact_future"].shape).float()
    output = model(values, flow_time=0.5)
    assert output["contact_logits"].shape == (2, 4, 4)
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)


def test_contact_target_outside_configured_phase_count_is_rejected():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values["contact_future"][0, 0, 0] = 3.0
    output = model(values, flow_time=0.5)
    with pytest.raises(ValueError, match="outside model.contact_state_count"):
        ContactWorldModelLoss(cfg)(output, values)


def test_different_source_noise_generates_different_futures():
    cfg = config()
    model = ContactWorldModel(cfg).eval()
    values = batch(cfg)
    first = model.predict(values, source_noise=torch.zeros(2, 4, 8))
    second = model.predict(values, source_noise=torch.ones(2, 4, 8))
    assert not torch.equal(first["flow_state_pred"], second["flow_state_pred"])
    assert first["contact_logits"].shape == (2, 4, 3)


def test_missing_selected_state_is_rejected_without_zero_fill():
    cfg = config(["q", "tau"])
    values = batch(cfg)
    del values["tau"]
    with pytest.raises(KeyError, match="tau"):
        ContactWorldModel(cfg)(values)


def test_selected_streams_define_continuous_output_contract():
    cfg = config(["q", "delta_q", "tau"])
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values.pop("dq", None)
    values.pop("dq_future", None)
    output = model(values, flow_time=0.5)
    assert model.predicted_state_streams == ("q", "delta_q", "tau")
    assert model.PREDICTED_STATE_STREAMS == model.predicted_state_streams
    assert model.TARGET_KEYS == ("q_future", "delta_q_future", "tau_future", "contact_future")
    assert model.flow_dim == 3 * 2
    assert output["flow_velocity_pred"].shape == (2, 4, 6)
    assert all(f"{key}_pred" in output for key in ("q", "delta_q", "tau"))
    assert "dq_pred" not in output
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)
    assert "dq_loss" not in metrics


def test_teacher_inputs_and_outputs_are_independent():
    cfg = config(
        inputs=["q", "dq", "delta_q", "tau"],
        outputs=["q", "tau"],
    )
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values.pop("dq_future")
    values.pop("delta_q_future")

    output = model(values, flow_time=0.5)

    assert tuple(model.state_encoders) == ("q", "dq", "delta_q", "tau")
    assert model.outputs == ("q", "tau")
    assert model.predicted_state_streams == ("q", "tau")
    assert model.PREDICTED_STATE_STREAMS == ("q", "tau")
    assert model.CONDITION_KEYS == (
        "q", "dq", "delta_q", "tau", "action", "action_mask",
    )
    assert model.TARGET_KEYS == ("q_future", "tau_future", "contact_future")
    assert model.flow_dim == 2 * model.joint_dim
    assert output["flow_velocity_pred"].shape == (2, 4, 4)
    assert "q_pred" in output and "tau_pred" in output
    assert "dq_pred" not in output and "delta_q_pred" not in output
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)
    assert "q_loss" in metrics and "tau_loss" in metrics
    assert "dq_loss" not in metrics and "delta_q_loss" not in metrics

    contract = model.checkpoint_contract()
    assert contract["input_state_streams"] == ["q", "dq", "delta_q", "tau"]
    assert contract["predicted_continuous_streams"] == ["q", "tau"]


def test_teacher_can_predict_a_stream_not_used_as_history():
    cfg = config(inputs=["q", "dq"], outputs=["q", "tau"])
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)

    assert tuple(model.state_encoders) == ("q", "dq")
    assert model.predicted_state_streams == ("q", "tau")
    assert "tau" not in values
    assert output["tau_pred"].shape == (2, 4, 2)
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "outputs,match",
    [
        ([], "at least one"),
        (["q", "q"], "duplicates"),
        (["q", "temperature"], "unsupported"),
    ],
)
def test_invalid_teacher_outputs_are_rejected(outputs, match):
    cfg = config(outputs=outputs)
    with pytest.raises(ValueError, match=match):
        ContactWorldModel(cfg)
    with pytest.raises(ValueError, match=match):
        ContactWorldModelLoss(cfg)


def test_null_outputs_fall_back_to_inputs():
    cfg = config(inputs=["q", "tau"])
    cfg["model"]["outputs"] = None
    model = ContactWorldModel(cfg)
    calculator = ContactWorldModelLoss(cfg)

    assert model.outputs == ("q", "tau")
    assert calculator.predicted_state_streams == ("q", "tau")


def test_disabled_cross_stream_regularizers_allow_reduced_outputs():
    cfg = config(inputs=["q", "dq", "delta_q", "tau"], outputs=["tau"])
    cfg["loss"]["kinematic_consistency_weight"] = 0.0
    cfg["loss"]["ddq_smoothness_weight"] = 0.0
    values = batch(cfg)
    values.pop("q_future")
    values.pop("dq_future")
    values.pop("delta_q_future")
    model = ContactWorldModel(cfg)

    loss, _ = ContactWorldModelLoss(cfg)(model(values, flow_time=0.5), values)

    assert torch.isfinite(loss)


def test_contract_allows_ablation_without_q():
    cfg = config(["dq", "tau"])
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values.pop("q", None)
    values.pop("q_future", None)
    output = model(values, flow_time=0.5)
    assert output["flow_velocity_pred"].shape[-1] == 2 * 2
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)


def test_endpoint_schedule_and_delta_q_contract():
    cfg = config()
    calculator = ContactWorldModelLoss(cfg)
    calculator.set_global_step(0, 100)
    assert calculator.endpoint_weight == pytest.approx(0.1)
    calculator.set_global_step(15, 100)
    assert calculator.endpoint_weight == pytest.approx(0.05)
    calculator.set_global_step(30, 100)
    assert calculator.endpoint_weight == pytest.approx(0.0)
    assert calculator.delta_q_consistency_weight == 0.0


def test_kinematic_loss_is_zero_for_trapezoidal_integration():
    cfg = config()
    calculator = ContactWorldModelLoss(cfg)
    values = batch(cfg)
    values["q"][:] = 0.0
    values["dq"][:] = 1.0
    q = torch.arange(1, 5, dtype=torch.float32)[None, :, None].repeat(2, 1, 2) * 0.01
    out = {"q_pred": q, "dq_pred": torch.ones_like(q)}
    assert torch.max(calculator._kinematic_consistency(out, values)) < 1.0e-8


@pytest.mark.parametrize('stride', [True, 3])
def test_action_horizon_need_not_be_divisible_by_state_stride(stride):
    cfg = config()
    cfg['train'] = {'downsample': stride}
    cfg['dataloader'].update(state_history_horizon=6, prediction_horizon=6,
                             action_condition_horizon=5, expert_fps=25)
    model = ContactWorldModel(cfg)
    assert model.action_condition_horizon == 5
    assert model.action_rate_hz == 25
    values = batch(cfg)
    for key in cfg['model']['inputs']:
        values[key] = torch.randn(2, 6, 2)
    for key in PREDICTED_STATE_STREAMS:
        values[f'{key}_future'] = torch.randn(2, 6, 2)
    values['contact_future'] = torch.zeros(2, 6, 1)
    values['action'] = torch.randn(2, 5, 2)
    values['action_mask'] = torch.ones(2, 5, dtype=torch.bool)
    encoded = model.encode_conditions(values)
    assert encoded['action_tokens'].shape == (2, 5, 8)
    assert encoded['_prepared_batch']['q'].shape[1] == 6 // model.temporal_stride


def test_forward_adds_discrete_future_pe_and_preserves_flow_interpolation():
    cfg = config()
    model = ContactWorldModel(cfg).eval()
    values = batch(cfg)
    noise = torch.randn(2, model.future_horizon, model.flow_dim)
    seen = {}
    def capture(module, args):
        seen['features'] = args[0]
    handle = model.flow_blocks[0].register_forward_pre_hook(capture)
    output = model(values, flow_time=0.3, source_noise=noise)
    handle.remove()
    interpolated = 0.7 * noise + 0.3 * output['flow_target_state']
    torch.testing.assert_close(output['flow_interpolated'], interpolated)
    expected = (model.flow_input_projection(interpolated)
                + model.future_pos_embedding(torch.arange(model.future_horizon))[None]
                + model.flow_time_embedding(output['flow_time'])[:, None])
    torch.testing.assert_close(seen['features'], expected)


def test_dual_branches_use_parallel_queries_separate_memories_and_residual_sum():
    block = FlowDecoderBlock(8, 2, 2, 0.0).eval()
    history_parameters = {id(p) for p in block.history_cross_attn.parameters()}
    assert history_parameters.isdisjoint(id(p) for p in block.action_cross_attn.parameters())
    assert block.history_norm is not block.action_norm
    trajectory = torch.randn(2, 4, 8)
    history = torch.randn(2, 100, 8)
    action = torch.randn(2, 8, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, -1] = True
    seen = {}
    def norm_hook(name):
        def hook(module, args):
            seen[name] = args[0]
        return hook
    def attention_hook(name):
        def hook(module, args, kwargs, output):
            seen[name] = (kwargs, output[0])
        return hook
    handles = [block.history_norm.register_forward_pre_hook(norm_hook('history_z')),
               block.action_norm.register_forward_pre_hook(norm_hook('action_z')),
               block.ffn_norm.register_forward_pre_hook(norm_hook('fused')),
               block.self_attention.register_forward_hook(attention_hook('self'), with_kwargs=True),
               block.history_cross_attn.register_forward_hook(attention_hook('history'), with_kwargs=True),
               block.action_cross_attn.register_forward_hook(attention_hook('action'), with_kwargs=True)]
    output = block(trajectory, history, action, mask)
    for handle in handles:
        handle.remove()
    assert seen['history_z'] is seen['action_z']
    z = trajectory + seen['self'][1]
    torch.testing.assert_close(seen['history_z'], z)
    for branch, memory, norm in [('history', history, block.history_norm),
                                 ('action', action, block.action_norm)]:
        kwargs = seen[branch][0]
        torch.testing.assert_close(kwargs['query'], norm(z))
        assert kwargs['key'] is memory and kwargs['value'] is memory
        assert kwargs.get('attn_mask') is None
    assert seen['history'][0].get('key_padding_mask') is None
    assert seen['action'][0]['key_padding_mask'] is mask
    fused = z + seen['history'][1] + seen['action'][1]
    torch.testing.assert_close(seen['fused'], fused)
    torch.testing.assert_close(output, fused + block.ffn(block.ffn_norm(fused)))


def test_action_attention_normalizes_independently_of_history_length():
    block = FlowDecoderBlock(8, 2, 2, 0.0).eval()
    trajectory, history, action = torch.randn(1, 4, 8), torch.randn(1, 100, 8), torch.randn(1, 8, 8)
    outputs = []
    handle = block.action_cross_attn.register_forward_hook(
        lambda module, args, output: outputs.append(output[0]))
    block(trajectory, history, action)
    block(trajectory, history.repeat(1, 2, 1), action)
    handle.remove()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    _, weights = block.action_cross_attn(trajectory, action, action, need_weights=True)
    assert weights.shape == (1, 4, 8)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 4))


def test_state_to_action_module_and_config_are_removed():
    from model.pinn_model import contact_world_model as module
    assert not hasattr(module, 'StateToActionBlock')
    model = ContactWorldModel(config())
    assert not any('condition_attention' in name or 'state_to_action' in name
                   for name, _ in model.named_modules())
    cfg = config()
    cfg['model']['state_to_action_attention_heads'] = 2
    with pytest.raises(ValueError, match='state_to_action_attention_heads was removed'):
        ContactWorldModel(cfg)
