import torch
import torch.nn as nn


class TauFSequenceRegressor(nn.Module):
    """Causal recurrent regressor for free-space joint torque."""

    DEFAULT_INPUT_DIMS = {
        "q": 7,
        "dq": 7,
        "ddq": 7,
        "tau": 7,
    }

    def __init__(self, config):
        super().__init__()
        model_config = config.get("model") or {}

        self.active_inputs = list(
            model_config.get("inputs") or ["q", "dq", "ddq", "tau"]
        )
        if not self.active_inputs:
            raise ValueError("model.inputs must contain at least one input key.")
        if len(set(self.active_inputs)) != len(self.active_inputs):
            raise ValueError(
                f"model.inputs contains duplicate keys: {self.active_inputs}"
            )

        self.input_dims = {
            **self.DEFAULT_INPUT_DIMS,
            **(model_config.get("input_dims") or {}),
        }
        unknown_inputs = [
            key for key in self.active_inputs if key not in self.input_dims
        ]
        if unknown_inputs:
            raise ValueError(f"Missing model.input_dims for inputs: {unknown_inputs}")

        self.architecture = str(model_config.get("architecture", "lstm")).lower()
        recurrent_types = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        if self.architecture not in recurrent_types:
            raise ValueError(
                f"model.architecture must be one of {sorted(recurrent_types)}, "
                f"got {self.architecture!r}."
            )

        self.hidden_dim = int(model_config.get("hidden_dim", 128))
        self.num_layers = int(model_config.get("num_layers", 2))
        self.output_dim = int(model_config.get("output_dim", 7))
        self.dropout = float(model_config.get("dropout", 0.1))
        self.target_key = str(model_config.get("target_key", "tau_f"))

        if self.hidden_dim <= 0 or self.num_layers <= 0 or self.output_dim <= 0:
            raise ValueError("hidden_dim, num_layers, and output_dim must be positive.")

        recurrent_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.recurrent = recurrent_types[self.architecture](
            input_size=sum(int(self.input_dims[key]) for key in self.active_inputs),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )

        self.head = self._build_head(model_config)

    def _build_head(self, model_config):
        head_hidden_dim = int(model_config.get("head_hidden_dim", 256))
        head_num_layers = int(model_config.get("head_num_layers", 2))
        activation_name = str(model_config.get("activation", "relu")).lower()
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }

        if head_hidden_dim <= 0 or head_num_layers <= 0:
            raise ValueError("head_hidden_dim and head_num_layers must be positive.")
        if activation_name not in activations:
            raise ValueError(
                f"model.activation must be one of {sorted(activations)}, "
                f"got {activation_name!r}."
            )

        if head_num_layers == 1:
            return nn.Linear(self.hidden_dim, self.output_dim)

        layers = []
        in_dim = self.hidden_dim
        for _ in range(head_num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, head_hidden_dim),
                    activations[activation_name](),
                    nn.Dropout(self.dropout),
                ]
            )
            in_dim = head_hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        return nn.Sequential(*layers)

    def _prepare_inputs(self, batch):
        sequences = []
        batch_size = None
        time_steps = None

        for key in self.active_inputs:
            if key not in batch:
                raise KeyError(f"Missing model input {key!r} in batch.")

            value = batch[key]
            if value.ndim != 3:
                raise ValueError(
                    f"Input {key!r} must have shape [B, H, D], got {tuple(value.shape)}."
                )

            expected_dim = int(self.input_dims[key])
            if value.shape[-1] != expected_dim:
                raise ValueError(
                    f"Input {key!r} has last dimension {value.shape[-1]}, "
                    f"expected {expected_dim}."
                )

            if batch_size is None:
                batch_size, time_steps = value.shape[:2]
            elif value.shape[:2] != (batch_size, time_steps):
                raise ValueError(
                    f"All inputs must share [B, H]; {key!r} has {tuple(value.shape[:2])}, "
                    f"expected {(batch_size, time_steps)}."
                )
            sequences.append(value)

        return torch.cat(sequences, dim=-1)

    def _prepare_step_inputs(self, batch):
        step_inputs = []
        batch_size = None

        for key in self.active_inputs:
            if key not in batch:
                raise KeyError(f"Missing model input {key!r} in batch.")

            value = batch[key]
            if value.ndim != 2:
                raise ValueError(
                    f"Step input {key!r} must have shape [B, D], "
                    f"got {tuple(value.shape)}."
                )

            expected_dim = int(self.input_dims[key])
            if value.shape[-1] != expected_dim:
                raise ValueError(
                    f"Step input {key!r} has last dimension {value.shape[-1]}, "
                    f"expected {expected_dim}."
                )

            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"All step inputs must share batch size; {key!r} has "
                    f"{value.shape[0]}, expected {batch_size}."
                )
            step_inputs.append(value)

        return torch.cat(step_inputs, dim=-1).unsqueeze(1)

    @staticmethod
    def _last_target(value):
        if value.ndim == 3:
            return value[:, -1]
        if value.ndim == 2:
            return value
        raise ValueError(
            f"tau_f target must have shape [B, H, D] or [B, D], got {tuple(value.shape)}."
        )

    def init_recurrent_state(self, batch_size, device=None, dtype=None):
        """Create a zero recurrent state for a new episode or control stream."""
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (self.num_layers, int(batch_size), self.hidden_dim)
        hidden = torch.zeros(shape, device=device, dtype=dtype)
        if self.architecture == "lstm":
            return hidden, torch.zeros_like(hidden)
        return hidden

    @staticmethod
    def detach_recurrent_state(recurrent_state):
        """Detach a carried state at a truncated-BPTT or inference boundary."""
        if recurrent_state is None:
            return None
        if isinstance(recurrent_state, tuple):
            return tuple(state.detach() for state in recurrent_state)
        return recurrent_state.detach()

    def _run_recurrent(self, sequence, recurrent_state=None):
        if recurrent_state is None:
            return self.recurrent(sequence)
        return self.recurrent(sequence, recurrent_state)

    def forward(self, batch, recurrent_state=None):
        sequence = self._prepare_inputs(batch)
        recurrent_output, recurrent_state = self._run_recurrent(
            sequence,
            recurrent_state,
        )
        tau_f_pred = self.head(recurrent_output[:, -1])

        out = {
            "tau_f_pred": tau_f_pred,
            "sequence_features": recurrent_output,
            "recurrent_state": recurrent_state,
        }
        if self.target_key in batch:
            tau_f_target = self._last_target(batch[self.target_key])
            if tau_f_target.shape != tau_f_pred.shape:
                raise ValueError(
                    f"Prediction shape {tuple(tau_f_pred.shape)} does not match "
                    f"target shape {tuple(tau_f_target.shape)}."
                )
            out["tau_f_target"] = tau_f_target
        return out

    def forward_step(self, batch, recurrent_state=None):
        """Predict one control step while carrying GRU/LSTM state explicitly.

        Each configured input must have shape [B, D]. The returned
        ``recurrent_state`` should be passed into the next call and reset when an
        episode, robot session, or other independent stream starts.
        """
        sequence = self._prepare_step_inputs(batch)
        recurrent_output, recurrent_state = self._run_recurrent(
            sequence,
            recurrent_state,
        )
        tau_f_pred = self.head(recurrent_output[:, -1])

        out = {
            "tau_f_pred": tau_f_pred,
            "sequence_features": recurrent_output,
            "recurrent_state": recurrent_state,
        }
        if self.target_key in batch:
            tau_f_target = self._last_target(batch[self.target_key])
            if tau_f_target.shape != tau_f_pred.shape:
                raise ValueError(
                    f"Prediction shape {tuple(tau_f_pred.shape)} does not match "
                    f"target shape {tuple(tau_f_target.shape)}."
                )
            out["tau_f_target"] = tau_f_target
        return out
