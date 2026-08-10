from pathlib import Path

from data_process.tool.sequence_torque_inference_visualizer import (
    TorqueVisualizationTask,
    build_parser,
    run_visualization,
)


TASK = TorqueVisualizationTask(
    name="tau_f",
    target_key="tau_f",
    target_label="tau_f label",
    prediction_label="tau_f prediction",
    default_output_dir=Path("outputs/inference_visualization/tau_f"),
    rollout_mode="tau_f",
)


def main() -> None:
    run_visualization(TASK, build_parser(TASK).parse_args())


if __name__ == "__main__":
    main()
