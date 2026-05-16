"""
Open-loop evaluation: feed recorded episode data to the policy frame-by-frame
and compare inferred actions against recorded ground-truth actions.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "openpi/src")

from demo import DummyPolicy, ROBOT_CONFIGS, TASK_PROMPTS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def load_episode_aloha(data_dir: Path, image_size: tuple[int, int]):
    """Load an aloha dual-arm episode: left/right states + 3 cameras."""
    left_states = [json.loads(l) for l in open(data_dir / "states/left_states.jsonl")]
    right_states = [json.loads(l) for l in open(data_dir / "states/right_states.jsonl")]
    assert len(left_states) == len(right_states)

    frames_high = _read_video_frames(data_dir / "videos/cam_high_rgb.mp4", image_size)
    frames_left = _read_video_frames(data_dir / "videos/cam_wrist_left_rgb.mp4", image_size)
    frames_right = _read_video_frames(data_dir / "videos/cam_wrist_right_rgb.mp4", image_size)

    n = min(len(left_states), len(frames_high), len(frames_left), len(frames_right))
    episodes = []
    for i in range(n):
        state = (
            left_states[i]["joint_positions"] + [left_states[i]["gripper"]]
            + right_states[i]["joint_positions"] + [right_states[i]["gripper"]]
        )
        episodes.append({
            "state": state,
            "images": {
                "high": _encode_png(frames_high[i]),
                "left_hand": _encode_png(frames_left[i]),
                "right_hand": _encode_png(frames_right[i]),
            },
        })
    gt_actions = []
    for i in range(n):
        gt_actions.append(
            left_states[i]["joint_positions"] + [left_states[i]["gripper"]]
            + right_states[i]["joint_positions"] + [right_states[i]["gripper"]]
        )
    return episodes, gt_actions


def load_episode_arx5(data_dir: Path, image_size: tuple[int, int]):
    """Load an arx5 single-arm episode."""
    states = [json.loads(l) for l in open(data_dir / "states/states.jsonl")]

    frames_high = _read_video_frames(data_dir / "videos/right_realsense_rgb.mp4", image_size)
    frames_left = _read_video_frames(data_dir / "videos/arm_realsense_rgb.mp4", image_size)
    frames_right = _read_video_frames(data_dir / "videos/global_realsense_rgb.mp4", image_size)

    n = min(len(states), len(frames_high), len(frames_left), len(frames_right))
    episodes = []
    for i in range(n):
        state = states[i]["joint_positions"] + [states[i]["gripper_width"]]
        episodes.append({
            "state": state,
            "images": {
                "high": _encode_png(frames_high[i]),
                "left_hand": _encode_png(frames_left[i]),
                "right_hand": _encode_png(frames_right[i]),
            },
        })
    gt_actions = []
    for i in range(n):
        gt_actions.append(states[i]["joint_positions"] + [states[i]["gripper_width"]])
    return episodes, gt_actions


def load_episode_franka(data_dir: Path, image_size: tuple[int, int]):
    """Load a franka single-arm episode."""
    states = [json.loads(l) for l in open(data_dir / "states/states.jsonl")]

    frames_high = _read_video_frames(data_dir / "videos/side_realsense_rgb.mp4", image_size)
    frames_left = _read_video_frames(data_dir / "videos/handeye_realsense_rgb.mp4", image_size)
    frames_right = _read_video_frames(data_dir / "videos/main_realsense_rgb.mp4", image_size)

    n = min(len(states), len(frames_high), len(frames_left), len(frames_right))
    episodes = []
    for i in range(n):
        state = states[i]["joint_positions"] + [states[i]["gripper_width"]]
        episodes.append({
            "state": state,
            "images": {
                "high": _encode_png(frames_high[i]),
                "left_hand": _encode_png(frames_left[i]),
                "right_hand": _encode_png(frames_right[i]),
            },
        })
    gt_actions = []
    for i in range(n):
        gt_actions.append(states[i]["joint_positions"] + [states[i]["gripper_width"]])
    return episodes, gt_actions


def load_episode_ur5(data_dir: Path, image_size: tuple[int, int]):
    """Load a ur5 single-arm episode (2 cameras only)."""
    states = [json.loads(l) for l in open(data_dir / "states/states.jsonl")]

    frames_left = _read_video_frames(data_dir / "videos/handeye_realsense_rgb.mp4", image_size)
    frames_high = _read_video_frames(data_dir / "videos/global_realsense_rgb.mp4", image_size)

    n = min(len(states), len(frames_high), len(frames_left))
    episodes = []
    for i in range(n):
        state = states[i]["joint_positions"] + [states[i]["gripper"]]
        episodes.append({
            "state": state,
            "images": {
                "high": _encode_png(frames_high[i]),
                "left_hand": _encode_png(frames_left[i]),
            },
        })
    gt_actions = []
    for i in range(n):
        gt_actions.append(states[i]["joint_positions"] + [states[i]["gripper"]])
    return episodes, gt_actions


EPISODE_LOADERS = {
    "aloha": load_episode_aloha,
    "arx5": load_episode_arx5,
    "franka": load_episode_franka,
    "ur5": load_episode_ur5,
}


def _read_video_frames(video_path: Path, image_size: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, image_size)
        frames.append(frame)
    cap.release()
    return frames


def _encode_png(frame: np.ndarray) -> bytes:
    return cv2.imencode(".png", frame)[-1].tobytes()


def run_openloop_eval(policy, episodes, gt_actions, exec_horizon: int = 50):
    """
    Run open-loop evaluation.

    For each step (every exec_horizon frames), feed the recorded observation to the policy
    and compare the predicted action chunk against the ground-truth future actions.

    Returns metrics dict and collected (gt, pred) action chunks for visualization.
    """
    n = len(episodes)
    all_mse = []
    all_mae = []
    all_gt_chunks = []
    all_pred_chunks = []

    eval_indices = list(range(0, n - exec_horizon, exec_horizon))
    logging.info(f"Running open-loop eval on {len(eval_indices)} steps (total frames: {n})")

    for idx in eval_indices:
        obs = episodes[idx]
        state_input = {
            "action": obs["state"],
            "images": obs["images"],
            "state": "normal",
            "pending_actions": 0,
            "timestamp": time.time(),
        }

        pred_actions = np.array(policy.run_policy(state_input))
        gt_chunk = np.array(gt_actions[idx + 1: idx + 1 + exec_horizon])

        compare_len = min(len(pred_actions), len(gt_chunk))
        pred = pred_actions[:compare_len]
        gt = gt_chunk[:compare_len]

        all_gt_chunks.append(gt)
        all_pred_chunks.append(pred)

        mse = float(np.mean((pred - gt) ** 2))
        mae = float(np.mean(np.abs(pred - gt)))
        all_mse.append(mse)
        all_mae.append(mae)

        logging.info(f"  Chunk {len(all_mse)}/{len(eval_indices)} (frame {idx}): MSE={mse:.6f}, MAE={mae:.6f}")

    results = {
        "num_steps": len(all_mse),
        "mean_mse": float(np.mean(all_mse)),
        "std_mse": float(np.std(all_mse)),
        "mean_mae": float(np.mean(all_mae)),
        "std_mae": float(np.std(all_mae)),
    }
    return results, all_gt_chunks, all_pred_chunks


def plot_openloop_actions(all_gt_chunks, all_pred_chunks, save_path, max_plot=None):
    """
    Plot predicted vs ground-truth actions across consecutive inference steps.
    Each action chunk is concatenated to form a continuous trajectory per dimension.
    """
    num_plot = len(all_gt_chunks) if max_plot is None else min(len(all_gt_chunks), max_plot)
    if num_plot == 0:
        logging.warning("No data to plot.")
        return

    gt_concat = np.concatenate(all_gt_chunks[:num_plot], axis=0)
    pred_concat = np.concatenate(all_pred_chunks[:num_plot], axis=0)

    total_steps, num_dims = gt_concat.shape
    steps_per_chunk = all_gt_chunks[0].shape[0]

    ncols = 2
    nrows = (num_dims + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharex=True)
    axes = axes.flatten()

    x_axis = np.arange(total_steps)
    start_indices = np.arange(0, total_steps, steps_per_chunk)

    for dim_idx in range(num_dims):
        ax = axes[dim_idx]
        ax.plot(x_axis, gt_concat[:, dim_idx], label="Ground Truth", color="cornflowerblue", alpha=0.9)
        ax.plot(x_axis, pred_concat[:, dim_idx], label="Predicted", color="tomato", linestyle="--", alpha=0.9)

        ax.scatter(start_indices, gt_concat[start_indices, dim_idx],
                   c="blue", marker="o", s=20, zorder=5)
        ax.scatter(start_indices, pred_concat[start_indices, dim_idx],
                   c="darkred", marker="x", s=20, zorder=5)

        for i, si in enumerate(start_indices):
            ax.axvline(x=si, color="gray", linestyle=":", alpha=0.3)
            ax.text(si, 1.0, str(i), fontsize=6, ha="center", va="bottom",
                    color="gray", transform=ax.get_xaxis_transform())

        ax.set_title(f"Dim {dim_idx}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.6)
        if dim_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for dim_idx in range(num_dims, len(axes)):
        axes[dim_idx].set_visible(False)

    fig.supxlabel(f"Timestep (across {num_plot} inference chunks)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved action trajectory plot to {save_path}")


def plot_mse_per_step(all_gt_chunks, all_pred_chunks, save_path):
    """Plot per-timestep MSE within the action chunk, averaged over all inferences."""
    if not all_gt_chunks:
        return
    horizon = all_gt_chunks[0].shape[0]
    per_step_mse = np.zeros(horizon)
    for gt, pred in zip(all_gt_chunks, all_pred_chunks):
        compare_len = min(len(gt), len(pred), horizon)
        per_step_mse[:compare_len] += np.mean((gt[:compare_len] - pred[:compare_len]) ** 2, axis=-1)
    per_step_mse /= len(all_gt_chunks)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(horizon), per_step_mse, color="steelblue", alpha=0.8)
    ax.set_xlabel("Step within action chunk")
    ax.set_ylabel("MSE")
    ax.set_title("Per-step MSE (averaged over all inferences)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved per-step MSE plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Open-loop evaluation against recorded episodes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--robot", type=str, required=True, choices=list(ROBOT_CONFIGS),
        help="Robot type",
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=list(TASK_PROMPTS),
        help="Task name that determines the language prompt",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to episode directory")
    parser.add_argument("--exec_horizon", type=int, default=50, help="Number of actions to output per inference")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save plots")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = (224, 224)

    logging.info(f"Loading episode from {data_dir} for robot '{args.robot}'...")
    loader = EPISODE_LOADERS[args.robot]
    episodes, gt_actions = loader(data_dir, image_size)
    logging.info(f"Loaded {len(episodes)} frames")

    logging.info("Initializing policy...")
    policy = DummyPolicy(args.checkpoint, args.robot, TASK_PROMPTS[args.task], args.exec_horizon)

    logging.info("Starting open-loop evaluation...")
    results, all_gt_chunks, all_pred_chunks = run_openloop_eval(
        policy, episodes, gt_actions,
        exec_horizon=args.exec_horizon,
    )

    logging.info("=" * 50)
    logging.info("Open-loop evaluation results:")
    logging.info(f"  Steps evaluated: {results['num_steps']}")
    logging.info(f"  Mean MSE: {results['mean_mse']:.6f} (+/- {results['std_mse']:.6f})")
    logging.info(f"  Mean MAE: {results['mean_mae']:.6f} (+/- {results['std_mae']:.6f})")
    logging.info("=" * 50)

    logging.info("Generating plots...")
    plot_openloop_actions(
        all_gt_chunks, all_pred_chunks,
        save_path=output_dir / "openloop_actions.png",
    )
    plot_mse_per_step(
        all_gt_chunks, all_pred_chunks,
        save_path=output_dir / "openloop_mse_per_step.png",
    )


if __name__ == "__main__":
    main()
