"""
Minimal example: convert dataset to the LeRobot format.

CLI Example (using the *arrange_flowers* task as an example):
    python convert_libero_to_lerobot.py \
        --repo-name arrange_flowers_repo \
        --raw-dataset /path/to/arrange_flowers \
        --frame-interval 1 \

Notes:
- If you plan to push to the Hugging Face Hub later, handle that outside this script.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from lerobot.common.datasets.lerobot_dataset import (HF_LEROBOT_HOME,
                                                     LeRobotDataset)
from tqdm.auto import tqdm

ROBOT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "arx5": {
        "video_files": {
            "global_image": "global_realsense_rgb.mp4",
            "wrist_image": "arm_realsense_rgb.mp4",
            "right_image": "right_realsense_rgb.mp4",
        },
        "state_files": ("states.jsonl",),
        "joint_key": "joint_positions",
        "pose_key": "end_effector_pose",
        "gripper_key": "gripper_width",
    },
    "ur5": {
        "video_files": {
            "global_image": "global_realsense_rgb.mp4",
            "wrist_image": "handeye_realsense_rgb.mp4",
        },
        "state_files": ("states.jsonl",),
        "joint_key": "joint_positions",
        "pose_key": "ee_positions",
        "gripper_key": "gripper",
    },
    "franka": {
        "video_files": {
            "global_image": "main_realsense_rgb.mp4",
            "wrist_image": "handeye_realsense_rgb.mp4",
            "right_image": "side_realsense_rgb.mp4",
        },
        "state_files": ("states.jsonl",),
        "joint_key": "joint_positions",
        "pose_key": "ee_positions",
        "gripper_key": "gripper_width",
    },
    "aloha": {
        "video_files": {
            "observation.images.cam_high": "cam_high_rgb.mp4",
            "observation.images.cam_left_wrist": "cam_wrist_left_rgb.mp4",
            "observation.images.cam_right_wrist": "cam_wrist_right_rgb.mp4",
        },
        "state_files": ("left_states.jsonl", "right_states.jsonl"),
        "joint_key": "joint_positions",
        "pose_key": "ee_pose_quaternion",
        "gripper_key": "gripper",
        "dual_arm": True,
    },
}


@contextmanager
def suppress_stderr_on_success():
    """Hide noisy native stderr output from video encoding, but replay it if an error occurs."""
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")

    try:
        os.dup2(tmp.fileno(), stderr_fd)
        try:
            yield
        except Exception:
            os.dup2(saved_stderr_fd, stderr_fd)
            tmp.seek(0)
            stderr_output = tmp.read().decode("utf-8", errors="replace").strip()
            if stderr_output:
                print("Captured ffmpeg/libav stderr:", file=sys.stderr)
                print(stderr_output, file=sys.stderr)
            raise
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        tmp.close()


def normalize_robot(robot: str) -> str:
    """Normalize and validate robot tag used to select camera video files."""
    robot = robot.lower()
    if robot not in ROBOT_CONFIGS:
        supported = ", ".join(sorted(ROBOT_CONFIGS))
        raise ValueError(f"Unsupported robot '{robot}'. Supported robots: {supported}")
    return robot


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ffmpeg_decode_all_frames(
    video_path: Path,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Decode an entire video to a (N, H, W, 3) uint8 RGB array using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={target_width}:{target_height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {video_path}: {stderr}")

    frame_bytes = target_height * target_width * 3
    n_frames = len(result.stdout) // frame_bytes
    if n_frames == 0:
        raise RuntimeError(f"ffmpeg decoded zero frames from {video_path}")

    frames = np.frombuffer(result.stdout[: n_frames * frame_bytes], dtype=np.uint8)
    return frames.reshape(n_frames, target_height, target_width, 3)


def create_lerobot_dataset(
    repo_name: str,
    robot_type: str,
    fps: float,
    height: int,
    width: int,
    pose_dim: int,
    joint_dim: int,
    camera_names: List[str],
) -> LeRobotDataset:
    """
    Create a LeRobot dataset with custom feature schema
    """
    features: Dict[str, Any] = {}
    for camera_name in camera_names:
        features[camera_name] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "h264_nvenc",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    features.update(
        {
            "observation.eef_state": {
                "dtype": "float32",
                "shape": (pose_dim,),
            },
            "eef_action": {
                "dtype": "float32",
                "shape": (pose_dim,),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (joint_dim,),
            },
            "action": {
                "dtype": "float32",
                "shape": (joint_dim,),
            },
        }
    )

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=32,
        image_writer_processes=16,
        video_backend="torchcodec",
    )
    return dataset


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def make_pose(state: Dict[str, Any], robot_config: Dict[str, Any]) -> np.ndarray:
    """Build pose-space vector from EE pose plus gripper width."""
    pose_key = robot_config["pose_key"]
    gripper_key = robot_config["gripper_key"]
    return np.concatenate((_as_float_array(state[pose_key]), _as_float_array(state[gripper_key])))


def make_joint(state: Dict[str, Any], robot_config: Dict[str, Any]) -> np.ndarray:
    """Build joint-space vector from joint positions plus gripper width."""
    joint_key = robot_config["joint_key"]
    gripper_key = robot_config["gripper_key"]
    return np.concatenate((_as_float_array(state[joint_key]), _as_float_array(state[gripper_key])))


def make_state_vectors(
    state: Dict[str, Any],
    robot_config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    return {
        "pose": make_pose(state, robot_config),
        "joint": make_joint(state, robot_config),
    }


def make_dual_arm_state_vectors(
    left_state: Dict[str, Any],
    right_state: Dict[str, Any],
    robot_config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    return {
        "pose": np.concatenate((make_pose(left_state, robot_config), make_pose(right_state, robot_config))),
        "joint": np.concatenate((make_joint(left_state, robot_config), make_joint(right_state, robot_config))),
    }


def load_episode_state_vectors(
    episode_path: Path,
    robot_config: Dict[str, Any],
) -> List[Dict[str, np.ndarray]]:
    states_dir = episode_path / "states"
    state_files = robot_config["state_files"]
    if robot_config.get("dual_arm", False):
        left_states = load_jsonl(states_dir / state_files[0])
        right_states = load_jsonl(states_dir / state_files[1])
        if len(left_states) != len(right_states):
            raise AssertionError(
                f"Mismatch in episode {episode_path.name}: "
                f"left_states={len(left_states)}, right_states={len(right_states)}"
            )
        return [
            make_dual_arm_state_vectors(left_state, right_state, robot_config)
            for left_state, right_state in zip(left_states, right_states)
        ]

    return [make_state_vectors(state, robot_config) for state in load_jsonl(states_dir / state_files[0])]


def process_episode_dir(
    episode_path: Path,
    dataset: LeRobotDataset,
    prompt: str,
    robot_config: Dict[str, Any],
    video_files: Dict[str, str],
    height: int,
    width: int,
) -> None:
    """
    Process a single episode directory and append frames to the given dataset.

    episode_path : Path
        Episode directory containing `states/*.jsonl` and `videos/*.mp4`.
    dataset : LeRobotDataset
        Target dataset to which frames are added.
    prompt : str
        Language instruction of this episode.
    """
    videos_dir = episode_path / "videos"

    ep_states = load_episode_state_vectors(episode_path, robot_config)

    file_name_to_features: Dict[str, List[str]] = {}
    for feature_name, file_name in video_files.items():
        file_name_to_features.setdefault(file_name, []).append(feature_name)

    decoded_videos: Dict[str, np.ndarray] = {}
    for file_name, feature_names in file_name_to_features.items():
        video_path = videos_dir / file_name
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video: {video_path}")
        frames = ffmpeg_decode_all_frames(video_path, height, width)
        for feature_name in feature_names:
            decoded_videos[feature_name] = frames

    frame_counts = {feature_name: frames.shape[0] for feature_name, frames in decoded_videos.items()}
    n_states = len(ep_states)

    mismatched_counts = {feature_name: count for feature_name, count in frame_counts.items() if count != n_states}
    if mismatched_counts:
        counts = ", ".join(f"{feature_name}={count}" for feature_name, count in frame_counts.items())
        raise AssertionError(f"Mismatch in episode {episode_path.name}: states={n_states}, {counts}")

    # write frames to the episode of lerobot dataset
    for idx in range(1, n_states):
        obs_idx = idx - 1
        pose = ep_states[idx]["pose"]
        last_pose = ep_states[obs_idx]["pose"]
        joint = ep_states[idx]["joint"]
        last_joint = ep_states[obs_idx]["joint"]

        frame = {
            feature_name: frames[obs_idx]
            for feature_name, frames in decoded_videos.items()
        }

        frame.update(
            {
                "observation.eef_state": last_pose.astype(np.float32, copy=False),
                "eef_action": pose.astype(np.float32, copy=False),
                "observation.state": last_joint.astype(np.float32, copy=False),
                "action": joint.astype(np.float32, copy=False),
            }
        )

        dataset.add_frame(frame, task=prompt)

    with suppress_stderr_on_success():
        dataset.save_episode()


def main(
    repo_name: str,
    raw_dataset: Path,
    robot: str | None = None,
    overwrite_repo: bool = False,
) -> None:
    """
    Convert a dataset directory into LeRobot format.

    repo_name : str
        Output repo/dataset name (saved under $HF_LEROBOT_HOME / repo_name).
    raw_dataset : Path
        Path to the raw dataset root directory.
    robot : str | None, default=None
        Robot type used to select camera video files. If None, read from task_info.
    overwrite_repo : bool, default=False
        If True, remove the existing dataset directory before writing.
    """
    dst_dir = HF_LEROBOT_HOME / repo_name
    if overwrite_repo and dst_dir.exists():
        print(f"removing existing dataset at {dst_dir}")
        shutil.rmtree(dst_dir)

    # Load task_infos
    task_info_path = raw_dataset / "meta" / "task_info.json"
    with task_info_path.open("r", encoding="utf-8") as f:
        task_info = json.load(f)

    task_robot = task_info["task_desc"]["task_tag"][2]
    robot_type = normalize_robot(robot or task_robot)
    robot_config = ROBOT_CONFIGS[robot_type]
    video_files = robot_config["video_files"]
    video_info = task_info["video_info"]
    video_info["width"] = 640  # TODO: derive from task_info or actual videos
    video_info["height"] = 480
    fps = int(video_info["fps"])

    prompt = task_info["task_desc"]["prompt"]

    data_root = raw_dataset / "data"
    episode_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episodes found under {data_root}")

    first_states = load_episode_state_vectors(episode_dirs[0], robot_config)
    if not first_states:
        raise RuntimeError(f"No states found in first episode: {episode_dirs[0]}")
    pose_dim = first_states[0]["pose"].shape[0]
    joint_dim = first_states[0]["joint"].shape[0]

    # Create dataset, define feature in the form you need.
    # - proprio is stored in state/action fields for both pose and joint spaces
    # - LeRobot assumes that dtype of image data is `image`
    dataset = create_lerobot_dataset(
        repo_name=repo_name,
        robot_type=robot_type,
        fps=fps,
        height=video_info["height"],
        width=video_info["width"],
        pose_dim=pose_dim,
        joint_dim=joint_dim,
        camera_names=list(video_files),
    )

    # populate the dataset to lerobot dataset
    for episode_path in tqdm(episode_dirs, desc="Episodes", unit="episode"):
        process_episode_dir(
            episode_path=episode_path,
            dataset=dataset,
            prompt=prompt,
            robot_config=robot_config,
            video_files=video_files,
            height=video_info["height"],
            width=video_info["width"],
        )

    print(f"Done. Dataset saved to: {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a custom dataset to LeRobot format.")
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Name of the output dataset (under $HF_HF_LEROBOT_HOME).",
    )
    parser.add_argument(
        "--raw-dataset",
        required=True,
        type=str,
        help="Path to the raw dataset root.",
    )
    parser.add_argument(
        "--robot",
        type=normalize_robot,
        help="Robot type used to select input video files. Defaults to task_info task_tag[2].",
    )
    parser.add_argument(
        "--overwrite-repo",
        action="store_true",
        help="Remove existing output directory if it exists.",
    )
    args = parser.parse_args()

    main(
        repo_name=args.repo_name,
        raw_dataset=Path(args.raw_dataset),
        robot=args.robot,
        overwrite_repo=args.overwrite_repo,
    )
