import argparse
import logging
import time

from demo import GPUClient, DummyPolicy, ROBOT_CONFIGS, TASK_PROMPTS
from robot.interface_client import InterfaceClient

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

DEFAULT_USER_ID = "test_user"
DEFAULT_JOBS = ["test_job"]
DEFAULT_ROBOT_ID = "test_robot"


def process_job(client, gpu_client, job_id, robot_id, image_size, image_type, action_type, duration, max_wait=600):
    try:
        start_time = time.time()
        while True:
            client.start_motion()
            logging.info("Started robot")
            state = client.get_state(image_size, image_type, action_type)
            if not state:
                time.sleep(0.5)
                continue
            if state["state"] == "size_none":
                client.post_size()
                time.sleep(0.5)
                continue
            if state["state"] != "normal" or state["pending_actions"] != 0:
                time.sleep(0.5)
                continue
            logging.info("get_robot_state time: %.2f", time.time() - state["timestamp"])
            result = gpu_client.infer(state)
            logging.info(f"Inference result: {result}")
            client.post_actions(result, duration, action_type)
            if time.time() - start_time > max_wait:
                logging.warning(f"Job {job_id} exceeded max wait time.")
                break
        client.end_motion()
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
    finally:
        client.end_motion()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        choices=list(ROBOT_CONFIGS),
        help="Robot type to run inference for",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_PROMPTS),
        help="Task name that determines the language prompt",
    )
    parser.add_argument("--exec_horizon", type=int, default=50, help="Number of actions to output per inference")

    args = parser.parse_args()

    robot_cfg = ROBOT_CONFIGS[args.robot]
    image_size = [224, 224]
    image_type = robot_cfg["image_type"]
    action_type = robot_cfg["action_type"]
    duration = 0.05

    client = InterfaceClient(DEFAULT_USER_ID, mock=True)
    client.update_job_info(DEFAULT_JOBS[0], DEFAULT_ROBOT_ID)

    policy = DummyPolicy(args.checkpoint, args.robot, TASK_PROMPTS[args.task], args.exec_horizon)
    gpu_client = GPUClient(policy)

    jobs = DEFAULT_JOBS

    while jobs:
        for job_id in jobs[:]:
            try:
                process_job(client, gpu_client, job_id, DEFAULT_ROBOT_ID, image_size, image_type, action_type, duration)
                jobs.remove(job_id)
            except Exception as e:
                logging.error(f"Error processing job {job_id}: {e}")
                jobs.remove(job_id)
    logging.info("All jobs processed.")
    return True


if __name__ == "__main__":
    main()
