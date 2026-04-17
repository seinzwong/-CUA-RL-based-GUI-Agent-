import argparse
import json
import os
import pathlib
import time
from typing import Any, Dict

from tqdm import tqdm

import eval_agent.agents as agents
import eval_agent.envs as envs
import eval_agent.tasks as tasks


def construct_llm_data(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json"), "r", encoding="utf-8") as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json"), "r", encoding="utf-8") as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.model_name is not None:
        agent_config["config"]["model_name"] = args.model_name

    env_config = exp_config["env_config"]
    task_config = exp_config["task"]

    task_class = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    agent = getattr(agents, agent_config["agent_class"])(agent_config["config"])

    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(args.save_path, f"{args.exp_config}_traj_{args.part_idx}.json")

    final_data = []
    pbar = tqdm(total=n_tasks)

    for task_i, task in enumerate(all_tasks):
        if args.debug and task_i >= 5:
            break

        for iteration in range(args.iteration_num):
            env = getattr(envs, env_config["env_class"])(task, **env_config)
            _, state = env.reset()

            start_time = time.time()
            while not state.finished:
                try:
                    llm_output = agent(state.history)
                except Exception:
                    state.success = False
                    state.finished = True
                    state.terminate_reason = "agent_exception"
                    break

                _, state = env.step(llm_output)

            elapsed = time.time() - start_time
            final_data.append(
                {
                    "id": task.task_id,
                    "iteration": iteration,
                    "agent_conversations": state.history,
                    "agent_final_reward": state.reward,
                    "time": elapsed,
                    "success": state.success,
                    "terminate_reason": state.terminate_reason,
                }
            )

        pbar.update(1)

    pbar.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"Saved trajectories to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate exploration trajectories for Feishu CUA mock environment")
    parser.add_argument("--exp_path", type=str, default="eval_agent/configs/task")
    parser.add_argument("--exp_config", type=str, default="feishu_travel")
    parser.add_argument("--agent_path", type=str, default="eval_agent/configs/model")
    parser.add_argument("--agent_config", type=str, default="fastchat_explore")
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=0)
    parser.add_argument("--iteration_num", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="exploration/feishu_travel/exploration_outputs/explore")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    construct_llm_data(args)
