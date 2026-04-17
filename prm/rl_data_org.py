import argparse
import json
import random
import re
from typing import Any, Dict, List


template_human = "<|start_header_id|>user<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
template_gpt = "{system_message}<|eot_id|>"


def build_grounding_values(conversations: List[Dict[str, str]], num_turns: int, invalid_regex: re.Pattern) -> List[float]:
    values = []
    for j in range(1, num_turns):
        obs_text = conversations[2 * j]["value"]
        if invalid_regex.search(obs_text):
            values.append(0.0)
        else:
            values.append(0.5)
    values.append(0.5)
    return values


def main(args: argparse.Namespace):
    with open(args.input_file, "r", encoding="utf-8") as f:
        prm_annotations = json.load(f)

    invalid_regex = re.compile(args.invalid_observation_pattern, flags=re.IGNORECASE)

    all_prompts = []
    all_responses = []
    all_rewards = []

    for ann in prm_annotations:
        conversations = ann["conversations"]
        turn_values = ann["turn_values"]

        if len(conversations) % 2 != 0:
            continue

        turn_num = len(conversations) // 2
        if turn_num != len(turn_values):
            continue

        prompts = []
        responses = []
        for j in range(turn_num):
            prompts.append(template_human.format(system_message=conversations[2 * j]["value"]))
            responses.append(template_gpt.format(system_message=conversations[2 * j + 1]["value"]))

        grounding_values = build_grounding_values(conversations, turn_num, invalid_regex)
        merged_rewards = [float(turn_values[j]) + grounding_values[j] for j in range(turn_num)]

        all_prompts.append(prompts)
        all_responses.append(responses)
        all_rewards.append(merged_rewards)

    random.seed(args.seed)
    total_samples = len(all_prompts)
    if total_samples > args.max_samples:
        sampled_indices = random.sample(range(total_samples), args.max_samples)
    else:
        sampled_indices = list(range(total_samples))

    sampled_prompts = [all_prompts[i] for i in sampled_indices]
    sampled_responses = [all_responses[i] for i in sampled_indices]
    sampled_rewards = [all_rewards[i] for i in sampled_indices]

    filtered = [
        i
        for i in range(len(sampled_rewards))
        if len(sampled_rewards[i]) < args.max_turns
    ]

    sampled_prompts = [sampled_prompts[i] for i in filtered]
    sampled_responses = [sampled_responses[i] for i in filtered]
    sampled_rewards = [sampled_rewards[i] for i in filtered]

    data_dict = {
        "prompt": sampled_prompts,
        "response": sampled_responses,
        "reward": sampled_rewards,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    data_flatten = []
    for i in range(len(data_dict["prompt"])):
        data_flatten.append(
            {
                "prompt": data_dict["prompt"][i],
                "response": data_dict["response"][i],
                "reward": data_dict["reward"][i],
            }
        )

    with open(args.output_flatten_file, "w", encoding="utf-8") as f:
        json.dump(data_flatten, f, ensure_ascii=False, indent=2)

    print(f"Saved RL training data to {args.output_file}")
    print(f"Saved RL flattened data to {args.output_flatten_file}")
    print(f"Total samples: {len(data_flatten)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build PPO training data from PRM inference outputs")
    parser.add_argument(
        "--input_file",
        type=str,
        default="prm/exploration_inference_results_feishu_travel.json",
        help="PRM inference result file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="prm/sampled_data_rl_training_feishu_travel.json",
        help="Output file for PPO grouped format",
    )
    parser.add_argument(
        "--output_flatten_file",
        type=str,
        default="prm/sampled_data_rl_training_feishu_travel_flatten.json",
        help="Output file for PPO flattened format",
    )
    parser.add_argument(
        "--invalid_observation_pattern",
        type=str,
        default="action_mismatch|format_error|invalid action|invalid format",
        help="Regex for observations that should not receive positive grounding signal",
    )
    parser.add_argument("--max_samples", type=int, default=4200)
    parser.add_argument("--max_turns", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
