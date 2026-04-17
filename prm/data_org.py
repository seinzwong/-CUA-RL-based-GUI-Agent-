import argparse
import glob
import json
import os
from typing import Any, Dict, List


def normalize_conversations(agent_conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted = []
    for turn in agent_conversations:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        converted.append(
            {
                "from": "human" if role == "user" else "gpt",
                "value": content,
            }
        )

    # Keep complete (human, gpt) pairs only.
    if converted and converted[-1]["from"] == "human":
        converted = converted[:-1]

    return converted


def main(args: argparse.Namespace):
    json_files = glob.glob(os.path.join(args.explore_dir, "*.json"))
    print(f"Found {len(json_files)} json files under {args.explore_dir}")

    all_samples = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_samples.extend(data)
            else:
                print(f"Skip non-list file: {json_file}")
        except Exception as exc:
            print(f"Error reading {json_file}: {exc}")

    print(f"Collected {len(all_samples)} raw trajectory samples")

    processed = []
    for item in all_samples:
        conv = normalize_conversations(item.get("agent_conversations", []))
        if len(conv) < 2:
            continue
        processed.append(
            {
                "conversations": conv,
                "agent_final_reward": item.get("agent_final_reward", 0.0),
                "id": item.get("id"),
                "iteration": item.get("iteration", 0),
                "success": item.get("success", False),
            }
        )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"Saved merged exploration data to {args.output_file}")

    tiny = processed[: args.tiny_size]
    with open(args.tiny_output_file, "w", encoding="utf-8") as f:
        json.dump(tiny, f, ensure_ascii=False, indent=2)
    print(f"Saved tiny exploration data to {args.tiny_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Organize exploration trajectories into PRM training format")
    parser.add_argument(
        "--explore_dir",
        type=str,
        default="exploration/feishu_travel/exploration_outputs/explore",
        help="Directory containing raw trajectory json files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="exploration/feishu_travel/exploration_outputs/exploration.json",
        help="Merged exploration output",
    )
    parser.add_argument(
        "--tiny_output_file",
        type=str,
        default="exploration/feishu_travel/exploration_outputs/exploration_tiny.json",
        help="Small subset used for quick validation",
    )
    parser.add_argument(
        "--tiny_size",
        type=int,
        default=100,
        help="Number of samples in tiny output",
    )
    args = parser.parse_args()
    main(args)
