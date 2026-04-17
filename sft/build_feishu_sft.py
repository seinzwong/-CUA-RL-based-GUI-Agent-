import argparse
import json
import os
from typing import Dict, List

SYSTEM_PROMPT = (
    "You are a Feishu desktop CUA agent for travel approval workflows. "
    "Follow task requirements and output exactly one action each turn using format: "
    "Action: <ACTION_NAME>."
)

OBS_TEMPLATE = (
    "Observation:\n"
    "Desktop=Feishu\n"
    "Module=TravelApproval\n"
    "RequestID={request_id}\n"
    "Destination={destination}\n"
    "Budget={budget}\n"
    "State={state}\n"
    "Detail={detail}"
)


def build_example(task: Dict, sample_id: int, with_rollback: bool) -> Dict:
    req = task.get("context", {})
    request_id = req.get("request_id", "REQ-UNKNOWN")
    destination = req.get("destination", "UNKNOWN")
    budget = req.get("budget", "UNKNOWN")

    task_text = task.get("task_text", "")

    canonical_actions = [
        "OPEN_TRAVEL_CENTER",
        "PARSE_TRAVEL_REQUEST",
        "SEARCH_CANDIDATE_OPTIONS",
        "COMPARE_OPTIONS",
        "FILL_APPROVAL_FORM",
        "GENERATE_APPROVAL_DRAFT",
    ]

    conv = [
        {"from": "human", "value": SYSTEM_PROMPT},
        {"from": "gpt", "value": "OK"},
        {"from": "human", "value": task_text},
    ]

    for i, action in enumerate(canonical_actions):
        conv.append({"from": "gpt", "value": f"Action: {action}"})
        conv.append(
            {
                "from": "human",
                "value": OBS_TEMPLATE.format(
                    request_id=request_id,
                    destination=destination,
                    budget=budget,
                    state="in_progress",
                    detail=f"Action accepted. Progress={i + 1}/{len(canonical_actions) + 2}.",
                ),
            }
        )

    if with_rollback:
        conv.append({"from": "gpt", "value": "Action: REVISE_FORM"})
        conv.append(
            {
                "from": "human",
                "value": OBS_TEMPLATE.format(
                    request_id=request_id,
                    destination=destination,
                    budget=budget,
                    state="rollback",
                    detail="REVISE_FORM accepted. Workflow returned to form editing stage for safer correction.",
                ),
            }
        )
        conv.append({"from": "gpt", "value": "Action: FILL_APPROVAL_FORM"})
        conv.append(
            {
                "from": "human",
                "value": OBS_TEMPLATE.format(
                    request_id=request_id,
                    destination=destination,
                    budget=budget,
                    state="in_progress",
                    detail="Form revised and validated.",
                ),
            }
        )
        conv.append({"from": "gpt", "value": "Action: GENERATE_APPROVAL_DRAFT"})
        conv.append(
            {
                "from": "human",
                "value": OBS_TEMPLATE.format(
                    request_id=request_id,
                    destination=destination,
                    budget=budget,
                    state="in_progress",
                    detail="Updated draft generated.",
                ),
            }
        )

    conv.append({"from": "gpt", "value": "Action: CONFIRM_SUBMIT"})
    conv.append(
        {
            "from": "human",
            "value": OBS_TEMPLATE.format(
                request_id=request_id,
                destination=destination,
                budget=budget,
                state="pre_submit_confirmation",
                detail="Confirmation captured (1/2). Submit is unlocked only after two confirmations.",
            ),
        }
    )
    conv.append({"from": "gpt", "value": "Action: CONFIRM_SUBMIT"})
    conv.append(
        {
            "from": "human",
            "value": OBS_TEMPLATE.format(
                request_id=request_id,
                destination=destination,
                budget=budget,
                state="pre_submit_confirmation",
                detail="Confirmation captured (2/2). Submit is unlocked.",
            ),
        }
    )
    conv.append({"from": "gpt", "value": "Action: SUBMIT_APPROVAL"})

    return {
        "id": sample_id,
        "task_id": task.get("task_id", f"task-{sample_id}"),
        "conversations": conv,
        "reward": 1.0,
        "source": "feishu_seed",
        "true_reward": 1.0,
        "safety_label": {
            "double_confirm_used": True,
            "rollback_used": with_rollback,
            "high_risk_write_guarded": True,
        },
    }


def build_dataset(tasks: List[Dict], replicas: int) -> List[Dict]:
    data = []
    sample_id = 0
    for task in tasks:
        for rep in range(replicas):
            data.append(build_example(task, sample_id, with_rollback=(rep % 2 == 1)))
            sample_id += 1
    return data


def main(args: argparse.Namespace):
    with open(args.input_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    dataset = build_dataset(tasks, args.replicas)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(dataset)} SFT samples -> {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build Feishu CUA SFT seed dataset")
    parser.add_argument(
        "--input_file",
        type=str,
        default="eval_agent/data/feishu_travel/train.json",
        help="Input task file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="sft/data/feishu_travel_sft.json",
        help="Output SFT dataset file",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=8,
        help="How many variants to generate per task",
    )
    args = parser.parse_args()
    main(args)
