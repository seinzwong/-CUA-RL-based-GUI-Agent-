import json
from typing import List, Tuple

from eval_agent.tasks.base import Task


class FeishuTravelTask(Task):
    task_name = "feishu_travel"

    def __init__(
        self,
        task_text: str,
        expected_actions: List[str],
        context: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_text = task_text
        self.expected_actions = expected_actions
        self.context = context

    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1) -> Tuple[List[Task], int]:
        data_path = f"eval_agent/data/feishu_travel/{split}.json"
        tasks = json.load(open(data_path, "r", encoding="utf-8"))

        if part_num > 1:
            assert part_idx != -1
            part_len = len(tasks) // part_num + 1
            tasks = tasks[part_len * part_idx: part_len * (part_idx + 1)]

        n_tasks = len(tasks)

        def generator():
            for item in tasks:
                yield cls(
                    task_id=item["task_id"],
                    task_text=item["task_text"],
                    expected_actions=item["expected_actions"],
                    context=item.get("context", {}),
                )

        return generator(), n_tasks
