import re
from typing import Tuple

from eval_agent.envs.base import BaseEnv
from eval_agent.tasks.feishu_travel import FeishuTravelTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State


class FeishuDesktopEnvMock(BaseEnv):
    def __init__(
        self,
        task: FeishuTravelTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task = task
        self.state = State()
        self.action_cursor = 0
        self.confirm_count = 0
        self.submit_unlocked = False

    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        return action.strip()

    def _format_observation(self, status: str, detail: str) -> str:
        context = self.task.context
        request_id = context.get("request_id", "N/A")
        city = context.get("destination", "N/A")
        budget = context.get("budget", "N/A")
        return (
            "Observation:\n"
            f"Desktop=Feishu\n"
            f"Module=TravelApproval\n"
            f"RequestID={request_id}\n"
            f"Destination={city}\n"
            f"Budget={budget}\n"
            f"State={status}\n"
            f"Detail={detail}"
        )

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({"role": "assistant", "content": llm_output})

        try:
            action = self.parse_action(llm_output)
        except Exception:
            observation = self._format_observation(
                "format_error",
                "Invalid output format. You must respond with 'Action: <ACTION_NAME>[args]'.",
            )
            self.state.history.append({"role": "user", "content": observation})
            self.state.steps += 1
            self.state.reward = -0.2
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
            return observation, self.state

        expected = self.task.expected_actions
        done = False

        # Allow explicit rollback before final submission to model safer office automation behavior.
        if action in ["CANCEL_SUBMIT", "REVISE_FORM"]:
            fill_idx = expected.index("FILL_APPROVAL_FORM") if "FILL_APPROVAL_FORM" in expected else 0
            self.action_cursor = fill_idx
            self.confirm_count = 0
            self.submit_unlocked = False
            reward = -0.05 if action == "REVISE_FORM" else -0.1
            observation = self._format_observation(
                "rollback",
                f"{action} accepted. Workflow returned to form editing stage for safer correction.",
            )
            self.state.history.append({"role": "user", "content": observation})
            self.state.steps += 1
            self.state.reward = reward
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
            return observation, self.state

        if self.action_cursor >= len(expected):
            done = True
            reward = 1.0
            observation = self._format_observation("completed", "All required steps already completed.")
        else:
            target_action = expected[self.action_cursor]

            # Require double confirmation before submission.
            if target_action == "SUBMIT_APPROVAL":
                if action == "CONFIRM_SUBMIT":
                    self.confirm_count += 1
                    self.submit_unlocked = self.confirm_count >= 2
                    reward = 0.2
                    observation = self._format_observation(
                        "pre_submit_confirmation",
                        f"Confirmation captured ({self.confirm_count}/2). Submit is unlocked only after two confirmations.",
                    )
                elif action == "SUBMIT_APPROVAL":
                    if self.confirm_count < 2:
                        reward = -0.2
                        observation = self._format_observation(
                            "submit_blocked",
                            "Submission blocked: perform CONFIRM_SUBMIT twice before SUBMIT_APPROVAL.",
                        )
                    else:
                        self.action_cursor += 1
                        reward = 1.0
                        done = self.action_cursor == len(expected)
                        observation = self._format_observation(
                            "completed",
                            "Approval draft is submitted after double confirmation.",
                        )
                        self.confirm_count = 0
                        self.submit_unlocked = False
                else:
                    reward = -0.1
                    observation = self._format_observation(
                        "action_mismatch",
                        f"Unexpected action. Expected next action: {target_action}.",
                    )
            elif action == target_action:
                self.action_cursor += 1
                progress = self.action_cursor / len(expected)
                reward = round(progress, 3)
                if action == "CONFIRM_SUBMIT":
                    self.confirm_count += 1
                    self.submit_unlocked = self.confirm_count >= 2
                if self.action_cursor == len(expected):
                    done = True
                    observation = self._format_observation(
                        "completed",
                        "Approval draft is ready and submission is confirmed.",
                    )
                else:
                    observation = self._format_observation(
                        "in_progress",
                        f"Action accepted. Progress={self.action_cursor}/{len(expected)}.",
                    )
            else:
                reward = -0.1
                observation = self._format_observation(
                    "action_mismatch",
                    f"Unexpected action. Expected next action: {target_action}.",
                )

        self.state.history.append({"role": "user", "content": observation})
        self.state.steps += 1
        self.state.reward = reward

        if self.state.steps >= self.max_steps and not done:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
        elif done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"

        return observation, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.action_cursor = 0
        self.confirm_count = 0
        self.submit_unlocked = False
        cur_task = self.task.task_text
        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)

        if self.icl_format == "first":
            self.state.history.append({"role": "user", "content": observation})
        else:
            self.state.history = messages

        return observation, self.state
