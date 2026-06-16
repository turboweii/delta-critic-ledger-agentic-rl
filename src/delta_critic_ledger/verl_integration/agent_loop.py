from __future__ import annotations

import copy
from uuid import uuid4

from delta_critic_ledger.adaptive_control import AdaptiveEntropyController, decision_as_dict
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop


class TauBenchToolAgentLoop(ToolAgentLoop):
    """veRL tool loop that injects tau-bench's initial simulated-user message."""

    def __init__(self, *args, adaptive_entropy: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_entropy_controller = AdaptiveEntropyController.from_config(adaptive_entropy)

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image"))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            interaction_name = interaction_kwargs.get("name")
            if not interaction_name:
                raise ValueError("'name' is required in interaction_kwargs")
            if interaction_name not in self.interaction_map:
                raise ValueError(f"Unknown interaction {interaction_name!r}")
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
            initial_user = await interaction.get_initial_response(request_id)
            messages.append({"role": "user", "content": initial_user})

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        try:
            base_sampling_params = dict(sampling_params)
            adaptive_decisions = []
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    step_sampling_params, decision = self._controlled_sampling_params(agent_data, base_sampling_params)
                    adaptive_decisions.append(decision_as_dict(decision))
                    agent_data.metrics.update(decision.to_metrics())
                    state = await self._handle_pending_state(agent_data, step_sampling_params)
                elif state == AgentState.GENERATING:
                    step_sampling_params, decision = self._controlled_sampling_params(agent_data, base_sampling_params)
                    adaptive_decisions.append(decision_as_dict(decision))
                    agent_data.metrics.update(decision.to_metrics())
                    state = await self._handle_generating_state(agent_data, step_sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    state = AgentState.TERMINATED

            score_info = await interaction.calculate_score(request_id) if interaction is not None else {"score": 0.0}
            final_score = float(score_info.get("score", 0.0))

            response_count = len(agent_data.response_mask)
            if response_count:
                response_ids = agent_data.prompt_ids[-response_count:]
                prompt_ids = agent_data.prompt_ids[:-response_count]
            else:
                response_ids = []
                prompt_ids = agent_data.prompt_ids
            multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=agent_data.response_mask[: self.response_length],
                multi_modal_data=multi_modal_data,
                response_logprobs=(
                    agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None
                ),
                num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                metrics=agent_data.metrics,
                extra_fields={
                    "final_score": final_score,
                    "score_info": score_info,
                    "turn_scores": agent_data.turn_scores,
                    "tool_rewards": agent_data.tool_rewards,
                    "adaptive_control": adaptive_decisions[-1] if adaptive_decisions else {},
                },
            )
            return output
        finally:
            if interaction is not None:
                await interaction.finalize_interaction(request_id)

    def _controlled_sampling_params(self, agent_data: AgentData, base_sampling_params: dict) -> tuple[dict, object]:
        state = None
        if agent_data.interaction is not None and hasattr(agent_data.interaction, "get_controller_state"):
            state = agent_data.interaction.get_controller_state(agent_data.request_id)
        return self.adaptive_entropy_controller.adjust_sampling_params(base_sampling_params, state)

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        next_state = await super()._handle_processing_tools_state(agent_data)
        if agent_data.interaction is not None and agent_data.interaction.is_done(agent_data.request_id):
            return AgentState.TERMINATED
        return next_state
