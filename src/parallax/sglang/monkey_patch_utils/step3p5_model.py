from typing import Optional

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def apply_step3p5_monkey_patch():
    """Patch Step3p5 forward path to support PP proxy tensors in Parallax."""
    import sglang.srt.models.step3p5 as step3p5_module

    @torch.no_grad()
    def pp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        model_output = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if isinstance(model_output, PPProxyTensors):
            return model_output

        if isinstance(model_output, tuple):
            hidden_states, hidden_states_before_norm = model_output
        else:
            # Backward compatibility with versions that return only hidden states.
            hidden_states, hidden_states_before_norm = model_output, None

        pp_group = getattr(self, "pp_group", None) or get_pp_group()
        if pp_group.is_last_rank:
            try:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    hidden_states_before_norm=hidden_states_before_norm,
                )
            except TypeError:
                return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        else:
            return hidden_states

    step3p5_module.Step3p5ForCausalLM.forward = pp_forward
