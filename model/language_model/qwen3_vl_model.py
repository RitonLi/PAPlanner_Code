


import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from qwen3_vl.model.loss import soft_cross_entropy

from ...train.utils import calculate_loss_weight
from ..configuration_qwen3_vl import Qwen3VlConfig
from ..qwen3_vl_arch import Qwen3VlMetaForCausalLM, Qwen3VlMetaModel


class Qwen3VLModel(Qwen3VlMetaModel, Qwen3VlMetaForCausalLM, PreTrainedModel):
    config_class = Qwen3VlConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3VlConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(Qwen3VLModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        support_packing = "seqlens_in_batch" in inspect.signature(self.llm.forward).parameters

        if self.training and support_packing and not dpo_forward:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids

        if support_packing:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
        else:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # print(outputs)

        if self.training and self.config.time_token_ids:
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                new_labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )

        # Loss rescale for SP & DP loss match
        loss_weight = calculate_loss_weight(new_labels)
        outputs.loss = outputs.loss * loss_weight
        # outputs.loss = torch.nan_to_num(outputs.loss)
        # print(outputs.loss)

        if dpo_forward:
            return outputs.logits, new_labels

        return outputs


AutoConfig.register("qwen3_vl", Qwen3VlConfig)
AutoModel.register(Qwen3VlConfig, Qwen3VLModel)
