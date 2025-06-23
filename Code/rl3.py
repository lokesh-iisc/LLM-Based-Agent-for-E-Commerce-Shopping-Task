import warnings
from typing import Callable, Optional, Union, Any, List

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available
)
from verifiers import RewardFunc
from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.imports import LLM, SamplingParams
from verifiers.inference.vllm_client import VLLMClient

# monkey patch vllm client
import trl.extras.vllm_client
trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
from torch.nn.utils.rnn import pad_sequence
if is_wandb_available():
    import wandb



# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            scale_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        self.vllm_client = None
        # if not args.use_vllm: # type: ignore
        #     raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env
        self.scale_rewards = scale_rewards
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            repetition_penalty=self.repetition_penalty
        )

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        print("✅ Optimizer class:", optimizer.__class__)
        return optimizer


    def _generate_and_score_completions(self, instruction=None, inputs=None):
        device = self.accelerator.device
        if self.accelerator.is_main_process:
            # instruction = self.env.agent.start_session()
            env_results = []
            for _ in range(self.num_generations):
                result = self.env.generate(instruction)
                assert result["instruction"] == instruction  # sanity check
                env_results.append(result)
        else:
            env_results = None

        if env_results is None:
            return None
        

        query_prompt_ids_list = []
        query_prompt_mask_list = []
        query_completion_ids_list = []
        query_completion_mask_list = []

        selection_prompt_ids_list = []
        selection_prompt_mask_list = []
        selection_completion_ids_list = []
        selection_completion_mask_list = []

        query_old_per_token_logps_list = []
        query_ref_per_token_logps_list = []

        selection_old_per_token_logps_list = []
        selection_ref_per_token_logps_list = []

        reward_vals = []

        #########################################################################
        # instruction    = env_result["instruction"]         
        # product_block  = env_result["product_block"]       
        # completion_txt_1 = env_result["query"] 
        # completion_txt_2 = env_result["index"]        
        # reward_val     = env_result["reward"]           
        
        
        # process_slice = slice(
        #     self.accelerator.process_index * len(prompts),
        #     (self.accelerator.process_index + 1) * len(prompts),
        # )

        ###########################################################################################################################

        for result in env_results:
            instruction    = result["instruction"]         
            product_block  = result["product_block"]       
            completion_txt_1 = result["query"] 
            completion_txt_2 = result["index"]        
            reward_val     = result["reward"]

            prompt_query = [
                {"role": "system", "content": "You are a helpful assistant that writes concise e-commerce search queries."},
                {"role": "user", "content": (
                    f"Instruction: {instruction}\n\n"
                    "Now generate only the search query. Do not include any extra explanation or formatting. "
                    "Just write the query as a single line."
                )}
            ]
            prompt_text_1 = self.processing_class.apply_chat_template(prompt_query, tokenize=False) + "<|im_start|>assistant\n"

            prompt_tokenized_1 = self.processing_class(prompt_text_1, add_special_tokens=False)
            prompt_len_1 = len(prompt_tokenized_1["input_ids"])  

            full_text_1 = prompt_text_1 + completion_txt_1
            prompt_tok_1 = self.processing_class(
                full_text_1,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )


            full_tok_1 = Trainer._prepare_inputs(self, prompt_tok_1)
            input_ids_1 = full_tok_1["input_ids"]
            attention_mask_1 = full_tok_1["attention_mask"]

            prompt_ids_1 = input_ids_1[:, :prompt_len_1]
            completion_ids_1 = input_ids_1[:, prompt_len_1:]

            prompt_mask_1 = attention_mask_1[:, :prompt_len_1]
            completion_mask_1 = attention_mask_1[:, prompt_len_1:]

            if self.args.max_prompt_length is not None:
                prompt_ids_1 = prompt_ids_1[:, -self.args.max_prompt_length :]
                prompt_mask_1 = prompt_mask_1[:, -self.args.max_prompt_length :]

            if self.args.max_completion_length is not None:
                completion_ids_1 = completion_ids_1[:, :self.args.max_completion_length]
                completion_mask_1 = completion_mask_1[:, :self.args.max_completion_length]

            prompt_completion_ids_1 = torch.cat([prompt_ids_1, completion_ids_1], dim=1)
            attention_mask_1 = torch.cat([prompt_mask_1, completion_mask_1], dim=1)

            logits_to_keep_1 = completion_ids_1.size(1)

            with torch.no_grad():
                # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
                # computation here, and use per_token_logps.detach() instead.
                if self.num_iterations > 1:
                    query_old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_1, attention_mask_1, logits_to_keep_1
                    )
                else:
                    query_old_per_token_logps = None

                if self.beta == 0.0:
                    query_ref_per_token_logps = None
                elif self.ref_model is not None:
                    query_ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids_1, attention_mask_1, logits_to_keep_1
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        query_ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids_1, attention_mask_1, logits_to_keep_1
                        )
            
            # print("prompt_ids_1", prompt_ids_1)
            # print("completion_ids_1", completion_ids_1)
            # print("prompt_mask_1", prompt_mask_1)
            # print("completion_mask_1", completion_mask_1)

            ####################################################################################################################

            prompt_select = [
                {"role": "system", "content": "Pick the best product that matches the instruction."},
                {"role": "user", "content": f"""
            Instruction: {instruction}
            Search Query: {completion_txt_1}

            Here are the top-10 search results:
            {product_block}

            Please respond with **only a single number from 1 to 10**, representing the best matching product. Do not explain or include any other text. Just write the number.
            """}
            ]

            prompt_text_2 = self.processing_class.apply_chat_template(prompt_select, tokenize=False) + "<|im_start|>assistant\n"
            prompt_tokenized_2 = self.processing_class(prompt_text_2, add_special_tokens=False)
            prompt_len_2 = len(prompt_tokenized_2["input_ids"])

            if not isinstance(completion_txt_2, str):
                completion_txt_2 = str(completion_txt_2)

            full_text_2 = prompt_text_2 + completion_txt_2
            prompt_tok_2 = self.processing_class(
                full_text_2,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )

            full_tok_2 = Trainer._prepare_inputs(self, prompt_tok_2)
            input_ids_2 = full_tok_2["input_ids"]
            attention_mask_2 = full_tok_2["attention_mask"]

            prompt_ids_2 = input_ids_2[:, :prompt_len_2]
            completion_ids_2 = input_ids_2[:, prompt_len_2:]

            prompt_mask_2 = attention_mask_2[:, :prompt_len_2]
            completion_mask_2 = attention_mask_2[:, prompt_len_2:]

            if self.args.max_prompt_length is not None:
                prompt_ids_2 = prompt_ids_2[:, -self.args.max_prompt_length :]
                prompt_mask_2 = prompt_mask_2[:, -self.args.max_prompt_length :]

            if self.args.max_completion_length is not None:
                completion_ids_2 = completion_ids_2[:, :self.args.max_completion_length]
                completion_mask_2 = completion_mask_2[:, :self.args.max_completion_length]

            prompt_completion_ids_2 = torch.cat([prompt_ids_2, completion_ids_2], dim=1)
            attention_mask_2 = torch.cat([prompt_mask_2, completion_mask_2], dim=1)

            logits_to_keep_2 = completion_ids_2.size(1)

            with torch.no_grad():
                # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
                # computation here, and use per_token_logps.detach() instead.
                if self.num_iterations > 1:
                    selection_old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_2, attention_mask_2, logits_to_keep_2
                    )
                else:
                    selection_old_per_token_logps = None

                if self.beta == 0.0:
                    selection_ref_per_token_logps = None
                elif self.ref_model is not None:
                    selection_ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids_2, attention_mask_2, logits_to_keep_2
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        selection_ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids_2, attention_mask_2, logits_to_keep_2
                        )
            
            # print("prompt_ids_2", prompt_ids_2)
            # print("completion_ids_2", completion_ids_2)
            # print("prompt_mask_2", prompt_mask_2)
            # print("completion_mask_2", completion_mask_2)
            
            ######################################################################################################################

            query_prompt_ids_list.append(prompt_ids_1.squeeze(0))
            query_prompt_mask_list.append(prompt_mask_1.squeeze(0))
            query_completion_ids_list.append(completion_ids_1.squeeze(0))
            query_completion_mask_list.append(completion_mask_1.squeeze(0))

            selection_prompt_ids_list.append(prompt_ids_2.squeeze(0))
            selection_prompt_mask_list.append(prompt_mask_2.squeeze(0))
            selection_completion_ids_list.append(completion_ids_2.squeeze(0))
            selection_completion_mask_list.append(completion_mask_2.squeeze(0))

            if query_old_per_token_logps is not None:
                query_old_per_token_logps_list.append(query_old_per_token_logps)

            if query_ref_per_token_logps is not None:
                query_ref_per_token_logps_list.append(query_ref_per_token_logps)

            if selection_old_per_token_logps is not None:
                selection_old_per_token_logps_list.append(selection_old_per_token_logps)

            if selection_ref_per_token_logps is not None:
                selection_ref_per_token_logps_list.append(selection_ref_per_token_logps)


            reward_vals.append(reward_val)  

            mode = "eval" if self.control.should_evaluate else "train"

            # length of the generated completion (query + sep + index)
            completion_len = completion_mask_1.sum().item()          # batch = 1
            self._metrics[mode]["completion_length"].append(completion_len)

            # reward statistics (mean / std)
            self._metrics[mode]["reward"].append(reward_val)
            self._metrics[mode]["reward_std"].append(0.0)           

            if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
                if self.accelerator.is_main_process:
                    # console preview
                    print("▶︎", instruction[:80].replace("\n", " "),
                        "\n   ↳", completion_txt_1[:80], " | R =", reward_val)

                    # WandB table (only if you enabled report_to=["wandb"])
                    if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                        import pandas as pd, wandb
                        table = pd.DataFrame({
                            "step":        [self.state.global_step],
                            "instruction": [instruction],
                            "completion":  [completion_txt_1],
                            "reward":      [reward_val],
                        })
                        wandb.log({"samples": wandb.Table(dataframe=table)})          

        rewards = torch.tensor(reward_vals, dtype=torch.float32, device=device)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        print("Advantage", advantages)
        
        if self.scale_rewards:
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = advantages / (std_grouped_rewards + 1e-4)
        
        def pad_logps(logps_list):
            return pad_sequence([x.squeeze(0) for x in logps_list], batch_first=True, padding_value=0.0)


        return {
            "query_prompt_ids": pad_sequence(query_prompt_ids_list, batch_first=True, padding_value=self.processing_class.pad_token_id),
            "query_prompt_mask": pad_sequence(query_prompt_mask_list, batch_first=True, padding_value=0),
            "query_completion_ids": pad_sequence(query_completion_ids_list, batch_first=True, padding_value=self.processing_class.pad_token_id),
            "query_completion_mask": pad_sequence(query_completion_mask_list, batch_first=True, padding_value=0),

            "selection_prompt_ids": pad_sequence(selection_prompt_ids_list, batch_first=True, padding_value=self.processing_class.pad_token_id),
            "selection_prompt_mask": pad_sequence(selection_prompt_mask_list, batch_first=True, padding_value=0),
            "selection_completion_ids": pad_sequence(selection_completion_ids_list, batch_first=True, padding_value=self.processing_class.pad_token_id),
            "selection_completion_mask": pad_sequence(selection_completion_mask_list, batch_first=True, padding_value=0),

            "query_ref_per_token_logps": (pad_logps(query_ref_per_token_logps_list) if query_ref_per_token_logps_list else None),
            "query_old_per_token_logps": (pad_logps(query_old_per_token_logps_list) if query_old_per_token_logps_list else None),
            "selection_ref_per_token_logps": (pad_logps(selection_ref_per_token_logps_list) if selection_ref_per_token_logps_list else None),
            "selection_old_per_token_logps": (pad_logps(selection_old_per_token_logps_list) if selection_old_per_token_logps_list else None),

            "advantages": advantages,
        }
