import torch, tqdm, math, warnings, json
import numpy as np
from trl.data_utils import apply_chat_template, is_conversational
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import nanstd, nanmin, nanmax
from trl.trainer.utils import pad, selective_log_softmax
from trl.extras.profiling import profiling_context, profiling_decorator
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model
from collections import defaultdict, Counter
import torch.nn as nn
from typing import Any, Union
from trl import GRPOTrainer
from PIL import Image
from zeroshot import extract_results, compute_scores

######################################
########## MODS TO TRL GRPOTRainer ###
######################################
class GRPOTrainerMod(GRPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ### CHANGE START ###
    def post_init(self):
        print('👀 PEFT MODEL STATUS -->', is_peft_model(self.model))
        adapter_name = 'default'
        if self.cfg['load_peft_ckpt_path'] is not None:    
            from utils_model import set_peft_weights
            self.model  = set_peft_weights(self.cfg, self.model, adapter_name)
        else: self.model.set_adapter(adapter_name)      
        
        self.model.print_trainable_parameters()   
        self.model.is_model_parallel = False
        self.generated_outputs          = defaultdict(dict)
        self.holder_eval_outputs_status = set()
        if getattr(self, 'temperature', None) is None: 
            self.temperature = self.training_args.temperature

    def inference_on_eval_data(self, eval_dataset, bsz = 8, step = None):
        if self.args.use_vllm: raise NotImplementedError
        assert len(eval_dataset) > 0, len(eval_dataset)
        num_batches = max(1, math.ceil(len(eval_dataset)/bsz))
        self.model.eval()
        device = self.accelerator.device
        
        holder_eval_outputs = {}
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for __, bn in enumerate(tqdm.tqdm(range(num_batches))):
                inputs      = eval_dataset[bn * bsz : (bn+1) * bsz] # slice of Dataset object is a dict
                image_fp    = inputs['image_fp']
                labels      = inputs['label']
                inputs_holder, prompts_text = self.make_inputs_holder(inputs, device)
                prompt_ids  = inputs_holder['input_ids']
                prompt_mask = inputs_holder['attention_mask']

                if self.max_prompt_length is not None:
                    if prompt_ids.size(-1) > self.max_prompt_length: 
                        print('PROMPT LENGTH: ', prompt_ids.size(-1), self.max_prompt_length)
                    inputs_holder['input_ids']      =  prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                    inputs_holder['attention_mask'] = prompt_mask = prompt_mask[:, -self.max_prompt_length :]
                
                # Regular generation path
                with torch.no_grad():
                    prompt_completion_ids = unwrapped_model.generate(**inputs_holder, 
                                                                     max_new_tokens = self.cfg['gen_args']['max_new_tokens'],
                                                                     generation_config = self.generation_config)
            
                # Compute prompt length and extract completion ids
                prompt_completion_texts = self.processing_class.batch_decode(prompt_completion_ids, skip_special_tokens=True)
                prefills, completions, predictions = extract_results(self.cfg['mode'], prompt_completion_texts, 
                                                        prompts_text, self.model.model_path, self.processor)
                assert len(prompts_text) == len(prompt_completion_texts) == len(image_fp) == len(predictions), \
                    (len(prompts_text), len(prompt_completion_texts), len(image_fp), len(predictions))
                
                for pt, ct, pr, ll, idx_num in zip(prompts_text, completions, predictions, labels, image_fp):
                    holder_eval_outputs[idx_num] = {'prompts': [pt], 'generated': [ct], 'labels': [ll], 'predictions': [pr]}

                print(f'STEP: {step}', '\n', '#'*50)
                scores, error_tracker = compute_scores(holder_eval_outputs)
                print('MEAN SCORES:', round(np.mean(list(scores.values())),2))
                print('DIST SCORES:', Counter(list(scores.values())).most_common())
                print('ERRORS', {char: Counter(v) for char, v in error_tracker.items()})
        
        return holder_eval_outputs, scores, error_tracker
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, num_logits_to_keep, 
                             batch_size=None, **model_kwargs) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            ### CHANGE START ###
            model_kwargs_batch = {k:v[i : i + batch_size] if v is not None else v for k, v in model_kwargs.items()}
            # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, num_logits_to_keep=num_logits_to_keep + 1,
                ### CHANGE START ###
                **model_kwargs_batch
                ### CHANGE END ###
            ).logits

            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -num_logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -num_logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)
    
    def make_inputs_holder(self, inputs, device):
        inputs_holder   = defaultdict(list)
        if      type(inputs) is dict:
            prompts_text    = [x for x in inputs['prompt_text']]
            image_files     = inputs['image_files']
        elif    type(inputs) is list: 
            prompts_text    = [x["prompt_text"] for x in inputs]
            image_files     = [x["image_files"] for x in inputs]
        assert len(image_files) == len(prompts_text)
        for p_t, i_f in zip(prompts_text, image_files):
            images  = [Image.open(fp) for fp  in i_f]
            __      = self.processor(text = p_t, images = images, return_tensors = 'pt')
            for k,v in __.items(): 
                if k in ['image_sizes']: v = v[0]
                inputs_holder[k].append(v)
        for k,v in inputs_holder.items(): 
            if v[0] is None: inputs_holder[k] = None
            else: 
                if k in ['input_ids', 'input_image_embeds', 'attention_mask', 'image_attention_mask', 
                        'input_audio_embeds', 'audio_attention_mask']:
                    dtype = v[0].dtype
                    inputs_holder[k] = torch.cat(v, dim = 0).to(device).to(dtype)
                elif k in ['image_sizes', 'audio_embed_sizes', 'input_audio_embeds']: 
                    pass
                else: 
                    dtype = v[0].dtype
                    inputs_holder[k] = torch.stack(v, dim = 0).to(device).to(dtype)
        return inputs_holder, prompts_text

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        ### CHANGE START ###
        # prompts = [x["prompt"] for x in inputs]
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        # prompt_inputs = self.processing_class(
        #     text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        # )
        # prompt_inputs = super()._prepare_inputs_Trainer(prompt_inputs)
        # prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        warnings.simplefilter("once")
        inputs_holder, prompts_text = self.make_inputs_holder(inputs, device)
        prompts         = prompts_text
        prompt_ids      = inputs_holder['input_ids']
        prompt_mask     = inputs_holder['attention_mask']
        ### CHANGE END ###

        if self.max_prompt_length is not None:
            ### CHANGE START ###
            if prompt_ids.size(-1) > self.max_prompt_length: 
                print('PROMPT LENGTH: ', prompt_ids.size(-1), self.max_prompt_length)
            inputs_holder['input_ids']      =  prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            inputs_holder['attention_mask'] = prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            ### CHANGE END ###

        ### CHANGE START ###
        # image and audio embeds, their masks, their sizes and the input modes
        vlm_mm_inputs = {k: v for k, v in inputs_holder.items() if k not in ["input_ids", "attention_mask"]}
        ### CHANGE END ###

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                ### CHANGE START ###
                # prompt_completion_ids = unwrapped_model.generate(
                #     prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                # )
                prompt_completion_ids = unwrapped_model.generate(**inputs_holder, 
                                                                max_new_tokens = self.cfg['gen_args']['max_new_tokens'],
                                                                generation_config = self.generation_config)
                ### CHANGE END ###

            ### CHANGE START ###
            # Compute prompt length and extract completion ids
            # prompt_length = prompt_ids.size(1)
            # prompt_ids = prompt_completion_ids[:, :prompt_length]
            # completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_completion_texts = self.processing_class.batch_decode(prompt_completion_ids, 
                                            skip_special_tokens = True, clean_up_tokenization_spaces = False)
            prefills, completions_text, results = extract_results(self.cfg['mode'], prompt_completion_texts, 
                                                        prompts_text, self.model.model_path, self.processor)
            completion_ids = self.processing_class(
                text = completions_text, return_tensors = "pt", padding = True, padding_side = "left", 
                add_special_tokens = False)['input_ids']
            dtype = completion_ids.dtype
            completion_ids = completion_ids.to(device).to(dtype)
            ### CHANGE END ###

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        is_eos = is_eos.to(device)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        ### CHANGE START ### phi4mm is at transformers==4.48 where 'logits_to_keep' was 'num_logits_to_keep'
        # logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        num_logits_to_keep = completion_ids.size(1)  
        ### CHANGE START ###
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    ### CHANGE START ### 
                    self.model, prompt_completion_ids, attention_mask, num_logits_to_keep, batch_size,
                    **vlm_mm_inputs
                    ### CHANGE END ###
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    ### CHANGE START ###
                    self.ref_model, prompt_completion_ids, attention_mask, num_logits_to_keep, batch_size,
                    **vlm_mm_inputs
                    ### CHANGE END ###
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        ### CHANGE START ###
                        self.model, prompt_completion_ids, attention_mask, num_logits_to_keep, batch_size,
                        **vlm_mm_inputs
                        ### CHANGE END ###
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        ### CHANGE START ###
        if self.is_in_train and self.state.global_step % self.cfg['grpo_settings']['save_steps'] == 0:
            fp = self.cfg['savepath_train_outputs'].replace('.json', f'_step{self.state.global_step}.json')
            with open(fp, encoding= 'utf-8', mode = 'w+') as f:
                json.dump(self.generated_outputs, f)
            print(f'🔮🔮 Train outputs at {self.state.global_step} saved to: ', fp)

            # necessary check: global step only increments per gradient_accumulation_steps
            c1 = self.state.global_step not in self.holder_eval_outputs_status
            c2 = self.cfg['grpo_settings']['eval_every_save_step']
            if c1 and c2: 
                holder_eval_outputs, scores, error_tracker = \
                    self.inference_on_eval_data(self.eval_dataset, bsz = 20, step = self.state.global_step)
                
                fp = self.cfg['savepath_test_outputs'].replace('.json', f'_step{self.state.global_step}.json')
                with open(fp, encoding = 'utf-8', mode = 'w+') as f:
                    json.dump(holder_eval_outputs, f)
                self.holder_eval_outputs_status.add(self.state.global_step)
                print(f'🔮🔮 Eval at {self.state.global_step} saved to: ', fp)
                self.model.train()
        ### CHANGE END ###

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            ### CHANGE START ###
            "input_image_embeds":   inputs_holder["input_image_embeds"],
            "image_sizes":          inputs_holder["image_sizes"], 
            "image_attention_mask": inputs_holder["image_attention_mask"], 
            "input_audio_embeds":   inputs_holder["input_audio_embeds"],
            "audio_embed_sizes":    inputs_holder["audio_embed_sizes"], 
            "audio_attention_mask": inputs_holder["audio_attention_mask"],
            "input_mode":           inputs_holder["input_mode"],
            ### CHANGE END ###
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        ### CHANGE START ###
        # logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        num_logits_to_keep = completion_ids.size(1)  
        keys = ['input_image_embeds', 'image_sizes', 'image_attention_mask', 'input_audio_embeds', 
                'audio_embed_sizes', 'audio_attention_mask', 'input_mode']
        vlm_mm_inputs = {k: v for k, v in inputs.items() if k in keys}

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, 
                                                    num_logits_to_keep, **vlm_mm_inputs)
        ### CHANGE END ###

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        ### CHANGE START ###
                        self.ref_model, input_ids, attention_mask, num_logits_to_keep, **vlm_mm_inputs,
                        ### CHANGE END ###
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            ### CHANGE START ###
                            self.model, input_ids, attention_mask, num_logits_to_keep, **vlm_mm_inputs,
                            ### CHANGE END ###
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss
    