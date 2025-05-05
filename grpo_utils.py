import torch, re, os, json
import omegaconf
THINK_START, THINK_END  = '<think>', '</think>'
ANS_START, ANS_END      = '<answer>', '</answer>'
SEED = 117438

def prepare_datasets(cfg, exemplars, eval_data):
    import pandas as pd
    from datasets import Dataset

    def make_one_conversation_phi4mm(cfg, image_fp):
        dp = '' if cfg.dirpath is None else f'{cfg.dirpath}/'
        user_prompt         = '<|user|>'
        assistant_prompt    = '<|assistant|>'
        prompt_suffix       = '<|end|>'
        prompt_text         = ''
        image_files         = []
        
        mode_key    = 'zeroshot' if 'zeroshot' in cfg.mode else 'fewshot'
        think_key   = 'think_grpo' if 'grpo' in cfg.mode else 'think'
        if  cfg.mode.startswith('think'):
            prompt = cfg.prompt_library[think_key]
        else: prompt = cfg.prompt_library[mode_key]
        image_path = f'{dp}{cfg.inputs_path}/{image_fp}'
        image_files.append(image_path)
        prompt = f'{user_prompt}<|image_{len(image_files)}|>{prompt}{prompt_suffix}{assistant_prompt}'
        prompt_text += prompt
        return prompt_text, image_files

    holder_dataset_train = []
    for image_fp, label in exemplars.items():
        prompt_text, image_files = make_one_conversation_phi4mm(cfg, image_fp)
        holder_dataset_train.append({'image_fp': image_fp, 'label': label,
                                    'prompt_text': prompt_text, 'image_files': image_files})
    df_dataset_train = pd.DataFrame(holder_dataset_train)
    df_dataset_train.sample(frac = 1.0, random_state = SEED)
    train_dataset = Dataset.from_pandas(df_dataset_train)
    print('🟩🟩\tPREPARED TRAIN DATA...', df_dataset_train.shape, df_dataset_train.columns)

    holder_dataset_eval = []
    for image_fp, label in eval_data.items():
        prompt_text, image_files = make_one_conversation_phi4mm(cfg, image_fp)
        holder_dataset_eval.append({'image_fp': image_fp, 'label': label,
                                    'prompt_text': prompt_text, 'image_files': image_files})
    df_dataset_eval = pd.DataFrame(holder_dataset_eval)
    df_dataset_eval.sample(frac = 1.0, random_state = SEED)
    eval_dataset = Dataset.from_pandas(df_dataset_eval)
    print('🟩🟩\tPREPARED EVAL DATA...', df_dataset_eval.shape, df_dataset_eval.columns)
    return train_dataset, eval_dataset

def load_model(cfg):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers.utils import is_peft_available
    if   torch.mps.is_available():  device = 'mps'
    elif torch.cuda.is_available(): device = 'cuda'
    else:                           device = 'cpu'
    use_audio_in_video = False
    model_path         = cfg.models[cfg.model][cfg.size]    
    attn_implementation= 'flash_attention_2'
    if cfg.model in ['qwenomni']:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path,
                                                                    torch_dtype         = torch.bfloat16,
                                                                    device_map          = device,
                                                                    attn_implementation = attn_implementation)
        model.disable_talker()
        processor          = Qwen2_5OmniProcessor.from_pretrained(model_path)
        process_mm_info    = process_mm_info
    else: 
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, GenerationConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code = True)
        config.embd_layer['audio_embd_layer']['enable_gradient_checkpointing'] = False
        config.embd_layer['image_embd_layer']['enable_gradient_checkpointing'] = False
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code = True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map = device, 
            torch_dtype = torch.bfloat16, 
            trust_remote_code = True,
            config = config,
            attn_implementation = attn_implementation).cuda()
        # Load generation config
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        process_mm_info = None
    model.model_path = model_path

    # NOTE: peft set up delayed to GRPOTrainer initialisation
    print('🟧🟧\tCREATING PEFT CONFIG')
    task_type = TaskType.CAUSAL_LM    
    lora_rank = cfg.grpo_settings.lora_rank
    lora_alpha = cfg.grpo_settings.lora_alpha_ratio * lora_rank
    peft_config = LoraConfig(
        target_modules = list(cfg.grpo_settings.target_modules), 
        # exclude_modules= 'audio_embed',
        task_type      = task_type, 
        inference_mode = False,
        r              = lora_rank, 
        lora_alpha     = lora_alpha, 
        lora_dropout   = cfg.grpo_settings.lora_dropout,
        use_dora       = cfg.grpo_settings.use_dora,
        use_rslora     = True,            )
    
    if not is_peft_available():
        raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
    model = get_peft_model(model, peft_config)
    peft_config = None
    
    model.train()
    return model, processor, peft_config

def give_grpo_trainer(cfg, model, processor, train_dataset, 
                        eval_dataset, reward_funcs = [], 
                        reward_weights = None, peft_config = None): 
    from trl import GRPOConfig
    from grpo_mods import GRPOTrainerMod
    bf16 = True
    fp16 = False 

    ##### TRAINER ARGS #####    
    training_args = GRPOConfig(
        use_vllm                = False,
        beta                    = cfg.grpo_settings.beta, 
        epsilon                 = cfg.grpo_settings.epsilon,
        reward_weights          = reward_weights,
        sync_ref_model          = cfg.grpo_settings.sync_ref_model,         
        ref_model_mixup_alpha   = cfg.grpo_settings.ref_model_mixup_alpha,  
        ref_model_sync_steps    = cfg.grpo_settings.ref_model_sync_steps,   
        learning_rate           = cfg.grpo_settings.learning_rate,
        adam_beta1              = 0.9,
        adam_beta2              = 0.99,
        weight_decay            = 0.1,
        warmup_ratio            = cfg.grpo_settings.warmup_ratio,
        lr_scheduler_type       = 'cosine',
        optim                   = 'paged_adamw_8bit',
        logging_steps           = 20,
        bf16                    = bf16,
        fp16                    = fp16,
        # per_device_train_batch_size should be a multple of num_generations 
        per_device_train_batch_size = cfg.grpo_settings.gen_bsz,  
        num_generations         = cfg.grpo_settings.num_cands, 
        num_iterations          = cfg.grpo_settings.num_iterations,
        gradient_accumulation_steps = cfg.grpo_settings.gradient_accumulation_steps,
        gradient_checkpointing  = False,
        # gradient_checkpointing_kwargs = {'use_reentrant': False},
        max_prompt_length       = cfg.grpo_settings.max_prompt_length,
        max_completion_length   = int(cfg.grpo_settings.max_seq_length),
        save_steps              = cfg.grpo_settings.save_steps,
        save_total_limit        = 20,
        max_grad_norm           = 0.1,
        remove_unused_columns   = False, 
        report_to               = 'none', 
        log_level               = 'info',
        save_strategy           = 'steps',
        output_dir              = os.path.join(cfg.savepath, 'checkpoint_outputs'),
        ddp_find_unused_parameters = False,
        epsilon_high            = cfg.grpo_settings.epsilon_high,
        scale_rewards           = cfg.grpo_settings.scale_rewards,
        mask_truncated_completions = cfg.grpo_settings.mask_truncated_completions,
        )
    
    if cfg.grpo_settings.max_steps is None: 
        training_args.num_train_epochs  = 1, # Set to 1 for a full training run
    else: 
        training_args.max_steps         = cfg.grpo_settings.max_steps
    
    # force attach (GRPOConfig inherits TrainingArguments, which has generation_config)
    training_args.temperature           = cfg.grpo_settings.temperature
    training_args.num_return_sequences  = cfg.grpo_settings.num_cands
    training_args.ddp_backend           = None 
    
    ##### TRAINER ##### 
    trainer = GRPOTrainerMod(
        model               = model,
        processing_class    = processor.tokenizer,
        reward_funcs        = reward_funcs,
        args                = training_args,
        train_dataset       = train_dataset,
        eval_dataset        = eval_dataset,
        peft_config         = peft_config, 
        ) 
    trainer.processor            = processor
    # ensure _prepare_inputs can access cfg info. 
    trainer.cfg                  = convert_to_json_compatible(cfg)
    trainer.training_args        = training_args
    
    return trainer

def set_peft_weights(cfg, model, adapter_name):
    '''NOTE: model here assumes a non-pipeline object and peft has already been applied '''
    import gc
    from peft.utils import load_peft_weights
    load_peft_ckpt_path = cfg['load_peft_ckpt_path']
    if not load_peft_ckpt_path.endswith('checkpoint-0'):
        # load the checkpoint peft weights
        peft_model_state_dict = load_peft_weights(load_peft_ckpt_path)
        # add adapter name prefix before .weight
        
        peft_model_state_dict = fix_peft_weights(peft_model_state_dict, adapter_name, model.do_compile)
        seen = set()
        peft_keys = set(peft_model_state_dict.keys())
        for mn, mw in model.named_parameters():
            if mn in peft_model_state_dict:
                # print('FOUND MATCH AND ASSIGNED', mn)
                # print(mw.data.mean(dim=0), peft_model_state_dict[mn].data.mean(dim =0))
                dtype = mw.data.dtype
                mw.data = peft_model_state_dict[mn].data.clone().to(dtype)
                seen.add(mn)
        # ensure all checkpoint param weights used
        assert seen == set(peft_model_state_dict.keys()), \
                (seen.difference(peft_keys), peft_keys.difference(seen))
        print('👀 PEFT WEIGHTS SUCCESSFULLY LOADED FROM:', load_peft_ckpt_path)
        model.set_adapter(adapter_name)   
        print('👀 ACTIVE ADAPTER', model.active_adapter)
        del peft_model_state_dict
        gc.collect()
        torch.cuda.empty_cache()
    else: 
        print(f'👀 PEFT WEIGHTS set at {load_peft_ckpt_path}, i.e. ZEROSHOT...')
        model.unload()   
        model.delete_adapter(adapter_name)
        for mn, mw in model.named_parameters():
            assert '.lora' not in mn.lower(), mn

    return model

def fix_peft_weights(peft_model_state_dict, adapter_name, do_compile):
    peft_model_state_dict = {k.replace('.weight', f'.{adapter_name}.weight'): v \
                                        for k, v in peft_model_state_dict.items()}
    
    check_peft_weights_compiled = all('._orig_mod' in k for k,v in peft_model_state_dict.items())
    if do_compile and not check_peft_weights_compiled:
        peft_model_state_dict = {k.replace('.model.model', '.model._orig_mod.model'): v \
                                        for k, v in peft_model_state_dict.items()}
    elif not do_compile and check_peft_weights_compiled:
        peft_model_state_dict = {k.replace('.model._orig_mod.model', '.model.model'): v \
                                        for k, v in peft_model_state_dict.items()}
        
    return peft_model_state_dict

def convert_to_json_compatible(cfg):

    if isinstance(cfg, (omegaconf.ListConfig, list)):
        return [convert_to_json_compatible(item) for item in cfg]
    
    elif isinstance(cfg, (omegaconf.DictConfig, dict)):
        return {key: convert_to_json_compatible(value) for key, value in cfg.items()}
    
    else:
        return cfg

######################################
########## REWARD FUNCTIONS ##########
######################################
# format reward func from: 
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    '''Reward function that checks if the completion has a specific format.'''
    # NOTE: forces newline. but soft_format_reward_func relaxes this
    pattern = rf'^{THINK_START}\n.*?\n{THINK_END}\n{ANS_START}\n.*?\n{ANS_END}\n$'
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    '''Reward function that checks if the completion has a specific format.'''
    pattern = rf'{THINK_START}.*?{THINK_END}\s*{ANS_START}.*?{ANS_END}'
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count(f'{THINK_START}\n') == 1:
        count += 0.125
    if text.count(f'\n{THINK_END}\n') == 1:
        count += 0.125
    if text.count(f'\n{ANS_START}\n') == 1:
        count += 0.125
        count -= len(text.split(f'\n{ANS_END}\n')[-1])*0.001
    if text.count(f'\n{ANS_END}') == 1:
        count += 0.125
        count -= (len(text.split(f'\n{ANS_END}')[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]

def extract_xml_answer_no_tags(text: str) -> str:
    answer = text.split(f'{ANS_START}')[-1]
    answer = answer.split(f'{ANS_END}')[0]
    return answer.strip()

def extract_xml_answer_strict_with_tags(text: str) -> str:
    answer = re.search(rf'{ANS_START}.+{ANS_END}', text)
    if answer:     return answer.group(0).strip()
    else:               return ''

def no_xml_in_answer(completions, **kwargs):
    answers = [extract_xml_answer_no_tags(c) for c in completions]
    scores = [0.0 if not c or re.search(r'<\w+>.+</\w+>', c) else 0.5 for c in answers] 
    return scores

def extract_think_words_strict_no_tags(completion) -> str:
    think_words = re.search(rf'{THINK_START}(.*){THINK_END}', completion)
    if think_words:     return think_words.group(0).strip()
    else:               return ''

def extract_think_words_strict_with_tags(completion) -> str:
    think_words = re.search(rf'{THINK_START}.+{THINK_END}', completion)
    if think_words:     return think_words.group(0).strip()
    else:               return ''

def think_length_reward_func(completions, **kwargs) -> list[float]:
    scores = []
    for c in completions:
        think_words = extract_think_words_strict_no_tags(c)
        if think_words:
            len_think_words = len(think_words)

            if   250 <= len_think_words <= 350: scores.append(0.5)
            elif 175 <= len_think_words <  250: scores.append(0.25)
            elif 100 <= len_think_words <  175: scores.append(0.125)
            elif 25 <= len_think_words  <  100: scores.append(0.075)
            elif len_think_words        <   25: scores.append(-0.25)
            elif 350 < len_think_words  <= 400: scores.append(0.25)
            elif 400 < len_think_words  <= 450: scores.append(0.125)
            elif len_think_words        >  450: scores.append(0.0)
            
        else: scores.append(0.0)
    
    return scores

# task-specific RLVR reward funcs
def num_correct_characters_reward_func(completions, label, **kwargs) -> list[float]:
    scores = []
    assert len(completions) == len(label)
    for c, l in zip(completions, label):
        a = extract_xml_answer_no_tags(c).strip()
        s = [1 if aa==ll else 0 for ll, aa in zip(l, a)]
        # could be empty string
        if s: scores.append(sum(s)/len(s)*0.5)
        else: scores.append(0.0)
    return scores

def exact_match_reward_func(completions, label, **kwargs) -> list[float]:
    scores = []
    assert len(completions) == len(label)
    for c, l in zip(completions, label):
        a = extract_xml_answer_no_tags(c).strip()
        if a == l: scores.append(0.5)
        else:      scores.append(0.0)
    return scores

def characters_constraints_reward_func(completions, **kwargs) -> list[float]:
    scores = []
    check  = set([i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'])
    for c in completions:
        a = set(i for i in extract_xml_answer_no_tags(c).strip())
        if len(a.difference(check)) > 0: scores.append(0.0)
        else:                            scores.append(0.5)
    return scores

def num_characters_reward_func(completions, label, **kwargs) -> list[float]:
    scores = []
    assert len(completions) == len(label)
    for c, l in zip(completions, label):
        a = extract_xml_answer_no_tags(c)
        if len(a) == len(l): scores.append(0.5)
        else:                scores.append(0.0)
    return scores