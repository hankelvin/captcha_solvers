import torch, hydra, os, json
SEED = 117438
torch.manual_seed(SEED)

@hydra.main(version_base = None, config_path = '', config_name = 'config.yaml')
def main(cfg):
    import yaml
    from omegaconf import OmegaConf
    from datasets import concatenate_datasets
    from zeroshot import give_data    
    from grpo_utils import prepare_datasets, give_grpo_trainer, load_model
    
    # 0. get some settings 
    hubpath = cfg.dirpath_hub
    if hubpath is not None:
        os.environ['HF_HOME']       = os.path.dirname(hubpath)
        os.environ['HF_HUB_CACHE']  = hubpath
    cfg.grpo_settings.max_seq_length = cfg.gen_args.max_new_tokens + cfg.grpo_settings.max_prompt_length
    cfg.gen_args.temperature = cfg.grpo_settings.temperature
    assert cfg.mode.startswith('think'), cfg.mode
    assert cfg.model in ['phi4mm'], cfg.model
    cfg.savepath                = os.path.join(cfg.dirpath, f'results/grpo_{cfg.model}_{cfg.size}_{cfg.mode}')
    cfg.savepath_train_outputs  = os.path.join(cfg.savepath, 'train_outputs.json')
    cfg.savepath_test_outputs   = os.path.join(cfg.savepath, 'test_outputs.json')

    # 1. load data 
    exemplars, eval_data = give_data(chars = set(i for i in cfg.common_chars))
    train_dataset, eval_dataset = prepare_datasets(cfg, exemplars, eval_data)
    full_dataset = concatenate_datasets([train_dataset, eval_dataset])

    # 2. set reward funcs
    reward_funcs = []
    from grpo_utils import (xmlcount_reward_func, soft_format_reward_func, 
                            strict_format_reward_func, no_xml_in_answer, think_length_reward_func,
                            num_correct_characters_reward_func, exact_match_reward_func,
                            characters_constraints_reward_func, num_characters_reward_func)
    reward_funcs        += [xmlcount_reward_func, soft_format_reward_func, 
                            strict_format_reward_func, no_xml_in_answer, think_length_reward_func,
                            num_correct_characters_reward_func, exact_match_reward_func,
                            characters_constraints_reward_func, num_characters_reward_func]
    reward_weights = [1.0 for rf in reward_funcs]
    
    # 3. load model and set up trainer 
    model, processor, peft_config = load_model(cfg)
    trainer = give_grpo_trainer(cfg, model, processor, train_dataset, eval_dataset, 
                                reward_funcs, reward_weights, peft_config)
    trainer.post_init()

    # 4. start post-training
    trainer.train()

    # 5a. evaluate on test split and save outputs, scores
    holder_eval_outputs, scores, error_tracker = \
        trainer.inference_on_eval_data(trainer.eval_dataset, bsz = 20, step = 'Test set on final step')
    with open(f'{cfg.savepath}/predictions.json', encoding = 'utf-8', mode = 'w+') as f:
        json.dump(holder_eval_outputs, f)
    with open(f'{cfg.savepath}/scores.json', encoding = 'utf-8', mode = 'w+') as f:
        json.dump(scores, f)
    with open(f'{cfg.savepath}/error_tracker.json', encoding = 'utf-8', mode = 'w+') as f:
        json.dump(error_tracker, f)
    print(cfg.savepath)
    print('#'*50 + '\n'*5)

    # 5b. output the predictions on the entire dataset
    holder_full_outputs, scores, error_tracker = \
        trainer.inference_on_eval_data(full_dataset, bsz = len(full_dataset), step = 'Full dataset')
    if not os.path.exists(f'{cfg.savepath}/output'): os.makedirs(f'{cfg.savepath}/output')
    for idx_num, __ in holder_full_outputs.items():
        fp = idx_num.replace('.jpg', '.txt').replace('input', 'output')
        with open(f'{cfg.savepath}/output/{fp}', encoding = 'utf-8', mode = 'w+') as f:
            f.write(__['predictions'][0])

    # 6. save model
    cfg_obj = OmegaConf.to_yaml(cfg)
    with open(f'{cfg.savepath}/config.yaml', encoding='utf-8', mode = 'w+') as f:
        yaml.dump(cfg_obj, f)

    # NOTE: use saved peft checkpoints instead
    # output_dir = os.path.join(cfg.savepath, 'model.ckpt')
    # trainer.accelerator.save_state(output_dir = output_dir, safe_serialization = False)

if __name__ == '__main__':
    main()