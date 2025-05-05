import itertools, json
import torch, hydra, re, random
from collections import Counter, defaultdict
from PIL import Image
SEED = 117438
torch.manual_seed(SEED)

@hydra.main(version_base = None, config_path = '', config_name = 'config.yaml')
def main(cfg):
    USE_AUDIO_IN_VIDEO = False
    import os, json, tqdm, numpy as np, time
    hubpath = cfg.dirpath_hub
    if hubpath is not None:
        os.environ['HF_HOME']       = os.path.dirname(hubpath)
        os.environ['HF_HUB_CACHE']  = hubpath

    # 2. load data 
    exemplars, holder_data = give_data(chars = set(i for i in cfg.common_chars))
    if 'zeroshot' in cfg.mode: exemplars = {}
    else: 
        if   'oneshot'  in cfg.mode: k = 1
        elif 'sixshot'  in cfg.mode: k = 6
        else: raise NotImplementedError
        picks = random.sample(list(exemplars), k = k)
        exemplars = {pick: exemplars[pick] for pick in picks}
    vlm = VLM(cfg, USE_AUDIO_IN_VIDEO, exemplars)

    predictions = {}
    start = time.time()
    for __, (fp,label) in enumerate(tqdm.tqdm(holder_data.items())):
        predictions[fp] =  vlm.inference_one_file(fp, label)
    avg_time = round((time.time()-start)/len(predictions), 2)
    
    scores, error_tracker = compute_scores(predictions)
    print('#'*50)
    print('AVG TIME   :', avg_time)
    print('MEAN SCORES:', round(np.mean(list(scores.values())),2))
    print('DIST SCORES:', Counter(list(scores.values())).most_common())
    print('ERRORS', {char: Counter(v) for char, v in error_tracker.items()})

    dp = dp = '' if cfg.dirpath is None else f'{cfg.dirpath}/'
    dp = f'{dp}results/{cfg.model}_{cfg.size}_{cfg.mode}'
    if not os.path.exists(dp): os.makedirs(dp)
    with open(f'{dp}/predictions.json', encoding = 'utf-8', mode = 'w+') as f:
        json.dump(predictions, f)
    with open(f'{dp}/scores.json', encoding = 'utf-8', mode = 'w+') as f:
        json.dump(scores, f)
    with open(f'{dp}/error_tracker.json', encoding = 'utf-8', mode = 'w+') as f:
        json.dump(error_tracker, f)
    print(dp)
    print('#'*50 + '\n'*5)
        
class VLM:
    def __init__(self, cfg, use_audio_in_video, exemplars):
        if   torch.mps.is_available():  device = 'mps'
        elif torch.cuda.is_available(): device = 'cuda'
        else:                           device = 'cpu'
        
        self.cfg                = cfg
        self.use_audio_in_video = use_audio_in_video
        self.exemplars          = exemplars
        # 1. load model and media processor
        self.model_path         = cfg.models[cfg.model][cfg.size]    
        attn_implementation     = 'flash_attention_2'
        if cfg.model in ['qwenomni']:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            from qwen_omni_utils import process_mm_info
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.model_path,
                                                                            torch_dtype         = torch.bfloat16,
                                                                            device_map          = device,
                                                                            attn_implementation = attn_implementation)
            self.model.disable_talker()
            self.processor          = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
            self.process_mm_info    = process_mm_info
        else: 
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code = True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map = device, 
                torch_dtype = torch.bfloat16, 
                trust_remote_code = True,
                _attn_implementation = attn_implementation).cuda()

            # Load generation config
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
            self.process_mm_info = None


    def make_one_conversation_qwenomni(self, image_fp):
        dp = '' if self.cfg.dirpath is None else f'{self.cfg.dirpath}/'
        conversation = [{'role': 'system',
                        'content': [{'type': 'text', 'text': self.cfg.prompt_library.system}],},]
        
        mode_key    = 'zeroshot' if 'zeroshot' in self.cfg.mode else 'fewshot'
        think_key   = 'think_grpo' if 'grpo' in self.cfg.mode else 'think'
        if self.exemplars: 
            for oneshot_fp, oneshot_label in self.exemplars.items():
                if  self.cfg.mode.startswith('think'):
                    prompt  = self.cfg.prompt_library[think_key]
                    label = self.cfg.prompt_library['label_think_fewshot'][oneshot_fp]
                else: 
                    prompt  = self.cfg.prompt_library[mode_key]
                    label   = oneshot_label
                image_path = f'{dp}{self.cfg.inputs_path}/{oneshot_fp}'
                conversation.append({'role': 'user',
                                    'content': [{'type': 'image', 'image': image_path},
                                                {'type': 'text', 'text': prompt },]})
                conversation.append({'role': 'assistant',
                                    'content': [{'type': 'text', 'text':  label},],})

        if  self.cfg.mode.startswith('think'):
            prompt = self.cfg.prompt_library[think_key]
        else: prompt = self.cfg.prompt_library[mode_key]
        image_path = f'{dp}{self.cfg.inputs_path}/{image_fp}'
        conversation.append({'role': 'user',
                            'content': [{'type': 'image', 'image': image_path},
                                        {'type': 'text', 'text':  prompt},],})
        return conversation
    
    def make_one_conversation_phi4mm(self, image_fp):
        dp = '' if self.cfg.dirpath is None else f'{self.cfg.dirpath}/'
        user_prompt         = '<|user|>'
        assistant_prompt    = '<|assistant|>'
        prompt_suffix       = '<|end|>'
        prompt_text         = ''
        image_files         = []
        
        mode_key    = 'zeroshot' if 'zeroshot' in self.cfg.mode else 'fewshot'
        think_key   = 'think_grpo' if 'grpo' in self.cfg.mode else 'think'
        if self.exemplars: 
            for oneshot_fp, oneshot_label in self.exemplars.items():
                if  self.cfg.mode.startswith('think'):
                    prompt  = self.cfg.prompt_library[think_key]
                    label = self.cfg.prompt_library['label_think_fewshot'][oneshot_fp]
                else: 
                    prompt  = self.cfg.prompt_library[mode_key]
                    label   = oneshot_label
                image_path = f'{dp}{self.cfg.inputs_path}/{oneshot_fp}'
                image_files.append(image_path)
                prompt = f'{user_prompt}<|image_{len(image_files)}|>{prompt}{prompt_suffix}{assistant_prompt}{label}{prompt_suffix}'
                prompt_text += prompt

        if  self.cfg.mode.startswith('think'):
            prompt = self.cfg.prompt_library[think_key]
        else: prompt = self.cfg.prompt_library[mode_key]
        image_path = f'{dp}{self.cfg.inputs_path}/{image_fp}'
        image_files.append(image_path)
        prompt = f'{user_prompt}<|image_{len(image_files)}|>{prompt}{prompt_suffix}{assistant_prompt}'
        prompt_text += prompt
        return prompt_text, image_files
    
    def inference_one_file(self, image_fp, label):
        if self.cfg.model in ['qwenomni']:
            conversation = self.make_one_conversation_qwenomni(image_fp)
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt = True, tokenize = False)
            audios, images, videos  = self.process_mm_info(conversation, use_audio_in_video = self.use_audio_in_video)
            inputs                  = self.processor(text = prompt_text, audio = audios, images = images, videos = videos, 
                                            return_tensors = "pt", padding = True, use_audio_in_video = self.use_audio_in_video)
            inputs                  = inputs.to(self.model.device).to(self.model.dtype)
            with torch.no_grad():
                prompt_completion_ids   = self.model.generate(**inputs, **self.cfg.gen_args,
                                                              use_audio_in_video = self.use_audio_in_video)
        
        elif self.cfg.model in ['phi4mm']:
            prompt_text, image_files = self.make_one_conversation_phi4mm(image_fp)
            images = [Image.open(fp) for fp in image_files]
            inputs = self.processor(text = prompt_text, images = images, return_tensors='pt')
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            with torch.no_grad():
                prompt_completion_ids   = self.model.generate(**inputs, 
                                                              generation_config = self.model.generation_config)
            # HACK for extract_results to align with qwenomni treatment
            prompt_text = [prompt_text]
        
        else: raise NotImplementedError
        
        prompt_completion_texts     = self.processor.batch_decode(prompt_completion_ids, 
                                    skip_special_tokens = True, clean_up_tokenization_spaces = False)
        assert len(prompt_completion_texts) == 1 and len(prompt_text) == 1

        prefills, completions, predictions = extract_results(self.cfg['mode'], prompt_completion_texts, 
                                                prompt_text, self.model_path, self.processor)
        
        return {'prompts': prefills, 'generated': prompt_completion_texts, 
                'labels': [label], 'predictions': predictions}
    
def extract_results(mode, prompt_completion_texts, prompt_text, model_path, processor):
    # get prefill size to trim from generated outputs
    c1 = model_path.startswith('Qwen/Qwen2.5-Omni') 
    c2 = model_path.startswith('microsoft/Phi-4-multimodal-instruct')
    if  c1 or c2:
        prefills = processor.tokenizer(prompt_text)['input_ids']
        prefills = processor.batch_decode(prefills, skip_special_tokens = True, clean_up_tokenization_spaces = False)
        if c2:
            # HACK: skip_special_tokens not removing <|image_1|>, these are missing in prompt_completion_texts (i.e. gen outputs)
            prefills = [re.sub(r'<\|image_\d\|>', '', pref) for pref in prefills]
            prompt_completion_texts = [re.sub(r'<\|image_\d\|>', '', pr_co) for pr_co in prompt_completion_texts]
        assert len(prompt_completion_texts) == len(prefills), (len(prompt_completion_texts), len(prefills))
        completions = [pr_co[len(pref):].strip() for pr_co, pref in zip(prompt_completion_texts, prefills)]

    else: raise NotImplementedError    

    if mode.startswith('think'):
        predictions = []
        for c in completions:
            __result = re.search(r'<answer>(.*)</answer>', c)
            if __result: c = __result.group(1).strip()
            else:      
                # might be poorly formatted in terms of tags. extract 5-char seq that meets specification
                try:    c = re.findall(r'\b([A-Z0-9]{5})\b', c)[-1]
                except: c = ''
            predictions.append(c)
    else: predictions = completions
    return prefills, completions, predictions

def find_exemplar(test_pool, cands, r):
    found_exp = {}
    used_char = set()
    for c in cands:
        for k,v in test_pool.copy().items():
            if len(set(c).intersection(set(v))) == r:
                found_exp[k] = v
                used_char.update(c)
                test_pool.pop(k)
    return test_pool, found_exp, used_char


def give_data(chars = set(i for i in 'Q0O1I5S2Z')):
    '''
    from the set of images and their labels, pick the smallest number 
    of exemplars that contain the characters in chars. return two sets
    (i) the set of exemplars, and (ii) the remaining set usable for testing.
    '''
    with open('labels.json', encoding='utf-8') as f:
        test_pool = json.load(f)
    test_pool = {k: [i for i in v] for k,v in test_pool.items()}
    
    min_r = 5
    exemplars = {}
    while chars:
        r = min(min_r, len(chars))
        cands = list(itertools.permutations(chars, r = r))
        test_pool, found_exp, used_char = find_exemplar(test_pool, cands, r)
        chars = chars.difference(used_char)
        exemplars.update(found_exp)
        min_r -= 1
        if r == 0 or len(chars) == 0: break
    
    exemplars = {k:''.join(v) for k,v in exemplars.items()}
    test_pool = {k:''.join(v) for k,v in test_pool.items()}
    return exemplars, test_pool

def compute_scores(predictions):
    scores          = {}
    error_tracker   = defaultdict(list)
    for fp, holder_pred in predictions.items():
        pred_seqs   = holder_pred['predictions']
        golds       = holder_pred['labels']
        assert len(pred_seqs) == 1 and len(golds) == 1, \
            (len(pred_seqs), len(golds))
        pred = pred_seqs[0]
        gold = golds[0]
        score       = 0
        for p, g in  zip(pred, gold):
            if p==g: score += 1
            if p!=g: error_tracker[g].append(p)
        # norm score by length
        scores[fp]  = score/len(gold)
    return scores, error_tracker

if __name__ == '__main__':
    main()