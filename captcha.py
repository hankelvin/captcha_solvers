import torch, itertools, re, json, os
from PIL import Image
SEED = 117438
torch.manual_seed(SEED)

class Captcha(object):
    def __init__(self, args):
        from omegaconf import OmegaConf
        if   torch.mps.is_available():  device = 'mps'
        elif torch.cuda.is_available(): device = 'cuda'
        else:                           device = 'cpu'
    
        # 0. set up config, load exemplars
        cfg         = OmegaConf.load('config.yaml')
        cfg.dirpath = os.getcwd()
        cfg.model   = args.model
        cfg.size    = args.size
        cfg.mode    = 'zeroshot'
        cfg.load_in_4bit = True if device in ['cuda'] else False
        # NOTE; bnb quantisation not yet ready for Apple silicon 
        # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/252
        from zeroshot import give_data
        exemplars, __ = give_data(chars = set(i for i in 'Q0O1I5S2Z'))

        self.cfg                = cfg
        self.args               = args
        self.use_audio_in_video = False
        self.exemplars          = exemplars

        # 1. load model and media processor
        self.model_path         = cfg.models[cfg.model][cfg.size]    
        attn_implementation     = 'flash_attention_2' if device in ['cuda'] else 'eager'
        torch_dtype             = torch.bfloat16

        assert (cfg.load_in_4bit and cfg.load_in_8bit) is False # both False or one False
        if cfg.load_in_4bit or cfg.load_in_8bit:
            from transformers import BitsAndBytesConfig        
            bnb_args = {'load_in_4bit': cfg.load_in_4bit, 'load_in_8bit': cfg.load_in_8bit}
            if cfg.load_in_4bit:
                bnb_args['bnb_4bit_quant_type']         = "nf4"
                bnb_args['bnb_4bit_use_double_quant']   = True
                bnb_args['bnb_4bit_compute_dtype']      = torch_dtype
            quantization_config = BitsAndBytesConfig(**bnb_args)
        else: quantization_config = None
    
        self.cloud_inference = self.args.cloud_inference
        if self.cloud_inference:
            from huggingface_hub import InferenceClient
            assert cfg.model in ['qwen25vl']
            self.client     = InferenceClient(provider = 'hf-inference', api_key = args.hf_token)
            cfg.size        = '32B' # HF inference has >= 32B only
            self.model_path = cfg.models[cfg.model][cfg.size]    

        self.model = None
        if cfg.model in ['qwenomni', 'qwen25vl']:
            if  cfg.model == 'qwenomni': 
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
                from qwen_omni_utils import process_mm_info
                if not self.cloud_inference:
                    self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.model_path,
                                                                                torch_dtype         = torch_dtype,
                                                                                device_map          = device,
                                                                                attn_implementation = attn_implementation,
                                                                                quantization_config = quantization_config)
                    self.model.disable_talker()
                self.processor          = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
                self.process_mm_info    = process_mm_info
            elif cfg.model == 'qwen25vl': 
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                from qwen_vl_utils import process_vision_info
                if not self.cloud_inference:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, 
                                                                                torch_dtype         = torch_dtype,
                                                                                device_map          = device,
                                                                                attn_implementation = attn_implementation,
                                                                                quantization_config = quantization_config)
                self.processor = AutoProcessor.from_pretrained(self.model_path)
                self.process_mm_info = process_vision_info
            else: raise NotImplementedError
            
        elif cfg.model in ['phi4mm']: 
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            if not self.cloud_inference:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    device_map = device, 
                    torch_dtype = torch.bfloat16, 
                    trust_remote_code = True,
                    quantization_config = quantization_config,
                    _attn_implementation = attn_implementation).cuda()

                # Load generation config
                self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code = True)
            self.process_mm_info = None
        
        else: raise NotImplementedError
        print('USING CLOUD INFERENCE:', self.cloud_inference)
        print('MODEL FOR INFERENCE:',   self.model_path)
        print('QUANTISED STATUS:',      self.cfg.load_in_4bit, self.cfg.load_in_8bit)

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        self.inference_one_file(im_path, label = None, save_path = save_path, full_path = True)

    def make_one_conversation_qwenomni(self, im_path, full_path = False):
        """
        Algo for producing conversation for Qwen Omni model
        args:
            im_path: .jpg image path to load and to infer
            full_path: whether the absolute path has been specified for im_path
        """
        dp = '' if self.cfg.dirpath is None else f'{self.cfg.dirpath}/'
        conversation = [{'role': 'system',
                        'content': [{'type': 'text', 'text': self.cfg.prompt_library.system}],},]
        
        mode_key    = 'zeroshot' if 'zeroshot' in self.cfg.mode else 'fewshot'
        think_key   = 'think_grpo' if 'grpo' in self.cfg.mode else 'think'

        # add few-shot exemplars (if specified)
        if self.exemplars and self.cfg.mode != 'zeroshot': 
            for oneshot_fp, oneshot_label in self.exemplars.items():
                if  self.cfg.mode.startswith('think'):
                    prompt  = self.cfg.prompt_library[think_key]
                    label = self.cfg.prompt_library['label_think_fewshot'][oneshot_fp]
                else: 
                    prompt  = self.cfg.prompt_library[mode_key]
                    label   = oneshot_label
                image_path = f'{dp}{self.cfg.inputs_path}/{oneshot_fp}'
                if self.cloud_inference:
                    image_line = {"type": "image_url", "image_url": {"url": image_path}}
                else: 
                    image_line = {'type': 'image', 'image': image_path}
                conversation.append({'role': 'user',
                                    'content': [image_line, 
                                                {'type': 'text', 'text': prompt },]})
                conversation.append({'role': 'assistant',
                                    'content': [{'type': 'text', 'text':  label},],})

        # add task to be solved
        if  self.cfg.mode.startswith('think'):
            prompt = self.cfg.prompt_library[think_key]
        else: prompt = self.cfg.prompt_library[mode_key]
        if full_path:   image_path = im_path
        else:           image_path = f'{dp}{self.cfg.inputs_path}/{im_path}'
        if self.cloud_inference: 
            image_line = {"type": "image_url", "image_url": {"url": image_path}}
        else: 
            image_line = {'type': 'image', 'image': image_path}
        conversation.append({'role': 'user',
                            'content': [image_line,
                                        {'type': 'text', 'text':  prompt},],})
        return conversation
    
    def make_one_conversation_phi4mm(self, im_path, full_path = False):
        """
        Algo for producing conversation input dict for Phi4 Multimodal model
        args:
            im_path: .jpg image path to load and to infer
            full_path: whether the absolute path has been specified for im_path
        """
        if self.cloud_inference: raise NotImplementedError
        dp = '' if self.cfg.dirpath is None else f'{self.cfg.dirpath}/'
        user_prompt         = '<|user|>'
        assistant_prompt    = '<|assistant|>'
        prompt_suffix       = '<|end|>'
        prompt_text         = ''
        image_files         = []
        
        mode_key    = 'zeroshot' if 'zeroshot' in self.cfg.mode else 'fewshot'
        think_key   = 'think_grpo' if 'grpo' in self.cfg.mode else 'think'
        # add few-shot exemplars (if specified)
        if self.exemplars and self.cfg.mode != 'zeroshot':
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

        # add task to be solved
        if  self.cfg.mode.startswith('think'):
            prompt = self.cfg.prompt_library[think_key]
        else: prompt = self.cfg.prompt_library[mode_key]
        if full_path:   image_path = im_path
        else:           image_path = f'{dp}{self.cfg.inputs_path}/{im_path}'
        image_files.append(image_path)
        prompt = f'{user_prompt}<|image_{len(image_files)}|>{prompt}{prompt_suffix}{assistant_prompt}'
        prompt_text += prompt
        return prompt_text, image_files
    
    def inference_one_file(self, im_path, label = None, save_path = None, full_path = False):
        """
        Algo for producing conversation input dict for Qwen Omni model
        args:
            im_path: .jpg image path to load and to infer
            label: the ground-truth label for the image in im_path
            save_path: if specified, an absolute filepath for saving the predicted sequence to (as a .txt file)
            full_path: whether the absolute path has been specified for im_path
        """
        if self.cloud_inference:
            conversation = self.make_one_conversation_qwenomni(im_path, full_path)
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt = True, tokenize = False)
            client_completion = self.client.chat.completions.create(model = self.model_path, messages = conversation)
            prompt_text = [prompt_text] # qwen25vl processor's apply_chat_template returns string for 1 conversation input

        elif not self.cloud_inference and self.cfg.model in ['qwenomni', 'qwen25vl']:
            conversation = self.make_one_conversation_qwenomni(im_path, full_path)
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt = True, tokenize = False)
            if  self.cfg.model == 'qwenomni':
                audios, images, videos  = self.process_mm_info(conversation, use_audio_in_video = self.use_audio_in_video)
                inputs                  = self.processor(text = prompt_text, audio = audios, images = images, videos = videos, 
                                            return_tensors = "pt", padding = True, use_audio_in_video = self.use_audio_in_video)
                inputs                  = inputs.to(self.model.device).to(self.model.dtype)
                with torch.no_grad():
                    prompt_completion_ids   = self.model.generate(**inputs, **self.cfg.gen_args,
                                                                use_audio_in_video = self.use_audio_in_video)
            elif self.cfg.model == 'qwen25vl':
                images, videos  = self.process_mm_info(conversation)
                inputs          = self.processor(text = prompt_text, images = images, videos = videos, 
                                            return_tensors = "pt", padding = True)
                inputs          = inputs.to(self.model.device).to(self.model.dtype)
                with torch.no_grad():
                    prompt_completion_ids   = self.model.generate(**inputs, **self.cfg.gen_args)
                prompt_text = [prompt_text] # qwen25vl processor's apply_chat_template returns string for 1 conversation input
        
        elif not self.cloud_inference and self.cfg.model in ['phi4mm']:
            prompt_text, image_files = self.make_one_conversation_phi4mm(im_path, full_path)
            images = [Image.open(fp) for fp in image_files]
            inputs = self.processor(text = prompt_text, images = images, return_tensors='pt')
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            with torch.no_grad():
                prompt_completion_ids   = self.model.generate(**inputs, 
                                                              generation_config = self.model.generation_config)
            # HACK for extract_results to align with qwenomni treatment
            prompt_text = [prompt_text]
        
        else: raise NotImplementedError
        
        if self.cloud_inference:
            assert type(prompt_text) is list
            prefills                = prompt_text
            predictions             = [client_completion.choices[0].message.content]
            assert len(prefills) == len(predictions)
            prompt_completion_texts = [f"{prefill} {pred}" for prefill, pred in zip(prefills, predictions)]
        else:
            prompt_completion_texts = self.processor.batch_decode(prompt_completion_ids, 
                                        skip_special_tokens = True, clean_up_tokenization_spaces = False)
            assert len(prompt_completion_texts) == 1 and len(prompt_text) == 1, prompt_completion_texts

            prefills, completions, predictions = extract_results(self.cfg['mode'], prompt_completion_texts, 
                                                    prompt_text, self.model_path, self.processor)
        
        if save_path is not None: 
            with open(save_path, encoding = 'utf-8', mode = 'w+') as f:
                assert type(predictions) is list and len(predictions) == 1
                f.write(predictions[0])
        
        return {'prompts': prefills, 'generated': prompt_completion_texts, 
                'labels': [label], 'predictions': predictions}
    
def extract_results(mode, prompt_completion_texts, prompt_text, model_path, processor):
    """
    Algo for extracting the prediction from the LLM generated outputs
    args:
        mode: the prompting mode to use (e.g. zeroshot, oneshot, sixshot etc)
        prompt_completion_texts: the string containing both the prompt, and the completion generated by the LLM
        prompt_text: the string containing only the prompt given to the LLM
        model_path: the huggingface model path
        processor: the text processing object (holding the LLM tokenizer)
    """
    # get prefill size to trim from generated outputs
    c1 = model_path.startswith('Qwen/Qwen2.5-Omni') or model_path.startswith('Qwen/Qwen2.5-VL')
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
    """
    Algo for identifying image instances whose labels match a character search criteria
    args:
        test_pool: a dictionary holding the instances that are remaining over time (after successively 
        removing the matching candidates from cands)
        cands: the set of potential candidates meeting criteria
        r: the number of unique characters in the sequence being searched for
    """
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
    Algo for loading datasets to identify the set of exemplars and a usable test set
    -- from the set of images and their labels, pick the smallest number 
    of exemplars that contain the characters in chars. return two sets
    (i) the set of exemplars, and (ii) the remaining set (usable for testing)
    args:
        chars: the sequence of characters (e.g. frequently mispredicted) to build the exemplars
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



if __name__ == '__main__':
    # NOTE: to run, create environment for Qwen Omni (instructions in README.md)
    import glob, tqdm, argparse
    parser = argparse.ArgumentParser(description='Arugments for running Captcha instance to predict captcha contents.')
    parser.add_argument('--model', help = 'short name of the model to load for inference (see config.yaml).', 
                        default = 'qwen25vl', type = str)
    parser.add_argument('--size', help = 'size of the model to load for inference.', 
                        default = '3B', type = str)
    parser.add_argument('--cloud_inference', help = 'whether to use cloud inference to run model', 
                        default = False, type = bool)
    parser.add_argument('--cloud_bucket', help = 'the cloud bucket storage for a known file useable for inference', 
                        default = 'https://storage.googleapis.com/captcha_solvers/', type = str)
    parser.add_argument('--hf_token', help = 'HuggingFace token for cloud inference. see https://huggingface.co/docs/hub/security-tokens', 
                        default = None, type = str)
    args = parser.parse_args()

    im_paths    = glob.glob('sampleCaptchas/input/input*.jpg')
    save_dp     = 'outputs'
    if not os.path.exists(save_dp): os.makedirs(save_dp)

    captcha     = Captcha(args = args)
    for im_path in  tqdm.tqdm(im_paths):
        # replace with hosted version of file (cloud API requires URL for image input)
        if args.cloud_inference: im_path = os.path.join(args.cloud_bucket, os.path.basename(im_path))
        print('Working on this im_path:', im_path)
        save_fp = os.path.basename(im_path).replace('.jpg', '.txt').replace('input', 'output')
        save_path = os.path.join(save_dp, save_fp)
        captcha(im_path, save_path)