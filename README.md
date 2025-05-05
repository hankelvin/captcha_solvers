# Solving website captchas

## Preliminaries
#### setting up the environments
##### a. for Qwen Omni
```
conda create -n qwenomni python==3.10 -y
conda activate qwenomni
python -m pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
python -m pip install qwen-omni-utils==0.0.4 -U
python -m pip install accelerate==1.6.0 torchvision==0.22.0 omegaconf==2.3.0 hydra-core==1.3.2 trl==0.17.0 
python -m pip install pandas==2.2.3 datasets==3.5.1 peft==0.15.2 bitsandbytes==0.45.5 backoff==2.2.1
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```
##### b. for Phi4 Multimodal
```
conda create -n phi4mm python==3.10 -y
conda activate phi4mm
python -m pip install torch==2.6.0 transformers==4.48.2 accelerate==1.3.0 soundfile==0.13.1 pillow==11.1.0
python -m pip install scipy==1.15.2 torchvision==0.21.0 backoff==2.2.1 peft==0.13.2 pandas==2.2.3 datasets==3.5.1 
python -m pip install bitsandbytes==0.45.5 omegaconf==2.3.0 hydra-core==1.3.2 trl==0.17.0 flash_attn==2.7.4.post1 
```

## Executive summary
Two broad approaches are proposed for solving the captcha task. The first is a zero/few-shot application of compact-sized (7B parameters and below) -- of the off-shelf and open weights variety -- models from Alibaba and Microsoft. The second is using a reinforcement learning-based method to post-train one of the models from Microsoft in an effort to improve its vision-language reasoning capabilities. 

We find that zero/few-shot performance by these models can already reach up to 19/20 (or 95% accuracy) for the larger version of the Qwen Omni model; but this required a 6-shot prompt (i.e. showing the model 6 example captchas and their solutions before asking it to solve the current one) which requires a longer inference time. 

The smaller sized (5.4B) Phi4 Multimodal model from Microsoft performed less well with 17/20 (85% accuracy); however our second approach using reinforcement learning with only 6 training instances indicates it may be possible to raise that accuracy performance by 5 percentage points (to 18/20, or 90%) with zeroshot prompting strategy. Details of the data preparation, experimental set-up and results are in the following sections.

## Data preparation
1. The labels for the images (from the outputs folder) were verified manually. The missing label for input021.jpg was added (CL69V). 
2. A set of exemplars that contain frequently mistaken characters was selected from the `input` folder; this are set aside for use as few-shot exemplars (and training instance in part two below). The remainder (20) was kept for testing. Refer to `give_data` in the `zeroshot.py` script. 

## Evaluation
3. Evaluation is done with the following measures:
    - a. average character accuracy rate (CAR), the ratio of correct characters per prediction. 
    - b. exact match (EM), whether the entire predicted 5-character sequence is the same as the label. i.e. CAR of 1.0
    - c. tracking the frequency of the mispredicted characters and the characters they are mistaken for

## Zero/few-shot benchmarking
1. Examine the off-the-shelf performance of two families of Vision-Language Models (VLM) instruction-tuned models.
    - `Qwen Omni` with multimodal input (text, audio, image) inputs and up to bi-modal outputs (e.g. speech and text)
    - `Phi4 Multimodal` with multimodal input (text, audio, image) and text output.
2. All of the results in this section came from running the code on a H100 NVL (94GB), but the 3B/5B models can run on a RTXA4000 (16GB) nvidia GPU 
3. To replicate, run the following (the rest of parameter settings are in config.yaml): 
    ```
    conda activate qwenomni
    python zeroshot.py mode='zeroshot'          size='3B' model='qwenomni'
    python zeroshot.py mode='oneshot'           size='3B' model='qwenomni'
    python zeroshot.py mode='sixshot'           size='3B' model='qwenomni'
    python zeroshot.py mode='think_zeroshot'    size='3B' model='qwenomni'
    python zeroshot.py mode='think_oneshot'     size='3B' model='qwenomni'
    python zeroshot.py mode='think_sixshot'     size='3B' model='qwenomni'
    python zeroshot.py mode='zeroshot'          size='7B' model='qwenomni'
    python zeroshot.py mode='oneshot'           size='7B' model='qwenomni'
    python zeroshot.py mode='sixshot'           size='7B' model='qwenomni'
    python zeroshot.py mode='think_zeroshot'    size='7B' model='qwenomni'
    python zeroshot.py mode='think_oneshot'     size='7B' model='qwenomni'
    python zeroshot.py mode='think_sixshot'     size='7B' model='qwenomni'

    conda activate phi4mm
    python zeroshot.py mode='zeroshot'          size='5B' model='phi4mm'
    python zeroshot.py mode='oneshot'           size='5B' model='phi4mm'
    python zeroshot.py mode='sixshot'           size='5B' model='phi4mm'
    python zeroshot.py mode='think_zeroshot'    size='5B' model='phi4mm'
    python zeroshot.py mode='think_oneshot'     size='5B' model='phi4mm'
    python zeroshot.py mode='think_sixshot'     size='5B' model='phi4mm'
    ```

4. NOTES:
    - 1-shot experiments are not optimised. Instead of randomly selecting one exemplar, better results might be obtainable with >1 selecting an exemplar that has the most commonly mistaken characters.

#### Summary stats
| model     | size| setting                       | avg time        | CAR   | EM    |
|-----------|-----|-------------------------------|-----------------|-------|-------|
|Qwen Omni  | 3B  | 0-shot                        | ~0.39 sec each  | 0.94  | 16/20 |
|Qwen Omni  | 3B  | 1-shot                        | ~0.22 sec each  | 0.94  | 15/20 |
|Qwen Omni  | 3B  | 6-shot                        | ~0.20 sec each  | 0.97  | 17/20 |
|Qwen Omni  | 3B  | 0-shot "CoT"-like thinking    | ~2.18 sec each  | 0.91  | 14/20 |
|Qwen Omni  | 3B  | 1-shot "CoT"-like thinking    | ~1.41 sec each  | 0.82  | 12/20 |
|Qwen Omni  | 3B  | 6-shot "CoT"-like thinking    | ~1.83 sec each  | 0.85  | 12/20 |
|-----------|-----|-------------------------------|-----------------|-------|-------|
|Qwen Omni  | 7B  | 0-shot                        | ~0.18 sec each  | 0.97  | 17/20 |
|Qwen Omni  | 7B  | 1-shot                        | ~0.18 sec each  | 0.98  | 18/20 |
|Qwen Omni  | 7B  | 6-shot                        | ~0.19 sec each  | 0.99  | 19/20 |
|Qwen Omni  | 7B  | 0-shot "CoT"-like thinking    | ~1.85 sec each  | 0.97  | 17/20 |
|Qwen Omni  | 7B  | 1-shot "CoT"-like thinking    | ~1.02 sec each  | 0.94  | 15/20 |
|Qwen Omni  | 7B  | 6-shot "CoT"-like thinking    | ~0.90 sec each  | 0.91  | 15/20 |
|-----------|-----|-------------------------------|-----------------|-------|-------|
|Phi4 MM    | 5B  | 0-shot                        | ~0.38 sec each  | 0.97  | 17/20 |
|Phi4 MM    | 5B  | 1-shot                        | ~0.40 sec each  | 0.75  | 13/20 |
|Phi4 MM    | 5B  | 6-shot                        | ~0.61 sec each  | 0.88  | 15/20 |
|Phi4 MM    | 5B  | 0-shot "CoT"-like thinking    | ~0.91  sec each | 0.86  | 15/20 |
|Phi4 MM    | 5B  | 1-shot "CoT"-like thinking    | ~1.01  sec each | 0.00   | 0/20 |
|Phi4 MM    | 5B  | 6-shot "CoT"-like thinking    | ~1.31 sec each  | 0.00  |  0/20 |
|-----------|-----|-------------------------------|-----------------|-------|-------|


## Reinforcement learning (GRPO) modeling
The zeroshot performance of the models above are already relatively high. EM of ~17/20. Some light post-tuning could potentially help improve the model's performance, especially if we can help it to better identify visually ambiguous characters and reason over them before providing the answer. In this part, we use reinforcement learning to try and drive further gains. 

1. This approach was inspired partly by a recent piece of work ([Reinforcement Learning for Reasoning in Large Language Models with One Training Example
](https://arxiv.org/abs/2504.20571)) that suggests a single example for reinforcement learning with verifiable rewards (RLVR) can already bring about meaningful gains on LLM math reasoning capabilities. Since we have a sample set of captchas for the task, we can transform the task into a verifiable task using these samples (or a subset) of them for training. 
2. We use Grouped Relative Policy Optimisation (GRPO), a method that was proposed by the DeepSeek authors for their math and general purpose reasoning models at the start of this year. Post-training was done on the `Phi4 Multimodal` (5.4B) parameters model.
3. For resource-efficient training, ease of deployment and speed of inference, we use the combination of LoRA fine-tuning on the 5.4B model. The prompting here is similar to the think 0-shot set up in part 1. 
4. The set of reward functions used are: (i) commonly employed rewards to encourage the model to adhere to desired output formats, (ii) rewards to encourage the model to think and reason before giving the final answer, (iii) verifiable rewards relating to the label of the training images and known constraints of the captchas (e.g. capital A-Z and 0-9). 
5. Training is done with the same set of 6 exemplars that were set aside for the few-shot experiments in part one above. We use the same test split (20 of the captchas) for testing here too, so that we can have comparable results across both approaches.
6. To replicate (training and eval), run the following (the rest of parameter settings are in config.yaml): 
    ```
    conda activate phi4mm
    python grpo.py mode='think_grpo_zeroshot' size='5B' model='phi4mm'
    ```

#### Summary stats
The results below show the EM and CAR scores across the training steps (3600 steps; gradient accumulation of 4 steps) of GRPO post-training using the 6 exemplars as RLVR training instances. The results indicate that a potential increase of 5% (i.e. increase in EM from 17/20 to 18/20) in captcha solving may be possible by using only 6 RLVR instances. 

| model     | size| setting          | CAR   | EM    |
|-----------|-----|------------------|-------|-------|
| Phi4 MM   | 5B  | GRPO step 4      | 0.94  | 18/20 |
| Phi4 MM   | 5B  | GRPO step 400    | 0.96  | 18/20 |
| Phi4 MM   | 5B  | GRPO step 800    | 0.97  | 17/20 |
| Phi4 MM   | 5B  | GRPO step 1200   | 0.87  | 16/20 |
| Phi4 MM   | 5B  | GRPO step 1600   | 0.92  | 17/20 |
| Phi4 MM   | 5B  | GRPO step 2000   | 0.86  | 15/20 |
| Phi4 MM   | 5B  | GRPO step 2400   | 0.92  | 16/20 |
| Phi4 MM   | 5B  | GRPO step 2800   | 0.87  | 15/20 |
| Phi4 MM   | 5B  | GRPO step 3200   | 0.87  | 16/20 |
| Phi4 MM   | 5B  | GRPO step 3600   | 0.96  | 16/20 |
|-----------|-----|------------------|-------|-------|

The outputs for these results can be found in the [`results/grpo_phi4mm_5B_think_grpo_zeroshot`](https://github.com/hankelvin/captcha_solvers/blob/main/results/grpo_phi4mm_5B_think_grpo_zeroshot) folder. The files in the `output` folder there are the predictions for the full set of files from the `sampleCaptchas/input` folder at the GRPO step 3600 checkpoint.