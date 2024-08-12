---
library_name: transformers
license: mit
base_model: microsoft/phi-1_5
datasets:
- teknium/OpenHermes-2.5
- HuggingFaceH4/ultrafeedback_binarized
- argilla/distilabel-intel-orca-dpo-pairs
- jondurbin/py-dpo-v0.1
- argilla/distilabel-math-preference-dpo
pipeline_tag: text-generation
---

# Phi-1.5
The language model [phi-1.5](https://huggingface.co/microsoft/phi-1_5) is a Transformer with **1.3 billion** parameters. It was trained using the same data sources as [phi-1](https://huggingface.co/microsoft/phi-1), augmented with a new data source that consists of various NLP synthetic texts. When assessed against benchmarks testing common sense, language understanding, and logical reasoning, phi-1.5 demonstrates a nearly state-of-the-art performance among models with less than 10 billion parameters.

# Phi-1_5-Instruct-v0.1
The model has underwent a post-training process that incorporates both **supervised fine-tuning** and **direct preference optimization** for instruction following. I used the [trl](https://huggingface.co/docs/trl/en/index) library and a single **A100 40GB** GPU during both the SFT and DPO steps.

- Supervised Fine-Tuning
  - SFT Model: [phi-1_5-sft](https://huggingface.co/rasyosef/phi-1_5-sft)
  - Used 128,000 instruction, response pairs from the [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) dataset

- Direct Preference Optimization (DPO)
  - LoRA Adapter: [phi-1_5-dpo](https://huggingface.co/rasyosef/phi-1_5-dpo)
  - Used a combination of the following preference datasets
    - [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
    - [argilla/distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)
    - [argilla/distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo)
    - [jondurbin/py-dpo-v0.1](https://huggingface.co/datasets/jondurbin/py-dpo-v0.1)

- Final Merged Model
    - https://huggingface.co/rasyosef/Phi-1_5-Instruct-v0.1

## How to use
### Chat Format

Given the nature of the training data, the Phi-1.5 Instruct model is best suited for prompts using the chat format as follows. 
You can provide the prompt as a question with a generic template as follow:
```markdown
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Question?<|im_end|>
<|im_start|>assistant
```

For example:
```markdown
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
How to explain Internet for a medieval knight?<|im_end|>
<|im_start|>assistant
```
where the model generates the text after `<|im_start|>assistant` .

### Sample inference code

This code snippets show how to get quickly started with running the model on a GPU:

```python
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 

model_id = "rasyosef/Phi-1_5-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained( 
    model_id,  
    device_map="cuda",  
    torch_dtype="auto" 
) 

tokenizer = AutoTokenizer.from_pretrained(model_id) 

messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 256, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])  
```

Note: If you want to use flash attention, call _AutoModelForCausalLM.from_pretrained()_ with _attn_implementation="flash_attention_2"_


## Benchmarks

This model outperforms HuggingFace's SmolLM-1.7B-Instruct and the TinyLlama-1.1B-Chat-v1.0 models on **all 5** of the following benchmarks. These benchmarks were run using EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

- **IFEval (Instruction Following Evaluation)**: IFEval is a fairly interesting dataset that tests the capability of models to clearly follow explicit instructions, such as “include keyword x” or “use format y”. The models are tested on their ability to strictly follow formatting instructions rather than the actual contents generated, allowing strict and rigorous metrics to be used.
- **GSM8k (5-shot)**: diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems.
- **MMLU (5-shot)** - a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
- **TruthfulQA** - a test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.
- **Winogrande (5-shot)** - an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning.

|Model|Size (# params)|IFEval|GSM8K|MMLU|TruthfulQA|Winogrande|
|:----|:--------------|:-----|:----|:---|:---------|:---------|
|[Phi-1_5-Instruct-v0.1](https://huggingface.co/rasyosef/Phi-1_5-Instruct-v0.1)|1.4B|**26.71**|**41.78**|39.72|**47.9**|70.4|
|[SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct)|1.7B|24.21|3.45|23.57|47.38|63.61|
|[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)|1.1B|21.23|0|24.03|39.14|61.01|
|[phi-1_5](https://huggingface.co/microsoft/phi-1_5)|1.4B|20.51|31.73|**42.48**|40.86|**71.74**|

## Demo

You can use this hugging face space to interact with the chat model
https://huggingface.co/spaces/rasyosef/phi-1_5-chat