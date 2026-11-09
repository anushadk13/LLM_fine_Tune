# LoRA fine-tuning for content safety

This repository contains Jupyter/Google Colab workflows for adapting small causal language models to content-safety and moderation tasks with [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora).

The notebooks use the [NVIDIA Aegis AI Content Safety Dataset 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0), format its records as instruction-following examples, fine-tune the selected model, evaluate held-out examples, and optionally publish the result to the Hugging Face Hub.

This is an experimental training project, not a production moderation service. Treat model outputs as advisory and add policy logic, monitoring, human review, and an appeal process before using them in a real system.

## What is included

| Notebook | Purpose | Base model(s) | Default sample count |
| --- | --- | --- | ---: |
| [`multi_model_finetuning.ipynb`](multi_model_finetuning.ipynb) | Comparative fine-tuning and evaluation | Qwen3-0.6B, Llama 3.2-1B, ERNIE 4.5-0.3B | 2,000 |
| [`deepseek_r1_finetuning_colab.ipynb`](deepseek_r1_finetuning_colab.ipynb) | Single-model Colab workflow | DeepSeek-R1-Distill-Qwen-1.5B | 1,000 |

Supporting files:

- [`setup_hf_token.py`](setup_hf_token.py) contains a legacy Hugging Face login helper. Review the security note below before running it.
- [`LICENSE`](LICENSE) contains the project license and third-party model/dataset attributions.

## Models and dataset

The multi-model notebook is configured for:

- [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [`baidu/ERNIE-4.5-0.3B-Base-PT`](https://huggingface.co/baidu/ERNIE-4.5-0.3B-Base-PT)

The Colab notebook is configured for [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).

Both workflows load the `train` split of the Aegis dataset, shuffle with seed `42`, select a configurable subset, and split it into training, validation, and test sets. The notebooks adapt their formatter to common dataset field layouts and produce a single `text` field for supervised fine-tuning.

Read the dataset card and each base model's license before downloading, modifying, or redistributing them. Access to Meta's Llama model may require accepting its Hugging Face terms first.

## Requirements

- Python 3.9 or newer
- A CUDA-capable GPU; a Colab T4 is the target environment for the DeepSeek workflow
- Enough disk space for model downloads, checkpoints, and cached datasets
- A Hugging Face account and token if you need gated model access or Hub uploads

The notebooks install or use the following packages:

```text
torch
transformers
datasets
accelerate
peft
trl
bitsandbytes
huggingface_hub
sentencepiece
protobuf
numpy
pandas
scikit-learn
matplotlib
seaborn
```

The exact package versions are defined in the notebook setup cells. If the `trl` or `transformers` APIs have changed, use the compatibility fallbacks already provided in the notebooks or pin a known-compatible environment.

## Quick start

1. Clone the repository and open one of the notebooks in Jupyter or Google Colab.
2. Select a GPU runtime. In Colab, use **Runtime → Change runtime type → T4 GPU** when available.
3. Run the setup and import cells. Adjust the PyTorch index URL if your CUDA version differs from the one in the notebook.
4. Authenticate without putting a token in source code:

   ```bash
   pip install -U huggingface_hub
   hf auth login
   ```

   Or authenticate interactively in Python:

   ```python
   from huggingface_hub import login

   login()  # paste a token when prompted
   ```

5. Review the configuration cell: model identifier, sample count, output directory, Hub repository name, and 4-bit loading.
6. Run the notebook cells from top to bottom. The workflow loads data, creates splits, adds LoRA adapters, trains, evaluates, saves the checkpoint, and optionally uploads it.
7. Inspect metrics and sample generations before considering a model for downstream use.

## Default training configuration

The two notebooks use closely related LoRA settings:

```python
{
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
```

The multi-model workflow defaults to three epochs, learning rate `2e-4`, maximum sequence length `512`, and an effective batch size of `16` (`4 × 4`). The Colab workflow uses a smaller per-device batch size and more gradient accumulation to fit a T4 GPU. Both workflows attempt 4-bit NF4 quantization with bitsandbytes and fall back to a higher-memory configuration when quantization is unavailable.

These are starting points, not validated production hyperparameters. Re-run with multiple seeds and a representative evaluation set before drawing model comparisons.

## Hugging Face Hub uploads

The multi-model notebook is configured to publish repositories under the account used during login, with names matching:

```text
qwen3-0.6b-aegis-safety-lora
llama-3.2-1b-aegis-safety-lora
ernie-4.5-0.3b-aegis-safety-lora
```

The DeepSeek notebook creates a repository named:

```text
deepseek-r1-distill-qwen-1.5b-aegis-safety-lora
```

Uploading is optional; save locally if you do not want to publish the result. Check the generated model card and confirm whether the repository should be public before calling `push_to_hub`.

## Loading a saved model

The notebooks save a tokenizer and LoRA checkpoint in their configured output directory. For an adapter checkpoint, load the base model first and then attach the adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-0.6B-Base"
adapter_dir = "./outputs/qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_dir)

prompt = "### Instruction:\nAnalyze this content for safety: 'Hello, how are you?'\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use the exact base model and prompt format associated with the checkpoint. If the notebook has merged the adapter into a standalone model for Hub upload, load that repository directly with `AutoModelForCausalLM.from_pretrained` instead.

## Evaluation and interpretation

The notebooks report validation loss/perplexity, training timing, and generated examples. These measurements are useful for comparing runs, but perplexity alone does not measure moderation quality. Add task-specific evaluation for:

- safe versus unsafe classification accuracy
- precision, recall, and F1 by safety category
- false-positive rates on benign content
- false-negative rates on harmful content
- robustness to paraphrases, adversarial wording, and distribution shift
- language, cultural, and demographic coverage

Do not interpret a generated explanation as a reliable safety rationale. Parse outputs against an explicit schema and reject malformed or uncertain responses in downstream code.

## Security note

The committed `setup_hf_token.py` file contains a hard-coded Hugging Face token. Do not run or reuse that credential. Revoke and replace it in the Hugging Face settings, then use `hf auth login`, an environment variable, or a secret manager instead. Never commit access tokens, API keys, checkpoints containing secrets, or private dataset contents.

## Limitations

- The notebooks use relatively small samples and compact causal language models; results are not representative of a complete safety system.
- Dataset composition, labels, and language coverage may not match your application.
- Quantization and GPU memory behavior depend on the installed CUDA, PyTorch, and bitsandbytes versions.
- The generated text format is instruction-tuned output, not a guaranteed classifier API.
- Model, dataset, and generated-output licenses may differ from this repository's MIT license.
- Content moderation decisions can affect people. Keep a human-review path for high-impact or ambiguous cases.

## Reproducibility checklist

For each run, record:

- base model revision and dataset revision
- package and CUDA versions
- sample count, split seed, and prompt formatter
- LoRA and optimizer settings
- GPU type and whether 4-bit quantization was enabled
- evaluation set, metrics, and generated samples
- whether the uploaded artifact is an adapter or a merged model

## License and attribution

The repository's own code and documentation are licensed under the MIT License. The base models and NVIDIA dataset have separate terms; see [`LICENSE`](LICENSE) and the linked model and dataset cards for the conditions that apply to your use.

## Contributing

Useful contributions include reproducible environment files, stronger task-specific evaluation, safer output parsing, support for additional model architectures, and fixes that keep the notebooks compatible with current Hugging Face libraries. Please avoid submitting credentials, private data, or large generated checkpoints.
