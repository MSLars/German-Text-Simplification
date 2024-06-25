# German-Text-Simplification
Repository for the Paper *German Text Simplification: Finetuning Large Language Models with Semi-Synthetic Data* accepted at LTEDI Workshop at EACL 2024.

## Pretrained Models

All models from our Paper are available via HuggingFace:

- Smaller GPT2 Model: https://huggingface.co/MSLars/erlesen-gpt (model name: `MSLars/erlesen-gpt`)
- Larger GPT2-xl Model: https://huggingface.co/MSLars/erlesen-gpt-xl (model name: `MSLars/erlesen-gpt-xl`)
- Leo 7b Model: https://huggingface.co/MSLars/erlesen-leo-7b (model name: `MSLars/erlesen-leo-7b`)
- Leo 13b Model: https://huggingface.co/MSLars/erlesen-leo-13b (model name: `MSLars/erlesen-leo-13b`)

## Setup

To follow the instructions, we provide a conda environment. We assume that you have a modern cuda-ready nvidia GPU.
If not, you need to adjust the env file.

Create environment. You need conda installed. The provided `env.yml` is necessary on model training.
If you want to use our models it is easier to access them via HuggingFace.

```bash
conda env create -f env.yml

conda develop .
```

Install FlashAttention afterwards. Following instructions are from:

- Make sure that `packaging` is installed (`pip install packaging`)
- Make sure that `ninja` is installed and that it works correctly (e.g. `ninja --version` then `echo $?` 
should return exit code 0). 
If not (sometimes `ninja --version` then echo `$?` returns a nonzero exit code), 
uninstall then reinstall `ninja` (`pip uninstall -y ninja && pip install ninja`). 
Without ninja, compiling can take a very long time (2h) 
since it does not use multiple CPU cores. 
With ninja compiling takes 3-5 minutes on a 64-core machine.
- Then:

```bash
pip install flash-attn --no-build-isolation
```


## Training Models

To train custom models you need a `.jsonl` (JSON Lines)[https://jsonlines.org/] trainng file with two mandatory fields:

```json
{"complex":"Non-simplified text", "easy": "simplified version of the text", "ll":"(optional metadata, necessary if you want to add domain information in training)", "domain":"(optional metadata, necessary if you want to add domain information in training)"}
```

In the Project root dir execute:

```bash
PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;WANDB_MODE=offline python -m gts.train --model_path benjamin/gpt2-wechsel-german --data_path data/beta_train.jsonl
```

## Making Predictions

The easiest way to make predictions is to load our models via Hugging Face.

We use the `MSLars/erlesen-leo-7b`. If your GPU cannot fit the model, either decrease the batch size or change
`DEVICE="cpu"`. Prediction on CPU may take several minutes.

Alternatively, you can switch to smaller models.

```python
import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':

    model_name = "MSLars/erlesen-leo-7b"

    text_to_simplify = "Bei den diesjährigen Europameisterschaften haben " \
            "deutsche Athleten zahlreiche Medaillen errungen. " \
            "Die Wettkämpfe fanden an verschiedenen Orten statt, "\
            "einige in Berlin und andere in Glasgow, Schottland."


    # Create configuration for Generation
    generation_config = GenerationConfig(
        no_ngram_repeat_size=5,
        max_length=1024,
        num_beams=2,
        early_stopping=True
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the formatted prompt
    formated_model_input = f"{tokenizer.bos_token}{text_to_simplify}{tokenizer.sep_token}"

    # now prepare input for generation
    inputs = tokenizer(text_to_simplify, return_tensors="pt").to(model.device)
    # start generation
    model_output = model.generate(**inputs,
                                  generation_config=generation_config,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id,
                                  bos_token_id=tokenizer.bos_token_id,)

    # decode token indices to string
    decoded_text = tokenizer.decode(model_output[0, inputs["input_ids"].shape[1]:-1])

    print(decoded_text)
```

Output:
```text
Die deutschen Sportlerinnen und Sportler haben bei den Europa-Meisterschaften viele Medaillen gewonnen.
Die Wett-Kämpfe waren in Berlin und in Glasgow.
Glasgow ist eine Stadt in Schottland.
Die Europa-Meisterschaften waren in verschiedenen Sport-Arten.
```

## Known Issues

As described in our Paper these models have several drawbacks, which we will address in future iterations.

### Non-stopping and random outputs from smaller modells

The GPT2 and GPT2-xl seem to have problems to predict the end of simplification correctly.
In some cases these models repeat phrases or sentence or start predicting random tokens.
We implemented `MaxTokenOccurenceInWindowCriteria` and `LineRepetitionCriteria` to deal with this.

Here is the simplfication of the GPT2-xl model of the previous example:

```text
Die deutschen Sportler haben viele Medaillen gewonnen.
Sie haben bei den Europameisterschaften viele Medaillen gewonnen.
Sie haben bei den Europameisterschaften viele Medaillen gewonnen.
Sie haben bei den Europameisterschaften viele Medaillen gewonnen.
Sie haben bei den Europameisterschaften viele Medaillen gewonnen.
Sie haben bei den Europameisterschaften viele Medaillen gewonnen.
Sie haben bei den Europameisterschaften viele Medaillen gewonnen.
...
```

### Different simplification styles

The larger models leo-7b and leo-13b seem to apply different simplification styles.
We assume that this is because the web sources we used for training contain different styles of simplifications.
