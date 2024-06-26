import re

import srsly
from datasets import Dataset
from tokenizers.processors import TemplateProcessing

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from transformers import LlamaTokenizerFast


def get_dataloaders(tokenizer,
                    train_size=1,
                    data_path="data/train.jsonl",
                    add_domain_info=False,
                    format=None):
    """
    Prepares data loaders for model training by tokenizing and formatting the training data.

    This function tokenizes the data present in the specified file and prepares it as input for model training. It allows for memory-efficient data loading if required and supports adjusting the size of the training dataset.

    Parameters:
    - tokenizer: The tokenizer to be used for processing the data. This should be compatible with the model to be trained.
    - train_size (float, optional): A multiplier for the size of the training data. Default is 1, which uses the entire dataset as training data. Values less than 1 will reduce the train dataset and increase the validation dataset proportionally.
    - data_path (str, optional): Path to the training data file. The default is "data/train.jsonl", assuming a JSONL format.
    - mem_efficient (bool, optional): If set to True, the function will load data in a memory-efficient manner. This is useful for large datasets. Default is False.
    - format (str, optional): Format of the sequence used for training.
    Returns:
    Train and Validation DataLoader
    """

    input_ids = []
    attention_masks = []
    logit_masks = []
    output_labels = []
    gptq_samples = []

    raw_json_data = srsly.read_jsonl(data_path)
    for raw_elem in tqdm(raw_json_data):
        joined_text_with_split_token = format.format(bos_token=tokenizer.bos_token,
                                     sep_token=tokenizer.sep_token,
                                     eos_token=tokenizer.eos_token,
                                     **raw_elem)

        input_txt, output_text = joined_text_with_split_token.split("|<START_LOSS>|")
        input_ids_complex, _ = tokenizer(input_txt, add_special_tokens=False).values()
        target_ids, _ = tokenizer(output_text, add_special_tokens=False).values()

        token_ids, attention_mask = tokenizer(input_txt + output_text, add_special_tokens=False).values()
        if len(token_ids) > 1500:
            continue

        complex_count = len(input_ids_complex)

        # check format of token_ids
        assert token_ids[:complex_count] == input_ids_complex
        assert token_ids[complex_count:complex_count + len(target_ids)] == target_ids
        assert token_ids[:complex_count][-1] == tokenizer.sep_token_id
        assert token_ids[complex_count + len(target_ids)-1] == tokenizer.eos_token_id

        logit_mask = np.array(attention_mask)
        logit_mask[:complex_count] = 0

        labels = np.array(token_ids)
        labels[:complex_count] = -100
        labels[~np.array(attention_mask, dtype=bool)] = -100

        assert [l for l in labels.tolist() if l >= 0] == target_ids

        input_ids.append(token_ids)
        attention_masks.append(attention_mask)
        logit_masks.append(logit_mask)
        output_labels.append(labels)
        gptq_samples.append((input_txt + output_text).replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, ""))

    data = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_masks, "labels": output_labels})
    train_data, validation_data = torch.utils.data.random_split(data, [int(train_size * len(data)), len(data) - int(train_size * len(data))], generator=torch.Generator().manual_seed(42))

    return train_data, validation_data, gptq_samples[:1000]