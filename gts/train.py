from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, GPTQConfig, \
    DataCollatorWithPadding, AutoConfig

from gts.data_loader import get_dataloaders

import logging
from transformers import logging as hf_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
hf_logging.set_verbosity_info()

torch.set_default_dtype(torch.bfloat16)
torch.cuda.empty_cache()


def main(args):
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else model_path
    save_path = args.save_path if args.save_path else model_path.split("/")[-1]

    config = AutoConfig.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(config)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Starting Fine-Tuning of Model {model_path}")

    # Check and possibly set special tokens
    if tokenizer.sep_token_id is None:
        tokenizer.add_special_tokens({'sep_token': '|<SEP>|'})
        print("Set SEP-Token to |<SEP>|")

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '|<PAD>|'})
        print("Set PAD-Token to |<PAD>|")

    print("Preprocessing...")
    train_dataset, validation_dataset, gptq_samples = get_dataloaders(
        tokenizer=tokenizer,
        train_size=args.train_size,
        data_path=args.data_path,
        format=args.format,
    )
    print("done")

    out_path = Path(args.output_dir) / save_path
    if not out_path.exists():
        print(f"{out_path} does not exist, creating directory.")
        out_path.mkdir(parents=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        overwrite_output_dir=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        torch_compile=True,
        eval_strategy="no",
        num_train_epochs=args.num_epochs,
        optim="adamw_bnb_8bit",
        save_strategy="epoch",
        save_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        bf16=True,
        warmup_steps=100,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=args.weight_decay,
        run_name="seminar_easy_language",
        logging_steps=1,
        logging_dir='./runs',
        accelerator_config={"split_batches": False},
        report_to="all"
    )


    with torch.no_grad():
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.sep_token_id = tokenizer.sep_token_id

    trainer = Trainer(
        model=model,
        args=training_args,
        # optimizers=(optimizer, None) if not args.deepspeed else (None, None),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    try:
        print("Starting training...")
        trainer.train()
    except RuntimeError as e:
        print(f"Caught RuntimeError: {e}")
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device="cuda"))
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    print(f"Model and tokenizer saved at {out_path}")

    print(f"Model and tokenizer saved at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
    parser.add_argument('--model_path', type=str, required=True, help='Model path on Huggingface hub')
    parser.add_argument('--tokenizer_path', type=str, help='Tokenizer path on Huggingface hub')
    parser.add_argument('--save_path', type=str, help='Save path for model and tokenizer')
    parser.add_argument('--train_size', type=float, default=1, help='Training dataset size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--include_domain_info', action='store_true', help='Include domain information in the data')
    parser.add_argument('--data_path', type=str, default="data/beta_train.jsonl", help='Path to training data')
    parser.add_argument('--output_dir', type=str, default="models", help='Output directory for saving results')
    parser.add_argument('--deepspeed', action='store_true', help='Enable deepspeed optimization')
    parser.add_argument("--format", type=str, default="{bos_token}{context}{sep_token}{input}{sep_token}|<START_LOSS>|{target}{eos_token}", help="Sequence format")

    args = parser.parse_args()
    main(args)
