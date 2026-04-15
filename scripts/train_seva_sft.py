"""SEVA v2 SFT Training Script.

Trains Qwen2.5-3B/7B-Instruct on structured reasoning chains.
Supports full fine-tuning (3B) and LoRA (7B+).

Usage:
    # 3B full fine-tuning
    CUDA_VISIBLE_DEVICES=0,1 python scripts/train_seva_sft.py

    # 7B with LoRA
    CUDA_VISIBLE_DEVICES=0 python scripts/train_seva_sft.py \
        --base-model /path/to/7B --lora --lora-rank 64

    # 7B LoRA, merge weights after training
    CUDA_VISIBLE_DEVICES=0 python scripts/train_seva_sft.py \
        --base-model /path/to/7B --lora --merge-lora
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str,
                        default="/home/yinian/models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--train-file", type=str,
                        default="/home/yinian/verifiable_agent/data/attribution/seva_sft_train.jsonl")
    parser.add_argument("--output-dir", type=str,
                        default="/home/yinian/verifiable_agent/checkpoints/seva_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1280)
    # LoRA arguments
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA instead of full fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--merge-lora", action="store_true",
                        help="Merge LoRA weights into base model after training")
    args = parser.parse_args()

    print(f"Base model: {args.base_model}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"LoRA: {args.lora} (rank={args.lora_rank})" if args.lora else "Full fine-tuning")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA if requested
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load data
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    print(f"Loaded {len(dataset)} samples")

    def tokenize_fn(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()

        # Mask system + user tokens (only train on assistant response)
        for i, messages in enumerate(examples["messages"]):
            prefix = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True
            )
            prefix_ids = tokenizer(
                prefix, truncation=True, max_length=args.max_length
            )["input_ids"]
            prefix_len = len(prefix_ids)
            tokenized["labels"][i, :prefix_len] = -100

        return tokenized

    dataset = dataset.map(
        tokenize_fn, batched=True, batch_size=32,
        remove_columns=dataset.column_names,
    )

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final
    final_dir = os.path.join(args.output_dir, "final")

    if args.lora and args.merge_lora:
        # Merge LoRA weights into base model
        print("Merging LoRA weights...")
        merged = model.merge_and_unload()
        merged.save_pretrained(final_dir)
    else:
        trainer.save_model(final_dir)

    tokenizer.save_pretrained(final_dir)
    print(f"SFT training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
