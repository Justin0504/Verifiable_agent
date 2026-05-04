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
    DataCollatorForSeq2Seq,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-file", type=str,
                        default="data/attribution/seva_sft_train.jsonl")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/seva_sft")
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
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from checkpoint directory")
    parser.add_argument("--optim-8bit", action="store_true",
                        help="Use Adafactor optimizer to reduce memory (for single GPU)")
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
        all_input_ids = []
        all_labels = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized = tokenizer(
                text, truncation=True, max_length=args.max_length
            )
            input_ids = tokenized["input_ids"]

            # Mask system + user tokens (only train on assistant response)
            # Find all non-assistant messages to build prefix
            assistant_idx = None
            for idx, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    assistant_idx = idx
                    break
            if assistant_idx is not None and assistant_idx > 0:
                prefix = tokenizer.apply_chat_template(
                    messages[:assistant_idx], tokenize=False,
                    add_generation_prompt=True
                )
                prefix_ids = tokenizer(
                    prefix, truncation=True, max_length=args.max_length
                )["input_ids"]
                prefix_len = len(prefix_ids)
            else:
                prefix_len = 0

            labels = list(input_ids)
            for j in range(min(prefix_len, len(labels))):
                labels[j] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "labels": all_labels,
                "attention_mask": [[1] * len(ids) for ids in all_input_ids]}

    dataset = dataset.map(
        tokenize_fn, batched=True, batch_size=32,
        remove_columns=dataset.column_names,
    )

    # Training
    # Detect distributed training for DeepSpeed ZeRO-3
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    ds_args = {}
    if is_distributed and not args.lora:
        ws = os.environ.get("WORLD_SIZE", "1")
        print(f"Distributed training: WORLD_SIZE={ws}, using DeepSpeed ZeRO-3")
        ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "none"},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": 100,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False,
        }
        ds_config_path = os.path.join(args.output_dir, "ds_config.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)
        ds_args["deepspeed"] = ds_config_path

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adafactor" if args.optim_8bit else "adamw_torch",
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        **ds_args,
    )

    # Dynamic padding: pad to longest in batch, not max_length
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

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
