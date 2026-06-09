"""
Optional training skeleton for later improvement.
Use this after the MVP works. It fine-tunes Whisper on FLEURS Swahili subset.
For a same-day demo, use app.py without training.

Run on GPU:
    python scripts/train_whisper_swahili.py
"""
from __future__ import annotations

import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

MODEL_NAME = "openai/whisper-small"
LANG = "sw_ke"  # FLEURS Swahili config; inspect google/fleurs configs if unavailable.


def main():
    ds = load_dataset("google/fleurs", LANG)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="Swahili", task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Swahili", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = "swahili"
    model.generation_config.task = "transcribe"

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=1)

    class Collator:
        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    args = Seq2SeqTrainingArguments(
        output_dir="whisper-small-swahili-fleurs",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=1000,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",           # renamed from evaluation_strategy in transformers>=4.45
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250,
        eval_steps=250,
        logging_steps=25,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=Collator(),
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model("whisper-small-swahili-fleurs/final")


if __name__ == "__main__":
    main()
