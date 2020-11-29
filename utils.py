import transformers
from transformers.modeling_bart import shift_tokens_right
import torch
import config as config
from typing import Dict

class NarrativesDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, src, tgt):
        self.tokenizer = tokenizer
        self.src = src
        self.tgt = tgt

    def __getitem__(self, idx) -> Dict[str, str]:
        src_text = self.src[idx]
        tgt_text = self.tgt[idx]
        return {"src_text": src_text, "tgt_text": tgt_text, "id": idx}

    def __len__(self):
        return len(self.tgt)

class NarrativesDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Dict[str, torch.tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [row["src_text"] for row in batch],
            tgt_texts=[row["tgt_text"] for row in batch],
            src_lang="en_XX",
            tgt_lang="en_XX",
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_encoding = batch_encoding.data

        decoder_input_ids = shift_tokens_right(batch_encoding["labels"], self.tokenizer.pad_token_id)

        batch_encoding = {
            "input_ids": batch_encoding["input_ids"],
            "attention_mask": batch_encoding["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "labels": batch_encoding["labels"],
        }
        return batch_encoding