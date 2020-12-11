from transformers import BartForConditionalGeneration, BartTokenizer
import sys

objects = sys.argv[1:]
objects = " ".join(objects)

model = BartForConditionalGeneration.from_pretrained("./results")
tokenizer = BartTokenizer.from_pretrained("./results")
batch = tokenizer.prepare_seq2seq_batch(src_texts=[objects], src_lang="en_XX", padding=True, truncation=True, return_tensors="pt")
translated_tokens = model.generate(**batch, max_length=128)
narrative = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print(narrative)