from transformers import MBartForConditionalGeneration, MBartTokenizer
import sys

objects = sys.argv[1:]
objects = " ".join(objects)

model = MBartForConditionalGeneration.from_pretrained("./results")
tokenizer = MBartTokenizer.from_pretrained("./results")
batch = tokenizer.prepare_seq2seq_batch(src_texts=[objects], src_lang="en_XX")
translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
narrative = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print(narrative)