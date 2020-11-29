from transformers import MBartForConditionalGeneration, MBartTokenizer

model = MBartForConditionalGeneration.from_pretrained("./results")
print('test')
tokenizer = MBartTokenizer.from_pretrained("./results")
print('test')
article = "person ski"
print('test')
batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], src_lang="en_XX")
print('test')
translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
print('test')
translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print(translation)

# assert translation == "Şeful ONU declară că nu există o soluţie militară în Siria"