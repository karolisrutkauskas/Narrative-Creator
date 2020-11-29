from transformers import MBartTokenizer, MBartForConditionalGeneration, Trainer, TrainingArguments
import torch
from utils import NarrativesDataset, NarrativesDataCollator
from sklearn.model_selection import train_test_split

import json
import sys
import os

def read_dataset(path):
    with open(path, 'r') as jsonl_file:
        json_list = list(jsonl_file)
        object_list = []
        narrative_list = []
        for item in json_list:
            result = json.loads(item)
            object_list.append(' '.join(result['objects']))
            narrative_list.append(result['narrative'])
    
    return object_list, narrative_list

dataset_path = sys.argv[1]

objects, narratives = read_dataset(dataset_path)

objects = objects[:1000]
narratives = narratives[:1000]

model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

train_objects, val_objects, train_narratives, val_narratives = train_test_split(objects, narratives, test_size=.2)

train_dataset = NarrativesDataset(tokenizer, train_objects, train_narratives)
val_dataset = NarrativesDataset(tokenizer, val_objects, val_narratives)

print(train_dataset.__len__())
print(val_dataset.__len__())

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=7,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=NarrativesDataCollator(tokenizer)
)

trainer.train()
trainer.save_model()
trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
tokenizer.save_pretrained(training_args.output_dir)