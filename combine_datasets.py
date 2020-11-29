import sys
import json

objects_dataset = sys.argv[1]
narrative_dataset = sys.argv[2]

with open(objects_dataset, 'r') as jsonl_file:
    json_list = list(jsonl_file)
    object_list = {}
    for item in json_list:
        result = json.loads(item)
        object_list[result['image_name']] = result['objects']

with open(narrative_dataset, 'r') as jsonl_file:
    json_list = list(jsonl_file)
    narrative_list = {}
    for item in json_list:
        result = json.loads(item)
        narrative_list[result['image_id'] + '.jpg'] = result['caption']

if len(object_list) != len(narrative_list):
    print('datasets don\'t match')

with open('data/dataset.jsonl', 'a') as jsonl_file:
    for line in object_list:
        print(object_list[line])
        if len(object_list[line]) != 0:
            data = {'objects': object_list[line], 'narrative': narrative_list[line]}
            jsonl_file.write(json.dumps(data))
            jsonl_file.write('\n')