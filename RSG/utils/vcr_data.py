import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'

repo_root = '/data2/KJE'

answer_label = ['A', 'B', 'C', 'D']

# Prepare `statement` data following CommonsenseQA, OpenBookQA
data_root = f'{repo_root}/VCR'
os.system(f'mkdir -p {data_root}/statement')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

image_data_path = '/data2/KJE/VCR/vcr1images'
image_prompt = "a photography of"

for fname in ["train", "val", "test"]:
    with open(f"{data_root}/{fname}.jsonl") as f:
        lines = f.readlines()
    examples = []

    for i in tqdm(range(len(lines))):
        line = json.loads(lines[i])
        _id = f"train-{i:05d}"

        img_path = os.path.join(image_data_path, line['img_fn'])
        image = Image.open(img_path).convert('RGB')
        model_input = processor(image, image_prompt, return_tensors="pt").to("cuda")
        out = model.generate(**model_input)
        image_caption = processor.decode(out[0], skip_special_tokens=True)

        if fname == 'test':
            answerKey = 'N'
            stem = ' '.join(map(str, line["question"]))

            stmts = [{"statement": ' '.join(map(str, line["rationale_choices"][c]))} for c in
                     range(len(line["rationale_choices"]))]


        else:

            answerKey = answer_label[line["answer_label"]]
            stem = line["question_orig"]

            stmts = []
            for idx in range(len(line["rationale_choices"])):
                r_choice_join = ' '.join(map(str, line["rationale_choices"][idx]))
                if idx == line["rationale_label"]:
                    stmts.append({"label": True, "statement": r_choice_join})
                else:
                    stmts.append({"label": False, "statement": r_choice_join})

        answer_choices_list = line["answer_choices"]
        choices = [{"label": answer_label[k], "text": ' '.join(map(str, answer_choices_list[k]))} for k in
                   range(len(answer_choices_list))]

        ex_obj = {"id": _id,
                  "question": {"stem": stem, "choices": choices},
                  "answerKey": answerKey,
                  "statements": stmts,
                  "image_caption": image_caption,
                  "image_path": img_path
                  }

        examples.append(ex_obj)
    with open(f"{data_root}/statement/{fname}.statement.jsonl", 'w') as fout:
        # json.dump(examples,fout)
        print(f'{fname} start')
        for dic in tqdm(examples):
            print(json.dumps(dic), file=fout)