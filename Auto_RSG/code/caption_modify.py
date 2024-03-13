import json
from detectron2.data import MetadataCatalog
import cv2
import os
from tqdm import tqdm
import torch
from promptcap import PromptCap

model = PromptCap("tifa-benchmark/promptcap-coco-vqa")

if torch.cuda.is_available():
  model.cuda()

data_path = '/data2/KJE/okvqa/okvqa_images(coco)/val2014'

input_path = '/data2/KJE/OK-VQA/statement/updated_OpenEnded_mscoco_val2014_questions_processed.json'

output_path ='/data2/KJE/OK-VQA/statement/updated_OpenEnded_mscoco_val2014_questions_processed_promptcap.jsonl'

updated_data = []

# 파일 불러오기
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 형식 오류가 발생했습니다: {file_path}, 오류: {e}")
        return None

data = load_json_file(input_path)


updated_data = []

for index, item in tqdm(enumerate(data), total=len(data), desc="Processing data"):

    image_path = os.path.join(data_path,item['image'])
    question = item["question"]

    prompt = f" please describe this image according to the given question:{question} "
    outputs = model.caption(prompt,image_path)


    item['image_caption'] = str(outputs)
    # print(item)

    updated_data.append(item)


with open(output_path, 'w', encoding='utf-8') as f:
    for item in updated_data:
        # JSONL 형식으로 파일에 쓰기
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')