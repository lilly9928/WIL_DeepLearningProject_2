import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
import cv2
import os
from tqdm import tqdm

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

data_path = '/data2/NS/GQA/images/images'

input_path = '/data2/KJE/GQA/statement/val_balanced_questions.statement.jsonl'
output_path ='/data2/KJE/GQA/statement/val_balanced_questions_object.statement.jsonl'

updated_data = []

# 파일 불러오기
with open(input_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f]

updated_data = []

for index, item in tqdm(enumerate(data), total=len(data), desc="Processing data"):
    # print(item)
    image_path = os.path.join(data_path,item['image'])
    # print(image_path)
    image = cv2.imread(image_path)
    outputs = predictor(image)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    classes = [metadata.thing_classes[i] for i in outputs["instances"].pred_classes]
    classes = set(classes)
    item['object_class'] = list(classes)
    # print(item)

    updated_data.append(item)


with open(output_path, 'w', encoding='utf-8') as f:
    for item in updated_data:
        # JSONL 형식으로 파일에 쓰기
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')