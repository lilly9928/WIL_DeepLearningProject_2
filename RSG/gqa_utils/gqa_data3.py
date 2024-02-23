import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from itertools import islice


os.environ["CUDA_VISIBLE_DEVICES"]='3'
# model = AutoModel.from_pretrained("cmp-nct/llava-1.6-gguf")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

data_root ="/data2/NS/GQA"  
image_data_path = '/data2/NS/GQA/images/images'
image_prompt1 = "the question of this photography is "
image_prompt2 = "the image is a photography of"

repo_root = '/data2/KJE'
os.system(f'mkdir -p {repo_root}/GQA/statement')

#"train_balanced_questions","testdev_balanced_questions","val_balanced_questions"
for fname_pattern in ["val_balanced_questions"]:
  
    if fname_pattern == "train_all_questions":
        range_limit = 10  
    
    else:
        range_limit = 1  
     
        
    for i in range(range_limit):
        if fname_pattern == "train_all_questions":
            fname = f"{fname_pattern}_{i}" 
            file_path = f"{data_root}/questions/train_all_questions/{fname}.json"
          
        else:
            fname = f"{fname_pattern}"
            file_path = f"{data_root}/questions/{fname}.json"
         
            
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)
            

        examples = []
        
        #data = dict(islice(data.items(),2500))
        
        
       
        for _id, item in tqdm(data.items()):
            
            img_path = os.path.join(image_data_path,item["imageId"]+'.jpg')
            image = Image.open(img_path).convert('RGB')
            model_input = processor(image, image_prompt2, return_tensors="pt").to("cuda")
            out = model.generate(**model_input)
            image_caption=processor.decode(out[0], skip_special_tokens=True)
            
    
            question_stem = item["question"]
            imageId = item["imageId"]
            answer = item.get("answer", "N/A")
            
            

            if 'test' in fname_pattern:  
                answerKey = 'N'  
            else:
                answerKey = 'A' #item["answer"]

            fullAnswer = item.get("fullAnswer", "N/A")  

            ex_obj = {
                "id": _id,
                "question": {
                    "stem": question_stem,
                    "choices": {"label": answer, "text": answer}},
                "answerKey": answerKey,
                "statements": {
                    "label":True,"statement": fullAnswer
                },
                "image_caption":image_caption,
                "image": f"{imageId}.jpg"
            }
            examples.append(ex_obj)
            
   
        with open(f"{repo_root}/GQA/statement/{fname}.statement.json", 'w') as fout:
            print(f'{fname} start')

            for dic in tqdm(examples):
                print (json.dumps(dic), file=fout)
        print(f'{fname} finished')
