import json
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm
import os
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로더 정의
def load_data_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data


data_file_path = "/data2/KJE/GQA/statement/testdev_balanced_questions_promptcap_relation_2.statement.jsonl"  # 데이터 파일 경로
data = load_data_from_jsonl(data_file_path)

# 이미지, 질문 및 정답 데이터 추출
image_urls = [item['image'] for item in data]
questions = [item['question']['stem'] for item in data]
answers = [item['question']['choices']['label'].lower() for item in data]
relations = [item['relation'] for item in data]

data_path = '/data2/NS/GQA/images/images'
output_file_path = "../logs/train_Blip2_output.txt"

# Blip2 모델
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
Blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

#bert
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
Blip2_model.to(device)
bert_model.to(device)

# 배치 크기 정의
batch_size = 256  # 한 번에 처리할 이미지 및 질문 쌍의 수

output = []
correct_predictions = 0
total_predictions = 0
bleu_scores = []

# 데이터를 배치로 나누어 처리하고 결과 평가
for batch_start in tqdm(range(0, len(image_urls), batch_size), desc="Evaluating"):
    # 배치 데이터 선택
    batch_image_urls = image_urls[batch_start:batch_start + batch_size]
    batch_questions = questions[batch_start:batch_start + batch_size]
    batch_answers = answers[batch_start:batch_start + batch_size]
    batch_relations = relations[batch_start:batch_start + batch_size]

    # 이미지와 질문을 하나의 배치로 처리하고 결과 평가
    for image_url, question, answer,relation in zip(batch_image_urls, batch_questions, batch_answers,batch_relations):
        # 이미지 불러오기
        image = Image.open(os.path.join(data_path, image_url)).convert('RGB')

        # 이미지 리사이징 및 출력
        image = image.resize((596, 437))


        # 질문과 이미지를 입력으로 사용하여 텍스트 생성
        prompt = f"Question: {question} Answer: "
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

        # 모델
        generated_ids = Blip2_model.generate(**inputs, max_new_tokens=10)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # 생성된 텍스트와 정답 비교하여 정확도 측정
        total_predictions += 1
        if generated_text.lower() == answer:
            correct_predictions += 1

        # BLEU-2 점수 계산
        bleu_score = sentence_bleu([answer.split()], generated_text.split(), weights=(0.5, 0.5))
        bleu_scores.append(bleu_score)

        output.append({
            "question": question,
            "generated_text": generated_text,
            "answer": answer
        })

# 정확도 계산
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU-2 Score: {average_bleu_score:.4f}")

# 결과 저장
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # 결과 출력
    json.dump(output, output_file, ensure_ascii=False, indent=4)
