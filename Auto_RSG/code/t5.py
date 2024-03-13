import json
import os
from tqdm import tqdm
from PIL import Image
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu

# 데이터 로드 함수 정의
def load_data_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

data_file_path = "/data2/KJE/OK-VQA/statement/updated_OpenEnded_mscoco_val2014_questions_processed_promptcap_relation.jsonl"  # 데이터 파일 경로
data = load_data_from_jsonl(data_file_path)

# 이미지, 질문, 정답 데이터 추출
# image_captions = [item['image_caption'] for item in data]
# questions = [item['question']['stem'] for item in data]
# answers = [item['question']['choices']['label'].lower() for item in data]
# relations = [item['relation'] for item in data]

image_captions = [item['image_caption'] for item in data]
questions = [item['question'] for item in data]
answers = [item['statement'].lower() for item in data]
relations = [item['relation'] for item in data]

data_path = '/data2/NS/GQA/images/images'
output_file_path = "../logs/okvqa_T5_promptcap_output_.txt"

# T5 모델 및 토크나이저 불러오기
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-t5-base")

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 배치 크기 정의
batch_size = 64  # 한 번에 처리할 이미지 및 질문 쌍의 수

output = []
correct_predictions = 0
total_predictions = 0
bleu_scores = []

# 데이터를 배치로 나누어 처리하고 결과 평가
for batch_start in tqdm(range(0, len(image_captions), batch_size), desc="Evaluating"):
    # 배치 데이터 선택
    batch_image_captions = image_captions[batch_start:batch_start + batch_size]
    batch_questions = questions[batch_start:batch_start + batch_size]
    batch_answers = answers[batch_start:batch_start + batch_size]
    batch_relations = relations[batch_start:batch_start + batch_size]

    # 이미지와 질문을 하나의 배치로 처리하고 결과 평가
    for image_caption, question, answer, relation in zip(batch_image_captions, batch_questions, batch_answers, batch_relations):

        # 텍스트 생성 입력 생성
        # input_text = f"Use summary to answer. summary:{image_caption} Below are the facts that might be relevant to answer the question: {relation} Question: {question} Answer in word. Answer:"
        # input_text = f"trivia question: {question} context:{image_caption} {relation} "
        # input_text = (f"question: {question} context:{image_caption} {relation}")
        input_text = (f"{question}  \n {image_caption}")
        # 토크나이징 및 패딩
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        # 모델에 입력 전달하여 텍스트 생성
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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

# BLEU 점수 평균 계산
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU-2 Score: {average_bleu_score:.4f}")

# 결과 저장
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # 결과 출력
    json.dump(output, output_file, ensure_ascii=False, indent=4)
