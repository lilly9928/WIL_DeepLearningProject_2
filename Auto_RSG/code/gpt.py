import json
from PIL import Image
import torch
from tqdm import tqdm
import os
from nltk.translate.bleu_score import sentence_bleu
import openai

#TODO
#데이터로더, 처리
#gpt3.5 engine

# 데이터 로더 정의
def load_data_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {line}")
                print(f"Error message: {e}")
    return data
"""
def load_data_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data
"""


#/data2/KJE/OK-VQA/statement/updated_OpenEnded_mscoco_val2014_questions_processed.json

#/data2/KJE/GQA/statement/val_balanced_questions.statement.json

#data_file_path = "/data2/KJE/GQA/statement/train_balanced_questions_object_relation.statement.jsonl"  # 데이터 파일 경로

data_file_path = "/data2/KJE/GQA/statement/val_balanced_questions.statement.json"
data = load_data_from_jsonl(data_file_path)

# 이미지, 질문 및 정답 데이터 추출
# Example of extracting specific fields from the loaded data
questions = [item['question']['stem'] for item in data]
answers = [item['question']['choices']['label'] for item in data]
image_captions = [item['image_caption'] for item in data]
images = [item['image'] for item in data]
relations = [item.get('relation', []) for item in data]

"""
image_urls = [item['image'] for item in data]
questions = [item['question']['stem'] for item in data]
answers = [item['question']['choices']['label'].lower() for item in data]
relations = [sum(item['relation'],[]) for item in data]
"""


data_path = '/data2/NS/GQA/images/images'
output_file_path = "../logs/gpt_gqa_val_balanced_questions_relation_output.txt"

# Blip2 모델 및 프로세서 불러오기
#processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
#model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

#gpt turbo 불러오기
openai.api_key = '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' #kimjian key

# 장치 설정
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)

# 배치 크기 정의
batch_size = 64  # 한 번에 처리할 이미지 및 질문 쌍의 수

output = []
correct_predictions = 0
total_predictions = 0
bleu_scores = []

# Process data in batches and evaluate results
for i in tqdm(range(len(questions)), desc="Evaluating"):
    question = questions[i]
    answer = answers[i]
    relation = relations[i]
    relation_str = ', '.join(set(relation))  # Convert relation list to string
    prompt = f"Use summary to answer.\nsummary: {relation_str} Question: {question} Answer in word. Answer: "

    # Call OpenAI API to generate text
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # Use the GPT-3.5 Turbo engine
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=["\n"],
        temperature=0.7
    )
    generated_text = response.choices[0].text.strip()

    # Compare generated text with the correct answer
    total_predictions += 1
    correct = generated_text.lower().strip() == answer.lower().strip()
    if correct:
        correct_predictions += 1

    # Calculate BLEU score
    reference = [answer.split()]
    candidate = generated_text.split()
    bleu_score = sentence_bleu(reference, candidate, weights=(0.5, 0.5))
    bleu_scores.append(bleu_score)

    # Log results for comparison
    result = {
        "question": question,
        "generated_text": generated_text,
        "reference_answer": answer,
        "correct": correct
    }
    output.append(result)

# Calculate accuracy and average BLEU score
accuracy = correct_predictions / total_predictions * 100
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

# Output results and metrics, and save to a file
metrics = {
    "accuracy": accuracy,
    "average_bleu_score": average_bleu_score
}

print(f"Accuracy: {accuracy:.2f}%")
print(f"Average BLEU Score: {average_bleu_score:.4f}")

# Save the results and metrics to the specified output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump({"results": output, "metrics": metrics}, output_file, ensure_ascii=False, indent=4)

"""
# 데이터를 배치로 나누어 처리하고 결과 평가
for i in tqdm(range(len(questions)), desc="Evaluating"):
    question = questions[i]
    answer = answers[i]
    relation = relations[i]
    relation_str = str(set(relation))
    prompt = f"{relation_str} Question: {question} Answer: "

    # OpenAI API를 사용하여 텍스트 생성
    response = openai.Completion.create(
        #engine="text-davinci-003",  # GPT-3.5 모델 선택
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    generated_text = response.choices[0].text.strip()

    # 생성된 텍스트와 정답 비교
    total_predictions += 1
    correct = generated_text.lower() == answer
    if correct:
        correct_predictions += 1

    # BLEU 점수 계산
    reference = [answer.split()]
    candidate = generated_text.split()
    bleu_score = sentence_bleu(reference, candidate, weights=(0.5, 0.5))
    bleu_scores.append(bleu_score)

    # 결과 및 비교 로그
    result = {
        "question": question,
        "generated_text": generated_text,
        "reference_answer": answer,
        "correct": correct
    }
    output.append(result)

# 정확도 및 평균 BLEU 점수 계산
accuracy = correct_predictions / total_predictions * 100
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

# 결과 및 메트릭스 출력 및 파일 저장
metrics = {
    "accuracy": accuracy,
    "average_bleu_score": average_bleu_score
}

print(f"Accuracy: {accuracy:.2f}%")
print(f"Average BLEU Score: {average_bleu_score:.4f}")

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump({"results": output, "metrics": metrics}, output_file, ensure_ascii=False, indent=4)
"""
"""
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

        relation_str = str(set(relation))

        # 질문과 이미지를 입력으로 사용하여 텍스트 생성
        # prompt = f"Question: {question} Answer: "
        prompt = f"{relation_str} Question: {question} Answer: "
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

        # 모델
        generated_ids = model.generate(**inputs, max_new_tokens=10)
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
"""