import json
from tqdm import tqdm
import openai
import time

# OpenAI API 키 설정
openai.api_key = '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'


# 데이터 로드 함수 정의
def load_data_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data


data_file_path = "/data2/KJE/OK-VQA/statement/updated_OpenEnded_mscoco_val2014_questions_processed_promptcap_relation.jsonl"
data = load_data_from_jsonl(data_file_path)

correct_predictions = 0
total_predictions = 0
# 배치 크기 정의
batch_size = 32

output = []  # 결과 저장을 위한 리스트

# 데이터를 배치로 나누어 처리하고 결과 평가
for i in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
    batch_data = data[i:i + batch_size]
    for item in batch_data:
        # question = item['question']['stem']
        # answer = item['question']['choices']['label'].lower()  # 또는 적절한 필드 사용
        # image_caption = item['image_caption']
        # relation = item['relation']

        question = item['question']
        answer = item['statement'].lower()  # 또는 적절한 필드 사용
        image_caption = item['image_caption']
        relation = item['relation']

        # 텍스트 생성 입력 생성
        prompt = f" Use summary to answer.\n summary:{image_caption}\n  Below are the facts that might be relevant to answer the question:{relation}  Question: {question} \nAnswer in word."

        # OpenAI API를 사용하여 텍스트 생성
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",  # 사용할 GPT 모델 지정
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        except openai.error.ServiceUnavailableError:
            print("서비스 이용 불가능. 5초 후에 다시 시도합니다.")
            time.sleep(5)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0125",  # 사용할 GPT 모델 지정
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            except openai.error.ServiceUnavailableError:
                # 여전히 서비스가 이용 불가능한 경우에 대한 처리
                print("서비스가 여전히 이용 불가능합니다. 나중에 다시 시도하세요.")


        generated_text = response['choices'][0]['message']['content'].strip()

        total_predictions += 1
        if generated_text.lower() == answer:
            correct_predictions += 1

        # 결과 저장
        output.append({
            "question": question,
            "generated_text": generated_text,
            "answer": answer
        })

accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")

# 결과를 JSON 파일로 저장create
output_file_path = "../logs/okvqa_val_gpt_evaluation_relation_output.txt"
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)

print(f"Saved generated texts and their evaluations to {output_file_path}")