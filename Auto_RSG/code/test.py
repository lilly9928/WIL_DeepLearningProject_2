# from PIL import Image
# import matplotlib.pyplot as plt
#
# # Assuming 'url' is a valid path in your environment
# url = '/data2/NS/GQA/images/images/n115614.jpg'
#
# # Open and convert the image to RGB
# image = Image.open(url).convert('RGB')
#
# # Resize the image
# resized_image = image.resize((596, 437))
#
# # Visualize the image using Matplotlib
# plt.imshow(resized_image)
# plt.axis('off')  # Optional: to hide the axis
# plt.show()

from transformers import AutoTokenizer, AutoModel
import torch

# 문장 준비
sentences = ["What is this street made of?", "firetruck related to fire"]

# 사전 훈련된 모델과 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 문장을 토크나이징하고 BERT 입력 형식에 맞게 변환
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# 모델을 통해 문장 임베딩 생성
with torch.no_grad():
    model_output = model(**encoded_input)

# 문장 임베딩 추출 (여기서는 간단히 [CLS] 토큰의 출력을 사용)
embeddings = model_output.last_hidden_state[:, 0, :]

# 코사인 유사도 계산
cosine_similarity = torch.nn.CosineSimilarity(dim=1)
similarity_score = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

print(f"유사도 점수: {similarity_score}")
