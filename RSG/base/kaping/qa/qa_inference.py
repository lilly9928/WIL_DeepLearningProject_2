"""
This contains the script to proceed the QA inference with some models
"""
from transformers import pipeline
import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/RSG')
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '4,5'
from PIL import Image
import requests


def qa_inference(task: str, model_name: str, prompt: str, device=-1):
	"""
	Only use pretrained model (without any extra finetuning on any dataset)
	The idea is to pass a whole sequence into the pipeline in form of 
	 "<prompt> question: <question> answer: ""

	Tasks used are text generations/text2text generations depending on how experimented models are supported
	on Hugging Face

	This requires model to learn from input to continue generate text that is suitable for given prompt as input
	
	:param task: task for inference: text2text-generation / text-generation
	:param model_name: model used to inference: bert-large-uncased, t5-small, t5-base, t5-large, gpt2
	:param prompt: input prompt
	:param device: to use gpu or cpu (default is CPU, if using GPU, change to positive value from 0)
	:return: generated answer
	"""

	# for bert-large-uncased, t5-small, t5-base, t5-large
	if task == "text2text-generation":
		print("This task is used for bert-large-uncased and t5 models")
		qa_pipeline = pipeline(task, model=model_name, device=device)
		answer = qa_pipeline(prompt)
		print(answer)
		return answer[0]['generated_text'].split('answer : ', 1)[-1]

	# for  gpt2
	# In this inference, as huggingface pipeline does not support text2text-generation for gpt2,
	# hence text-generation was used instead
	elif task == "text-generation":
		print("This task is used for GPT2 model")
		qa_pipeline = pipeline(task, model=model_name, max_new_tokens=200, device=device)
		answer = qa_pipeline(prompt)
		return answer[0]['generated_text'].split('answer : ', 1)[-1]




		





