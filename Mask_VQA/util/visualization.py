import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
from .text_helper import VocabDict
# import googletrans
# translator = googletrans.Translator()

# def trans(text):#번역기
#     try:
#         if text == '<pad>' or text == '<unk>' :
#             return text
#         elif not text.encode().isalpha():#한글=>영어
#             return text
#         elif text.encode().isalpha():#영어=>한글
#             return translator.translate(text, dest='ko').text
#
#         else:
#             return text
#
#     except:
#         return text


def print_examples(model,data_path,dataset,max_qst_length = 30):
    input_dir = 'D:/data/vqa/coco/simple_vqa'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    qst_vocab = VocabDict(input_dir + '/vocab_questions.txt')

    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        ]
    )
    model.eval()

    testdata = np.load(data_path, allow_pickle=True)
    num = random.randint(0, len(testdata))

    image_path = testdata[num]['image_path']
    image = testdata[num]['image_path']
    image = Image.open(image).convert('RGB')
    image = transform(image)
    question = testdata[num]['question_str']

    qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
    qst2idc[:len(testdata[num]['question_tokens'])] = [qst_vocab.word2idx(w) for w in testdata[num]['question_tokens']]
    qst2idc=torch.Tensor(qst2idc)
    res=model.visualization_vqa(image.to(device).float(), question.to(device).long(), dataset.ans_vocab)

    return image_path,question,res


def random_examples(data_path,max_qst_length = 30):
    input_dir = 'D:/data/vqa/coco/simple_vqa'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    qst_vocab = VocabDict(input_dir + '/vocab_questions.txt')

    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        ]
    )

    testdata = np.load(data_path, allow_pickle=True)
    num = random.randint(0, len(testdata))

    image_path = testdata[num]['image_path']
    image = testdata[num]['image_path']
    image = Image.open(image).convert('RGB')
    image = transform(image)
    question = testdata[num]['question_str']

    qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
    qst2idc[:len(testdata[num]['question_tokens'])] = [qst_vocab.word2idx(w) for w in testdata[num]['question_tokens']]
    qst2idc=torch.Tensor(qst2idc)


    return image_path,image,qst2idc

def print_korean_examples(model,data_path,dataset,max_qst_length = 30):
    input_dir = 'D:/data/vqa/coco/simple_vqa'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qst_vocab = VocabDict(input_dir + '/vocab_questions.txt')

    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        ]
    )
    model.eval()

    testdata = np.load(data_path, allow_pickle=True)
    num = random.randint(0, len(testdata))

    image_path = testdata[num]['image_path']
    image = testdata[num]['image_path']
    image = Image.open(image).convert('RGB')
    image = transform(image)
    question = testdata[num]['question_str']
    # question = translator.translate(testdata[num]['question_str'],dest='ko').text

    qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
    qst2idc[:len(testdata[num]['question_tokens'])] = [qst_vocab.word2idx(w) for w in testdata[num]['question_tokens']]
    qst2idc=torch.Tensor(qst2idc)
    res=model.visualization_vqa(image.to(device).float(), qst2idc.to(device).long(), dataset.ans_vocab)
    # res = translator.translate(res[0], dest='ko').text

    return image_path,question,res