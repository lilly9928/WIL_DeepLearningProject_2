import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from util import text_helper
import spacy

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=30,max_cap_length=100, max_num_ans=10, transform=None):
        self.df = pd.read_csv(input_dir+'/captions.txt')
        self.captionn=self.df["captions"]
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa,allow_pickle = True)
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
        self.cap_vocab = text_helper.VocabDict(input_dir+'/vocab_caption.txt')
        self.max_qst_length = max_qst_length
        self.max_cap_length = max_cap_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform
        self.vocab = Vocabulary(3)
        self.vocab.build_vocabulary(self.captionn.tolist())

    def __getitem__(self, idx):

        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        cap_vocab = self.cap_vocab
        max_qst_length = self.max_qst_length
        max_cap_length = self.max_cap_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans
        vocab = self.vocab
        caption_idx = random.choice(range(0,4))

        image = vqa[idx]['image_path']
        image = Image.open(image).convert('RGB')
        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
        sample = {'image': image, 'question': qst2idc}

        if load_ans:
            ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']]
            ans2idx = np.random.choice(ans2idc)
            sample['answer_label'] = ans2idx         # for training

            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

            numericalized_caption = [vocab.stoi["<SOS>"]]
            numericalized_caption += vocab.numericalize(vqa[idx]['caption'][caption_idx])
            numericalized_caption.append(vocab.stoi["<EOS>"])

            sample['caption'] = numericalized_caption

        if transform:
            sample['image'] = transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length,max_cap_length, max_num_ans, batch_size, num_workers):

    transform = {
        phase: transforms.Compose([
                                transforms.Resize((356, 356)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))])
        for phase in ['train', 'valid',]}

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_cap_length=max_cap_length,
            max_num_ans=max_num_ans,
            transform=transform['train']),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_cap_length=max_cap_length,
            max_num_ans=max_num_ans,
            transform=transform['valid']),

    }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader

if __name__ == "__main__":



    input_dir = 'D:/data/vqa/coco/simple_vqa'
    log_dir = './logs'
    model_dir='./models'
    max_qst_length = 30
    max_cap_length= 50
    max_num_ans =10
    embed_size=64
    word_embed_size=300
    num_layers=2
    hidden_size=16
    learning_rate = 0.001
    step_size = 10
    gamma = 0.1
    num_epochs=30
    batch_size = 64
    num_workers = 0
    save_step=1


    data = get_loader(
        input_dir=input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=max_qst_length,
        max_cap_length=max_cap_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers)

    print(data['train'])