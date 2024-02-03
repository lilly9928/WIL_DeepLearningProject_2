import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
from models import VqaModel
from util import text_helper

def print_examples(model,data_path,vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        ]
    )
    model.eval()

    for i in range(5):
        testdata = np.load(data_path, allow_pickle=True)
        num = random.randint(0, len(testdata))
        image = testdata[num]['image_path']
        image = Image.open(image).convert('RGB')


        question = testdata[num]['question_str']
        print(image)
        print(question)

        print(
            "Example", i ,"OUTPUT: "
            + " ".join(model.visualization_vqa(image.to(device), vocab))
        )



    # model.eval()
    # test_img1 = transform(Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/dog.jpg").convert("RGB")).unsqueeze(
    #     0
    # )
    # print("Example 1 CORRECT: Dog on a beach by the ocean")
    # print(
    #     "Example 1 OUTPUT: "
    #     + " ".join(model.caption_images(test_img1.to(device), dataset.vocab))
    # )
    # test_img2 = transform(
    #     Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/child.jpg").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 2 CORRECT: Child holding red frisbee outdoors")
    # print(
    #     "Example 2 OUTPUT: "
    #     + " ".join(model.caption_images(test_img2.to(device), dataset.vocab))
    # )
    # test_img3 = transform(Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/bus.png").convert("RGB")).unsqueeze(
    #     0
    # )
    # print("Example 3 CORRECT: Bus driving by parked cars")
    # print(
    #     "Example 3 OUTPUT: "
    #     + " ".join(model.caption_images(test_img3.to(device), dataset.vocab))
    # )
    # test_img4 = transform(
    #     Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/boat.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 4 CORRECT: A small boat in the ocean")
    # print(
    #     "Example 4 OUTPUT: "
    #     + " ".join(model.caption_images(test_img4.to(device), dataset.vocab))
    # )
    # test_img5 = transform(
    #     Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/horse.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    # print(
    #     "Example 5 OUTPUT: "
    #     + " ".join(model.caption_images(test_img5.to(device), dataset.vocab))
    # )
    # model.train()



if __name__ == "__main__":
    test_path = 'D:/data/vqa/coco/simple_vqa/test.npy'
    input_dir = 'D:/data/vqa/coco/simple_vqa'
    ans_vocab = text_helper.VocabDict(input_dir + '/vocab_answers.txt')

    print_examples(VqaModel,test_path,ans_vocab)