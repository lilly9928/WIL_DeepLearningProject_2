import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from ImageCaption_get_loader import get_loader
import torch.nn.functional as F
from tqdm import tqdm
from ImageCaption_model import CNNtoRNN
from captionData import CaptionDataset
from PIL import Image


# Parameters
data_folder = 'D:/data/vqa/coco/simple_vqa/cococaption'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\my_checkpoint_coco_30.pth.tar'  # model checkpoint
checkpoint_pth = 'D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\my_checkpoint_coco_30.pth'
#word_map_file = '/media/ssd/caption data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_loader, dataset = get_loader(
    root_folder="D:/data/vqa/coco/simple_vqa/Images/train2014/",
    annotation_file="D:/data/vqa/coco/simple_vqa/captions.txt",
    transform=transform,
    num_workers=2
)
# 하이퍼파라미터
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 2
learning_rate = 3e-4
num_epochs = 30

model = CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers).to(device)
model.load_state_dict(torch.load(checkpoint_pth)['state_dict'])
model.eval()

test_img1 = transform(Image.open("D:/data/vqa/coco/simple_vqa/test_image/1.jpg").convert("RGB")).unsqueeze(
    0
)

outputs = model.caption_images(image, dataset.vocab)