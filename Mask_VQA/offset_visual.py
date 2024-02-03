import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from models import VqaModel
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
from data_loader import get_loader
from util.visualization import random_examples

def plot_offsets(img, save_output, roi_x, roi_y):
    cv2.circle(img, center=(roi_x, roi_y), color=(0, 255, 0), radius=1, thickness=-1)
    input_img_h, input_img_w = img.shape[:2]
    for offsets in save_output.outputs:
        offset_tensor_h, offset_tensor_w = offsets.shape[2:]
        resize_factor_h, resize_factor_w = input_img_h / offset_tensor_h, input_img_w / offset_tensor_w

        offsets_y = offsets[:, ::2]
        offsets_x = offsets[:, 1::2]

        grid_y = np.arange(0, offset_tensor_h)
        grid_x = np.arange(0, offset_tensor_w)

        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        sampling_y = grid_y + offsets_y.detach().cpu().numpy()
        sampling_x = grid_x + offsets_x.detach().cpu().numpy()

        sampling_y *= resize_factor_h
        sampling_x *= resize_factor_w

        sampling_y = sampling_y[0]  # remove batch axis
        sampling_x = sampling_x[0]  # remove batch axis

        sampling_y = sampling_y.transpose(1, 2, 0)  # c, h, w -> h, w, c
        sampling_x = sampling_x.transpose(1, 2, 0)  # c, h, w -> h, w, c

        sampling_y = np.clip(sampling_y, 0, input_img_h)
        sampling_x = np.clip(sampling_x, 0, input_img_w)

        sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
        sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

        sampling_y = sampling_y[roi_y, roi_x]
        sampling_x = sampling_x[roi_y, roi_x]

        for y, x in zip(sampling_y, sampling_x):
            y = round(y)
            x = round(x)
            cv2.circle(img, center=(x, y), color=(0, 0, 255), radius=1, thickness=-1)

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dir = 'D:/data/vqa/coco/simple_vqa'
log_dir = './logs'
model_dir='./models'
max_qst_length = 30
max_cap_length=50
max_num_ans =10
embed_size=64
word_embed_size=300
num_layers=2
hidden_size=32
learning_rate = 0.001
step_size = 10
gamma = 0.1
num_epochs=30
batch_size = 16
num_workers = 4
save_step=1

data_loader = get_loader(
 input_dir=input_dir,
 input_vqa_train='train.npy',
 input_vqa_valid='test.npy',
 max_qst_length=max_qst_length,
 max_num_ans=max_num_ans,
 batch_size=batch_size,
 num_workers=num_workers)

qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
#model = models.resnet18(pretrained=True)
model = VqaModel(embed_size=embed_size,
                     qst_vocab_size=qst_vocab_size,
                     ans_vocab_size=ans_vocab_size,
                   #  cap_vocab_size = cap_vocab_size,
                     word_embed_size=word_embed_size,
                     num_layers=num_layers,
                     hidden_size=hidden_size)


model.load_state_dict(torch.load("./logs/221007/model-epoch-50.ckpt"),strict=False)
model.to(device)
model.eval()

save_output = SaveOutput()
hook_handles = []

for name, layer in model.named_modules():
    if "offset_conv" in name and isinstance(layer, nn.Conv2d):
        print(name, layer)
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)


image_path, img,question =random_examples('D:/data/vqa/coco/simple_vqa/test.npy')

#image = cv2.imread("D:/data/vqa/coco/simple_vqa/Images/train2014/COCO_train2014_000000533158.jpg",cv2.IMREAD_COLOR)
# transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
# X = transform(image).unsqueeze(dim=0).to(device)
# Q = 'What color is the photo?'

out = model(img.unsqueeze(dim=0).to(device),question.unsqueeze(dim=0).to(device).long())
#print(out)


print(len(save_output.outputs))

#image = cv2.imread(image_path, cv2.IMREAD_COLOR)
iamge = Image.open(image_path)
image = np.array(iamge)
# plt.imshow(image)
# plt.show()

input_img_h, input_img_w = image.shape[:2]
roi_point_y, roi_point_x = input_img_h // 2, input_img_w // 2

image_bgr = np.repeat(image[..., np.newaxis], 3, axis=-1)

for offsets in save_output.outputs:
    offset_tensor_h, offset_tensor_w = offsets.shape[2:]
    resize_factor_h, resize_factor_w = input_img_h / offset_tensor_h, input_img_w / offset_tensor_w

    offsets_y = offsets[:, :9]
    offsets_x = offsets[:, 9:]

    grid_y = np.arange(0, offset_tensor_h)
    grid_x = np.arange(0, offset_tensor_w)

    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    sampling_y = grid_y + offsets_y.detach().cpu().numpy()
    sampling_x = grid_x + offsets_x.detach().cpu().numpy()

    sampling_y *= resize_factor_h
    sampling_x *= resize_factor_w

    sampling_y = sampling_y[0]  # remove batch axis
    sampling_x = sampling_x[0]  # remove batch axis

    sampling_y = sampling_y.transpose(1, 2, 0)  # c, h, w -> h, w, c
    sampling_x = sampling_x.transpose(1, 2, 0)  # c, h, w -> h, w, c

    sampling_y = np.clip(sampling_y, 0, input_img_h)
    sampling_x = np.clip(sampling_x, 0, input_img_w)

    sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
    sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

    sampling_y = sampling_y[roi_point_y, roi_point_x]
    sampling_x = sampling_x[roi_point_y, roi_point_x]

    for y, x in zip(sampling_y, sampling_x):
        y = round(y)
        x = round(x)
        cv2.circle(image_bgr, center=(x, y), color=(0, 0, 255), radius=1, thickness=-1)

cv2.circle(image_bgr, center=(roi_point_x, roi_point_y), color=(0, 255, 0), radius=1, thickness=-1)
#image_bgr = cv2.resize(image_bgr, dsize=(224, 224))
#image_bgr=image_bgr.resize(224,224)
image_bgr = np.reshape(image_bgr,(224,224))
print(image_bgr)
# plt.imshow(image_bgr)
# plt.show()
# out = model(X)
# print(torch.argmax(out.squeeze(0)))
# with torch.no_grad():
#         image = cv2.imread("D:/data/vqa/coco/simple_vqa/Images/train2014/COCO_train2014_000000533158.jpg",cv2.IMREAD_COLOR)
#         input_img_h, input_img_w,_= image.shape
#         plt.imshow(image)
#
#         image_tensor = torch.from_numpy(image)*1.
#         image_tensor = image_tensor.view(1, 3, input_img_h, input_img_w)
#         image_tensor = T.Normalize((0.1307,), (0.3081,))(image_tensor)
#         image_tensor = image_tensor.to(device)
#
#         #image = np.repeat(image[..., np.newaxis], 3, axis=-1)
#         roi_y, roi_x = input_img_h // 2, input_img_w // 2
#         plot_offsets(image, save_output, roi_x=roi_x, roi_y=roi_y)
#
#         save_output.clear()
#         #image = cv2.resize(image, (3,224, 224))
#         plt.imshow(image)
#         #cv2.imshow("image", image)
#         plt.show()


