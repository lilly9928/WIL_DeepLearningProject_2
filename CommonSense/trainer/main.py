"""
Training script. Should be pretty adaptable to whatever.
"""

import sys
sys.path.append("/home/user2/code/WIL_DeepLearningProject_2/CommonSense")

import argparse
import os
import shutil

import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
from torchvision.transforms import transforms


from dataloaders.vcr.vcr import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import CommonSense.base.r2c.models

from transformers import BlipProcessor, BlipForConditionalGeneration


# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model


#################################
#################################
######## Data loading stuff
#################################
#################################

os.environ["CUDA_VISIBLE_DEVICES"]='3,4'

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    default='/home/user2/code/WIL_DeepLearningProject_2/CommonSense/dataloaders/vcr/default.json',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    default='aves/flagship_answer',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

args = parser.parse_args()
params = Params.from_file(args.params)

#VCR data
train, val, test = VCR.splits(mode='rationale' if args.rationale else 'answer',
                              embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True))

#checking GPU , CPU
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

#data to gpu
def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True)
    return td

##cal worker
num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)

#VCR data_loader
loader_params = {'batch_size': 32 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
test_loader = VCRLoader.from_dataset(test, **loader_params)

#IDONTKNOW
ARGS_RESET_EVERY = 100
print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)

#MODEL
model = Model.from_params(vocab=train.vocab, params=params['model'])

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

#Model Freeze
for submodule in model.detector.backbone.modules():
    if isinstance(submodule, BatchNorm2d):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False

#GPU 여러개 쓰기 or 하나 쓰기
model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()

# optim, sceduler
optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad],
                                  params['trainer']['optimizer'])

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

#batch pth check
if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)
else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)

#param print.. start!
param_shapes = print_para(model)

num_batches = 0

def tensor_to_pil_image(tensor):
    # 텐서를 numpy 배열로 변환 (0과 1 사이 값으로 정규화)
    image_np = tensor.numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    # PIL 이미지로 변환
    image_pil = Image.fromarray(image_np)
    return image_pil
#train
for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
    train_results = []
    norms = []
    model.train()
    for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        batch = _to_gpu(batch) #batch dat to gpu

        images = batch['images'].to("cuda")

        captions = []
        for i in range(len(images)):
            # 크롭된 이미지를 캡션 생성 모델에 입력하여 캡션 생성

            inputs = processor(tensor_to_pil_image(images[i]), return_tensors="pt").to("cuda")
            captions.append(caption_model.generate(**inputs))
            generated_captions = processor.batch_decode(captions, skip_special_tokens=True)


        optimizer.zero_grad()
#         output_dict = model(**batch) ## modify !
#
#         loss = output_dict['loss'].mean() + output_dict['cnn_regularization_loss'].mean()
#         loss.backward()
#
#         num_batches += 1
#         if scheduler:
#             scheduler.step_batch(num_batches)
#
#         norms.append(
#             clip_grad_norm(model.named_parameters(), max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
#         )
#         optimizer.step()
#
#         train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
#                                         'crl': output_dict['cnn_regularization_loss'].mean().item(),
#                                         'accuracy': (model.module if NUM_GPUS > 1 else model).get_metrics(
#                                             reset=(b % ARGS_RESET_EVERY) == 0)[
#                                             'accuracy'],
#                                         'sec_per_batch': time_per_batch,
#                                         'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
#                                         }))
#         if b % ARGS_RESET_EVERY == 0 and b > 0:
#             norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
#                 param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)
#
#             print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
#                 epoch_num, b, len(train_loader),
#                 norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
#                 pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
#             ), flush=True)
#
#     print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))
#     val_probs = []
#     val_labels = []
#     val_loss_sum = 0.0
#     model.eval()
#     for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
#         with torch.no_grad():
#             batch = _to_gpu(batch)
#             output_dict = model(**batch)
#             val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
#             val_labels.append(batch['label'].detach().cpu().numpy())
#             val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
#     val_labels = np.concatenate(val_labels, 0)
#     val_probs = np.concatenate(val_probs, 0)
#     val_loss_avg = val_loss_sum / val_labels.shape[0]
#
#     val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
#     if scheduler:
#         scheduler.step(val_metric_per_epoch[-1], epoch_num)
#
#     print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
#           flush=True)
#     if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
#         print("Stopping at epoch {:2d}".format(epoch_num))
#         break
#     save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
#                     is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))
#
# print("STOPPING. now running the best model on the validation set", flush=True)
# # Load best
# restore_best_checkpoint(model, args.folder)
# model.eval()
# val_probs = []
# val_labels = []
# for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
#     with torch.no_grad():
#         batch = _to_gpu(batch)
#         output_dict = model(**batch)
#         val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
#         val_labels.append(batch['label'].detach().cpu().numpy())
# val_labels = np.concatenate(val_labels, 0)
# val_probs = np.concatenate(val_probs, 0)
# acc = float(np.mean(val_labels == val_probs.argmax(1)))
# print("Final val accuracy is {:.3f}".format(acc))
# np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)
