import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
import math
from collections import OrderedDict
import copy
import time

import data.my_utils_2 as data_utils # changed Dataloader definitions!
import models.loss as loss
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem

import argparse

print("--------------PyTorch VERSION:", torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("..............device", device)

parser = argparse.ArgumentParser(description="MemoryNormality")
parser.add_argument('--gpus', nargs='+', type=str, default=0, help='gpus') # 
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training') # low for testing
parser.add_argument('--val_epoch', type=int, default=5, help='evaluate the model every %d epoch')
parser.add_argument('--h', type=int, default=128, help='height of input images') # reduced from 256x256
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate') # was 2e-4 # TODO: check how to modify according to batch size
parser.add_argument('--t_length', type=int, default=16, help='length of the frame sequences')
parser.add_argument('--ModelName', help='AE/MemAE', type=str, default='MemAE')
parser.add_argument('--ModelSetting', help='Conv3D/Conv3DSpar',type=str, default='Conv3DSpar')  # give the layer details later
parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=2000)
parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0.0002)
parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.0025)
parser.add_argument('--Suffix', help='Suffix', type=str, default='Non')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader') # use 8 later
parser.add_argument('--num_workers_test', type=int, default=8, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='Cataract', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='/local/scratch/hendrik/memAE/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='/local/scratch/hendrik/memAE/Cataract/checkpoints', help='directory of log')
parser.add_argument('--version', type=int, default=5, help='experiment version') # TODO: modify this after every run

args = parser.parse_args()
torch.manual_seed(2020)
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

def arrange_image(im_input):
    im_input = np.transpose(im_input, (0, 2, 1, 3, 4))
    b, t, ch, h, w = im_input.shape
    im_input = np.reshape(im_input, [b * t, ch, h, w])
    return im_input

#train_folder, test_folder = data_utils.give_data_folder(args.dataset_type, 
#                                                        args.dataset_path)
#train_folder = args.dataset_path + args.dataset_type + '/frames/training/'
#test_folder = args.dataset_path + args.dataset_type + '/frames/testing/' # only used for validation
train_folder = test_folder = "/local/scratch/Cataract-1K-Full-Videos/"
print("The training path", train_folder)
print("The testing path", test_folder)

frame_trans = data_utils.give_frame_trans([args.h, args.w]) # function returns a set of transformation instructions that will resize (to 256x256), grayscale and normalise frames upon which "frame_trans" is applied.
# At this step, nothing but the definition of the transformation happens yet. The transformation operation happens in data_utils.DataLoader(..., frame_trans, ..., ...) just below. 

train_csv = test_csv = "/local/scratch/hendrik/tiny_annotations.csv" # change path later, just for debugging - 13secs now
# test_csv = "/local/scratch/hendrik/tiny_annotations.csv" # obviously, this should be different later

train_dataset = test_dataset = data_utils.MyDataset(train_folder, frame_trans, train_csv, time_step=args.t_length - 1, num_pred=1) # calls data_utils.setup() (if folder is a str) or data_utils.setup_multiple() (if folder is a list) in init.
# test_dataset = data_utils.MyDataset(test_folder, frame_trans, test_csv, time_step=args.t_length - 1, num_pred=1)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True) # try with pin_memory=True
test_batch = data.DataLoader(test_dataset, batch_size = args.batch_size, 
                             shuffle=False, num_workers=args.num_workers, drop_last=True, pin_memory=True) # same here

print("Training data shape", len(train_batch)) # should be secs * fps / frame skip -> here with 18000 frames and frame_skip = 16 -> ~4500
print("Validation data shape", len(test_batch)) # TODO: check case_2000 fps

# Model setting
if (args.ModelName == 'AE'):
    model = AutoEncoderCov3D(args.c)
elif(args.ModelName=='MemAE'):
    model = AutoEncoderCov3DMem(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
else:
    model = []
    print('Wrong Name.')

model.to(device) # should be cuda here
parameter_list = [p for p in model.parameters() if p.requires_grad] 

for name, p in model.named_parameters():
    if not p.requires_grad:
        print("---------NO GRADIENT-----", name)
        
optimizer = torch.optim.Adam(parameter_list, lr = args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.2)  

#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)

# Report the training process
log_dir = os.path.join(args.exp_dir, 'lr_%.5f_entropyloss_%.5f_version_%d' % (
    args.lr, args.EntropyLossWeight, args.version))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

for arg in vars(args):
    print(arg, getattr(args, arg))

train_writer = SummaryWriter(log_dir=log_dir)

# warmup
#print("training starts")
#model.train()
#with torch.no_grad(): # takes 30mins with one 13s clip....
#    #print("now the batches", flush=True)
#    for batch_idx, frame in enumerate(train_batch): # this section seems to be reached, but no print outputs earlier??
#        # print(batch_idx, frame) # also reached, but no prints
#        frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
#        frame = frame.permute(0, 2, 1, 3, 4)
#        frame = frame.to(device)
#        model_output = model(frame)

# Training
for epoch in range(args.epochs):
    #print(epoch)
    model.train()
    tr_re_loss, tr_mem_loss, tr_tot = 0.0, 0.0, 0.0
    progress_bar = tqdm(train_batch)

    for batch_idx, frame in enumerate(progress_bar):
        #print("progress yay")
        progress_bar.update()
        frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
        frame = frame.permute(0, 2, 1, 3, 4)
        frame = frame.to(device) # cuda
        optimizer.zero_grad()

        model_output = model(frame)
        recons, attr = model_output['output'], model_output['att']
        re_loss = loss.get_reconstruction_loss(frame, recons, mean=0.5, std=0.5)
        mem_loss = loss.get_memory_loss(attr)
        tot_loss = re_loss + mem_loss * args.EntropyLossWeight
        tr_re_loss += re_loss.data.item()
        tr_mem_loss += mem_loss.data.item()
        tr_tot += tot_loss.data.item()
        
        tot_loss.backward()
        optimizer.step()
        
    train_writer.add_scalar("model/train-recons-loss", tr_re_loss/len(train_batch), epoch)
    train_writer.add_scalar("model/train-memory-sparse", tr_mem_loss/len(train_batch), epoch)
    train_writer.add_scalar("model/train-total-loss", tr_tot/len(train_batch), epoch)
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    train_writer.add_scalar('learning_rate', current_lr, epoch)
    
    if epoch % args.val_epoch == 0:
        model.eval()
        re_loss_val, mem_loss_val = 0.0, 0.0
        for batch_idx, frame in enumerate(test_batch):
            frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
            frame = frame.permute(0, 2, 1, 3, 4)
            frame = frame.to(device)
            model_output = model(frame)
            recons, attr = model_output['output'], model_output['att']
            re_loss = loss.get_reconstruction_loss(frame, recons, mean=0.5, std=0.5)
            mem_loss = loss.get_memory_loss(attr)
            re_loss_val += re_loss.data.item()
            mem_loss_val += mem_loss.data.item()
            if batch_idx == len(test_batch) - 1:
                _input_npy = frame.detach().cpu().numpy()
                _input_npy = _input_npy * 0.5 + 0.5
                _recons_npy = recons.detach().cpu().numpy()
                _recons_npy = _recons_npy * 0.5 + 0.5  # [batch_size, ch, time, imh, imw]
                train_writer.add_images("image/input_image", arrange_image(_input_npy), epoch)
                train_writer.add_images("image/reconstruction", arrange_image(_recons_npy), epoch)
        train_writer.add_scalar("model/val-recons-loss", re_loss_val / len(test_batch), epoch)
        train_writer.add_scalar("model/val-memory-sparse", mem_loss_val / len(test_batch), epoch)
        print("epoch %d" % epoch, "recons loss training %.4f validation %.4f" % (tr_re_loss, re_loss_val), 
              "memory sparsity training %.4f validation %.4f" % (tr_mem_loss, mem_loss_val))

    if epoch >= args.epochs - 50:
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), log_dir + "/model-{:04d}.pt".format(epoch))

sys.stdout = orig_stdout
f.close()




