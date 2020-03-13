import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from terminaltables import AsciiTable, DoubleTable, SingleTable
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import torch.distributed as dist
# import eval_widerface
import torchvision
del.centernet import efficientnet_b0
from model.losses import CtdetLoss
import os
from torch.utils.data.distributed import DistributedSampler
from torch.optim.optimizer import Optimizer, required
import math
from dataset.ext_dataset import CenterFaceData
from utils.utils import RAdam
from collections import OrderedDict
# import eval_widerface
import itertools
def get_args():
    parser = argparse.ArgumentParser(description="Train program for centerface.")
    parser.add_argument('--data_path', type=str, default='../../data/data_wider', help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--verbose', type=int, default=50, help='Log verbose')
    parser.add_argument('--lr', type=float, default=0.015, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=5, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=1, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./weight', help='Model save path')
    parser.add_argument('--cuda', help='Device ', type=int , default=None)
    args = parser.parse_args()
    print(args)

    return args
def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False

def main(resume = 40):
    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path,'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    data_path = args.data_path
    train_path = os.path.join(data_path,'train/label.txt')
    val_path = os.path.join(data_path,'val/label.txt')
    dataset_train = CenterFaceData(train_path)
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=args.batch, shuffle=True)

    dataset_val = CenterFaceData(val_path, split="test")
    dataloader_val = DataLoader(dataset_val, num_workers=2, batch_size=1 )#args.batch)
    
    total_batch = len(dataloader_train)
    cuda =  torch.cuda.is_available() 
    device = torch.device("cuda" if cuda else "cpu")

    centerface = efficientnet_b0(True)  
    centerface = centerface.to(device)

    if resume:
        state_dict = torch.load('weight/model_epoch_%s.pt'%resume)
        centerface.load_state_dict(state_dict)
        del state_dict
    params = [
            {'params': itertools.chain(
                centerface.first_conv.parameters(),
                centerface.layer0.parameters(), 
                centerface.layer1.parameters(),
                centerface.layer2.parameters(),
                centerface.layer3.parameters(),
                centerface.layer4.parameters(),
                centerface.layer5.parameters(),
                centerface.layer6.parameters()
            ), 'lr': 0.00001},
            {'params': itertools.chain(
                centerface.up1.parameters(),
                centerface.up2.parameters(),
                centerface.up3.parameters(),
                centerface.conv_last.parameters(),
                centerface.hm.parameters(),
                centerface.wh.parameters(),
                centerface.reg.parameters(),
                
            ), 
            'lr': 0.0001},

            {'params': itertools.chain(
                centerface.lm.parameters(),
            ),
            'lr': 0.001}
        ]
    optimizer = optim.Adam(params, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if resume:
        state_dict = torch.load('weight/optimizer_epoch_%s.pt'%resume)
        optimizer.load_state_dict(state_dict)
        del state_dict
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [25, 75, 120], gamma=0.099)

    print('Start to train.')

    epoch_loss = []
    iteration = 0
    centerface.train()

    loss_fnc = CtdetLoss(device)
    for epoch in range(args.epochs):
        exp_lr_scheduler.step()
        if resume:
            if epoch < resume:
                continue
        if epoch>70:
            hm_w, wh_w, off_w, lm_w = 1., 1, 1., 5.
        else:
            hm_w, wh_w, off_w, lm_w = 1., .1, 1., 1.
           
        # # Training
        centerface.train()
        for iter_num, data in enumerate(dataloader_train):
            if cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            optimizer.zero_grad()
            output = centerface(data['input'].float())

            loss, loss_sta = loss_fnc(output, data, hm_w = hm_w, wh_w = wh_w, off_w = off_w, lm_w = lm_w)
            # print(loss)
            loss.backward()
            optimizer.step()
            if iter_num % args.verbose == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, iter_num, total_batch)
                parts = ['loss','hm_loss', 'wh_loss', 'off_loss', 'lm_loss']
                table_data = []
                for p in parts:
                    table_data.append(['{}'.format(p), loss_sta['{}'.format(p)].item()])

                table = AsciiTable(table_data)
                log_str +=table.table
                print(log_str)

                iteration +=1

        if epoch % args.eval_step == 0:
            print('-------- centerface Pytorch --------')
            print ('Evaluating epoch {}'.format(epoch))
            recall, precision = eval_widerface.evaluate(dataloader_val, centerface)
            print('Recall:',recall)
            print('Precision:',precision)

            writer.add_scalar('Recall:', recall, epoch*args.eval_step)
            writer.add_scalar('Precision:', precision, epoch*args.eval_step)

        # Save model
        if (epoch + 1) % args.save_step == 0:
            torch.save(optimizer.state_dict(), args.save_path + '/optimizer_epoch_{}.pt'.format(epoch + 1))
            torch.save(centerface.state_dict(), args.save_path + '/model_epoch_{}.pt'.format(epoch + 1))

    writer.close()

if __name__=='__main__':
    main()