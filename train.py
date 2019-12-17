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
from model.detnet25 import ShuffleNetV2
from model.centernet import EfficientNet, efficientnet_b0
from model.mnet25 import CenterFace_MobileNet
from model.losses import CtdetLoss
import os
from torch.utils.data.distributed import DistributedSampler
from torch.optim.optimizer import Optimizer, required
import math
from dataset.dataset import CenterFaceData
from utils.utils import RAdam
from collections import OrderedDict
import eval_widerface
import itertools
def get_args():
    parser = argparse.ArgumentParser(description="Train program for centerface.")
    parser.add_argument('--data_path', type=str, default='../../data/data_wider', help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--verbose', type=int, default=50, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=10, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=1, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./out', help='Model save path')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    args = parser.parse_args()
    print(args)

    return args
def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False

def main():
    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path,'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_dir=log_path)

    data_path = args.data_path
    train_path = os.path.join(data_path,'train/label.txt')
    val_path = os.path.join(data_path,'val/label.txt')
    # dataset_train = TrainDataset(train_path,transform=transforms.Compose([RandomCroper(),RandomFlip()]))
    dataset_train = CenterFaceData(train_path)
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=args.batch, shuffle=True)

    dataset_val = CenterFaceData(val_path, split="test")
    dataloader_val = DataLoader(dataset_val, num_workers=2, batch_size=2 )#args.batch)
    
    total_batch = len(dataloader_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create torchvision model
    # centerface = ShuffleNetV2()#
    # centerface = CenterFace_MobileNet()
    centerface = efficientnet_b0(pretrained=True)
    
    resume = True
    centerface = centerface.to(device)
    from torchsummary import summary
    summary(centerface, (3,640, 640))
    base_lr = 1e-3
    if resume:
        state_dict = torch.load('out/model_epoch_10.pt')
        centerface.load_state_dict(state_dict)
        del state_dict
    # params = [
    #         {'params': itertools.chain(
    #             centerface.first_conv.parameters(),
    #             centerface.layer0.parameters(),
    #             centerface.layer1.parameters(),
    #             centerface.layer2.parameters(), 
    #             centerface.layer3.parameters(),
    #             centerface.layer4.parameters(),
    #             centerface.layer5.parameters(),
    #             centerface.layer6.parameters(),
    #             centerface.conv_last.parameters(),
    #             centerface.up1.parameters(),
    #             centerface.up2.parameters(),
    #             centerface.up3.parameters(),
    #             centerface.wh.parameters(),
    #             centerface.reg.parameters(),
    #             centerface.hm.parameters()
    #         ), 'lr': base_lr},
    #         {'params': centerface.lm.parameters()}]
    optimizer = optim.SGD(centerface.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    if resume:
        state_dict = torch.load('out/optimizer_epoch_40.pt')
        optimizer.load_state_dict(state_dict)
        del state_dict
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [30, 90, 140], gamma=0.1)

    print('Start to train.')

    epoch_loss = []
    iteration = 0
    centerface.train()

    loss_fnc = CtdetLoss(device)
    for epoch in range(args.epochs):
        exp_lr_scheduler.step()
        if resume:
            if epoch < 40:
                continue
        # # Training
        for iter_num, data in enumerate(dataloader_train):
            for k in data.keys():
                data[k] = data[k].cuda()
            optimizer.zero_grad()
            output = centerface(data['input'].float())
            loss, loss_sta = loss_fnc(output, data)
            # print(loss)
            loss.backward()
            optimizer.step()
            if iter_num % args.verbose == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, iter_num, total_batch)
                table_data = [
                    ['loss name','value'],
                    ['total_loss',str(loss.item())],
                    ['hm_loss', str(loss_sta['hm_loss'].item())],
                    ['wh_loss', str(loss_sta['wh_loss'].item())],
                    ['off_loss', str(loss_sta['off_loss'].item())],

                    ['landmarks',str(loss_sta['lm_loss'].item())]
                    ]
                table = AsciiTable(table_data)
                log_str +=table.table
                print(log_str)
                # write the log to tensorboard
                writer.add_scalar('losses:',loss.item(),iteration*args.verbose)
                writer.add_scalar('hm_loss:', loss_sta['hm_loss'].item(),iteration*args.verbose)
                writer.add_scalar('wh loss:',loss_sta['wh_loss'].item(),iteration*args.verbose)
                writer.add_scalar('off loss:',loss_sta['off_loss'].item(),iteration*args.verbose)
                iteration +=1

        # if epoch % args.eval_step == 0:
        #     print('-------- centerface Pytorch --------')
        #     print ('Evaluating epoch {}'.format(epoch))
        #     recall, precision = eval_widerface.evaluate(dataloader_val,centerface)
        #     print('Recall:',recall)
        #     print('Precision:',precision)

        #     writer.add_scalar('Recall:', recall, epoch*args.eval_step)
        #     writer.add_scalar('Precision:', precision, epoch*args.eval_step)

        # Save model
        if (epoch + 1) % args.save_step == 0:
            torch.save(optimizer.state_dict(), args.save_path + '/optimizer_epoch_{}.pt'.format(epoch + 1))
            torch.save(centerface.state_dict(), args.save_path + '/model_epoch_{}.pt'.format(epoch + 1))

    writer.close()


if __name__=='__main__':
    main()