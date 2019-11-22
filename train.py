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
from model.losses import CtdetLoss
import os
from torch.utils.data.distributed import DistributedSampler
from torch.optim.optimizer import Optimizer, required
import math
from dataset.dataset import CenterFaceData
from utils.utils import RAdam

def get_args():
    parser = argparse.ArgumentParser(description="Train program for centerface.")
    parser.add_argument('--data_path', type=str, default='../../data/data_wider', help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--verbose', type=int, default=50, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=10, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=3, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./out', help='Model save path')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    args = parser.parse_args()
    print(args)

    return args


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
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=args.batch,shuffle=True)

    # dataset_val = CenterFaceData(val_path, split="test")
    # dataloader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch)
    
    total_batch = len(dataloader_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create torchvision model
    centerface = ShuffleNetV2()
    # checkpoint = torch.load('%s/model_epoch_250.pt'%args.save_path)
    # centerface.load_state_dict(checkpoint)
    centerface = centerface.to(device)

    centerface.training = True
    centerface = centerface.cuda()
    centerface = torch.nn.DataParallel(centerface).cuda()
    centerface.training = True

    optimizer = RAdam(centerface.parameters(), lr=1.25e-3)
    # optimizer = optim.SGD(centerface.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [60, 90, 140], gamma=0.1)

    print('Start to train.')

    epoch_loss = []
    iteration = 0

    loss_fnc = CtdetLoss(device)
    for epoch in range(args.epochs):
        centerface.train()
        exp_lr_scheduler.step()

        # Training
        for iter_num, data in enumerate(dataloader_train):
            for k in data.keys():
                data[k] = data[k].cuda()
            optimizer.zero_grad()
            output = centerface(data['input'].cuda().float())
            loss, loss_sta = loss_fnc(output, data)
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

        # Eval
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
            torch.save(centerface.state_dict(), args.save_path + '/model_epoch_{}.pt'.format(epoch + 1))

    writer.close()


if __name__=='__main__':
    main()