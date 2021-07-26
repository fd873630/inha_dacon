import time
import random
import datetime

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from dataset import MS1MDataset
from model import ResNet, IRBlock, ResNet_Final


def model_train(model, train_loader, optimizer, criterion, scheduler, total_step, device):
    model.train()

    running_loss = 0 
    running_corrects = 0
    total_num = 0

    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        total_step += 1
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        inputs, targets, inputs_lengths, targets_lengths = data
        total_num += sum(targets_lengths)

        output, features = model(inputs) 
               
        _, preds = torch.max(output, 1)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 500 == 0:
            lr = scheduler.get_lr()[0]
            print('{} lr: {:7f}, train_batch: {:4d}/{:4d}, loss: {:.4f}, time: {:.2f}'
                  .format(datetime.datetime.now(), lr, i, total_batch_num, loss.item(), time.time() - start_time))
            start_time = time.time()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / total_batch_num
    epoch_acc = running_corrects.double() / total_batch_num
    
    return epoch_loss, epoch_acc


def main():
    seed_num = 123456
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    gpu_num = torch.cuda.device_count()
    #-------------------------- Model Initialize --------------------------
    #Prediction Network
    num_classes = 86876 

    res_model = ResNet(IRBlock, [3, 4, 6, 3], use_se=False, im_size=112)
    net = nn.Sequential(nn.Linear(512, num_classes))

    model = ResNet_Final(res_model, net)

    model = model.to(device)
    #-------------------------- Loss & Optimizer --------------------------
    criterion = nn.CrossEntropyLoss()
    
    if gpu_num > 1:
        print("DataParallel mode")
        model = nn.DataParallel(model).to(device)
        optimizer = optim.Adam(model.module.parameters(), lr=0.001)

        lr_lambda = lambda x: x/1000 if x < 1000 else (1 if x < 20000 else (x / 20000) ** -0.5 )
        scheduler = LambdaLR(optimizer, lr_lambda)

    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        lr_lambda = lambda x: x/1000 if x < 1000 else (1 if x < 20000 else (x / 20000) ** -0.5 )
        scheduler = LambdaLR(optimizer, lr_lambda)
    
    #-------------------------- Data load --------------------------
    #train dataset
    train_dataset = MS1MDataset('train', "/home/jhjeong/jiho_deep/inha_dacon/inha_data/ID_List.txt")
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=gpu_num * 4)

    print(" ")
    print("las_only 를 학습합니다.")
    print(" ")

    pre_test_cer = 100000
    pre_test_loss = 100000
    total_step = 0
    for epoch in range(0, 100):
        print('{} 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()
        epoch_loss, epoch_acc = model_train(model, dataloader, optimizer, criterion, scheduler, total_step, device)
        train_total_time = time.time() - train_time
        print('{} Epoch {} (Training) Loss {:.4f}, ACC {:.2f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, epoch_loss, epoch_acc, train_total_time))
        
        '''
        #eval은 나중에
        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        val_loss, test_cer = model_eval(las_model, val_loader, las_criterion, device)
        eval_total_time = time.time() - eval_time
        print('{} Epoch {} (val) Loss {:.4f}, CER {:.2f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, val_loss, test_cer, eval_total_time))
        '''

        '''
        #모델 저장도 나중에
        if pre_test_loss > val_loss:
            print("best model을 저장하였습니다.")
            if gpu_num > 1:
                torch.save(model.module.state_dict(), "./pth_file/model_best.pth")
            else:
                torch.save(model.state_dict(), "./pth_file/model_best.pth")
            pre_test_loss = val_loss

        if gpu_num > 1:
            torch.save(model.module.state_dict(), "./pth_file/model_" + str(epoch) + ".pth")
        else:
            torch.save(model.state_dict(), "./pth_file/model_" + str(epoch) + ".pth")
        '''        


if __name__ == '__main__':
    main()