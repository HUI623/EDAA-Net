import argparse
from utils.trainer import trainer
from utils.dataset import LITS_dataset, make_dataloaders
import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn import DataParallel
from models.DEANet import DEANet
import time
import datetime



if __name__ == '__main__':
    device = [torch.device('cuda:0'), torch.device('cuda:1')]  # 使用 GPU 0 和 GPU 1

    LEARNING_RATE = 1e-3
    LR_DECAY_STEP = 2
    LR_DECAY_FACTOR = 0.5
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 4
    MAX_EPOCHS = 60  # 30  60
    
  
    MODEL = DEANet(1, 2).to(device[0])   # 输入n_channels, 类别n_classes
    
    
    # 指定要用到的设备
    MODEL = torch.nn.DataParallel(MODEL, device_ids=device)    # 指定要用到的设备

    
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    
    CRITERION = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.75,1])).float(),size_average=True).to(device[0]) 


    tr_path_raw = '/root/autodl-tmp/liver/tr/raw'
    tr_path_label = '/root/autodl-tmp/liver/tr/label'
    ts_path_raw = '/root/autodl-tmp/liver/ts/raw'
    ts_path_label = '/root/autodl-tmp/liver/ts/label'
    
    
    checkpoints_dir = './checkpoints_EDAA_ceW'
    checkpoint_frequency = 10000
    dataloaders = make_dataloaders(tr_path_raw, tr_path_label, ts_path_raw, ts_path_label, BATCH_SIZE, n_workers=24)  
    
    comment = 'EDAA_ceW_on_LITS_dataset_'
    verbose_train = 1
    verbose_val = 5000

    trainer = trainer(MODEL, OPTIMIZER, LR_SCHEDULER, CRITERION, dataloaders, comment, verbose_train, verbose_val, checkpoint_frequency, MAX_EPOCHS, checkpoint_dir=checkpoints_dir, device=device[0])
       
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    runtime = end_time - start_time
    
    print('Training runtime: %.2f h' % (runtime/3600))    



