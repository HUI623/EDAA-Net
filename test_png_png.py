import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import imageio
import imageio.v2 as imageio
import cv2  # 添加用于图像缩放的库

import utils.metrics as m
from models.EDAANet import EDAANet



def normalize_image(image):
    return image / 255.

if __name__ == '__main__':
    image_folder_path = '/root/autodl-tmp/test3Dircadb/raw/'
    label_folder_path = '/root/autodl-tmp/test3Dircadb/label/'
   
    
    model_path = '/checkpoints_EDAA_ceW/EDAA_ceW_on_LITS_dataset_iter_190000.pth'
    prediction_path = './results/test3Dircadb_EDAA_ceW_PNG/'
    
    metrics_path = './metrics/'
    factor = 255   # 可视化颜色 [0-255]
    
    
    # 双卡
    device = [torch.device('cuda:0'), torch.device('cuda:1')]
    
    model = EDAANet(1, 2).to(device[0])
    model = torch.nn.DataParallel(model, device_ids=device)
    model.load_state_dict(torch.load(model_path, map_location=device[0]))
    
    model.eval()
    sm = nn.Softmax(dim=1)

    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    # 获取图像和标签文件列表
    image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_folder_path) if f.endswith('.png')])

    all_metrics = []

    for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        # print(f"Processing image {image_file} and label {label_file}")

        image_path = os.path.join(image_folder_path, image_file)
        label_path = os.path.join(label_folder_path, label_file)

        # 读取 PNG 图像
        ct_array = imageio.imread(image_path)
        seg_array = imageio.imread(label_path)

        # 调整图像大小512*512到256x256
        ct_array = cv2.resize(ct_array, (256, 256), interpolation=cv2.INTER_AREA)
        seg_array = cv2.resize(seg_array, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 将标签值从255归一化到1
        seg_array = seg_array / 255

        # 预测
        ct_array = normalize_image(ct_array)
        
        ct_tensor = torch.from_numpy(ct_array).float().unsqueeze(0).unsqueeze(0).to(device[0])     # 双卡
        with torch.no_grad():
            output = model(ct_tensor)
        prediction = sm(output)
        _, prediction = torch.max(prediction, dim=1)
        prediction = prediction.squeeze(0).cpu().detach().numpy().astype(np.uint8)

              
        # 可视化分割结果
        vis_label = prediction * factor
        vis_label_path = os.path.join(prediction_path, f'{label_file}')
        imageio.imwrite(vis_label_path, vis_label)
             

        # 计算评价指标
        if np.count_nonzero(prediction) == 0 and np.count_nonzero(seg_array) == 0:
            dice = 0.0
            iou = 0.0
            rvd = 0.0
            
        else:
            dice = m.dc(prediction, seg_array)
            iou = m.jc(prediction, seg_array) if np.count_nonzero(prediction) > 0 and np.count_nonzero(seg_array) > 0 else 0
            rvd = m.ravd(prediction, seg_array) if np.count_nonzero(prediction) > 0 and np.count_nonzero(seg_array) > 0 else np.nan
            

        metrics = {
            'image': image_file,
            'dice': dice,
            'iou': iou,
            'rvd': rvd
            
        }

        all_metrics.append(metrics)
        
        # print(f"Finished processing {image_file} and {label_file}")

    # 保存所有图像的评价指标到一个 CSV 文件
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(metrics_path, 'test3Dircadb_EDAA_ceW_PNG.csv'), index=False)
    
    print("--------------------Finished saving all metrics to CSV file--------------------")

    