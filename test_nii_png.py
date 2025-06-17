import imageio
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk
import os

import utils.metrics as m
from models.EDAANet import EDAANet

from utils.preprocessing_utils import sitk2slices, sitk2labels
from utils.surface import Surface



if __name__ == '__main__':
    LITS_fixed_data_path = '/root/autodl-tmp/test_fixed_nii/'
    model_path = '/EDAANet_ceW_on_LITS_dataset_iter_190000.pth'
    prediction_path = './results/EDAANet_ceW_PNG/'
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

    
    for i in range(31):
        print(i)

        ct = sitk.ReadImage(LITS_fixed_data_path + 'volume-' + str(i) + '.nii', sitk.sitkInt16)
        vxlspacing = ct.GetSpacing()
        ct_array = np.rot90(sitk.GetArrayFromImage(ct))
        ct_array = np.rot90(ct_array)

        seg = sitk.ReadImage(LITS_fixed_data_path + 'segmentation-' + str(i) + '.nii', sitk.sitkInt16)
        seg_array = np.rot90(sitk.GetArrayFromImage(seg))
        seg_array = np.rot90(seg_array)

        slices_in_order = sitk2slices(ct_array, 0, 400)
        labels_in_order = sitk2labels(seg_array)

        predictions_in_order = []

        slice_idx_list = []
        dice_list = []
        iou_list = []
        rvd_list = []
        

        for idx, (slice, label) in enumerate(zip(slices_in_order, labels_in_order)):
            print(f'Processing slice {idx}')
            slice_idx_list.append(idx)

            slice = torch.from_numpy(slice).float() / 255.
            
            output = model(slice.unsqueeze(0).unsqueeze(0).to(device[0]))
            prediction = sm(output)
            _, prediction = torch.max(prediction, dim=1)
            prediction = prediction.squeeze(0).cpu().detach().numpy().astype(np.uint8)
            predictions_in_order.append(prediction)

           
            # Save visualization of segmentation with label multiplied by factor 255
            vis_label = prediction * factor
            vis_label_path = os.path.join(prediction_path, f'{i}_{idx:04d}.png')
            imageio.imwrite(vis_label_path, vis_label)
            
           
            v_prediction = prediction.astype(np.uint)
            v_label = label.astype(np.uint)

            # Calculate evaluation metrics
            if np.count_nonzero(v_prediction) == 0 and np.count_nonzero(v_label) == 0:
                dice = 0.0
                iou = 0.0
                rvd = 0.0
               
            else:
                dice = m.dc(v_prediction, v_label)
                iou = m.jc(v_prediction, v_label) if np.count_nonzero(v_prediction) > 0 and np.count_nonzero(v_label) > 0 else 0
                # 如果至少有一个为 0，则将结果赋值为空(np.nan)
                rvd = m.ravd(v_prediction, v_label) if np.count_nonzero(v_prediction) > 0 and np.count_nonzero(v_label) > 0 else np.nan
                

            dice_list.append(dice)
            iou_list.append(iou)
            rvd_list.append(rvd)
                       
        metric_data = {
            'slice_idx': slice_idx_list,
            'dice': dice_list,
            'iou': iou_list,
            'rvd': rvd_list
        }
        csv_data = pd.DataFrame(metric_data)
        csv_file_path = os.path.join(prediction_path, f'metrics_volume_{i}.csv')
        csv_data.to_csv(csv_file_path, index=False)
