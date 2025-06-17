import os
import shutil
import nibabel as nib
import glob
import imageio
import numpy as np
import multiprocessing
from utils.preprocessing_utils import sitk2slices, sitk2labels
import SimpleITK as sitk


if __name__ == '__main__':
    fixed_raw_path = '/DataSets/LiTS/EDAA-dataset/raw/'
    fixed_label_path = '/DataSets/LiTS/EDAA-dataset/label/'
    tr_path = '/DataSets/LiTS/EDAA-dataset/save_first_rot_equal/tr/'   # 训练train  31-131
    ts_path = '/DataSets/LiTS/EDAA-dataset/save_first_rot_equal/ts/'   # 测试test 0-30
    raw_path = 'raw/'
    label_path = 'label/'

    for i in range(31):
        print(i)
        ct = sitk.ReadImage(fixed_raw_path + 'volume-' + str(i) + '.nii', sitk.sitkInt16)
        ct_array = np.rot90(sitk.GetArrayFromImage(ct))  # x与z轴发生对调
        ct_array = np.rot90(ct_array)     # numpy旋转90读

        seg = sitk.ReadImage(fixed_label_path + 'segmentation-' + str(i) + '.nii', sitk.sitkInt16)
        seg_array = np.rot90(sitk.GetArrayFromImage(seg))
        seg_array = np.rot90(seg_array)

        slices_in_order = sitk2slices(ct_array, 0, 400)  # 窗口范围可修改
        labels_in_order = sitk2labels(seg_array)
        for n in range(len(slices_in_order)):
            imageio.imwrite(ts_path+raw_path+str(i)+'_'+str(n).zfill(4)+'.png', slices_in_order[n].astype(np.uint8))
            imageio.imwrite(ts_path+label_path+str(i)+'_'+str(n).zfill(4)+'.png', labels_in_order[n].astype(np.uint8))

    for i in range(31, 131):
        print(i)
        ct = sitk.ReadImage(fixed_raw_path + 'volume-' + str(i) + '.nii', sitk.sitkInt16)
        ct_array = np.rot90(sitk.GetArrayFromImage(ct))
        ct_array = np.rot90(ct_array)

        seg = sitk.ReadImage(fixed_label_path + 'segmentation-' + str(i) + '.nii', sitk.sitkInt16)
        seg_array = np.rot90(sitk.GetArrayFromImage(seg))
        seg_array = np.rot90(seg_array)

        slices_in_order = sitk2slices(ct_array, 0, 400)
        labels_in_order = sitk2labels(seg_array)
        for n in range(len(slices_in_order)):
            imageio.imwrite(tr_path+raw_path+str(i)+'_'+str(n).zfill(4)+'.png', slices_in_order[n].astype(np.uint8))
            imageio.imwrite(tr_path+label_path+str(i)+'_'+str(n).zfill(4)+'.png', labels_in_order[n].astype(np.uint8))