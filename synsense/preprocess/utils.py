"""
This file contains different useful functions to make your preprocessing 
much easier.
"""

import os
import re
import shutil
import numpy as np
from typing import Tuple
from .preprocessors import PreprocessStream
from .parallelProcessing import load_single_sample

def rename_files(folder_absPath: str, pose_mapping: dict) -> None:
    """
    在 folder_absPath 中查找所有符合 taps_trial_x_pose_y_events_on.npy 格式的文件,
    根据pose_mapping映射表改名为:
    将 pose_0 重命名为 Wool_trial_x_on.npy
    将 pose_1 重命名为 Canvas_trial_x_on.npy

    User can change this function to suit their own need!

    e.g.:

    pose_mapping = {
        '0': 'Wool',
        '1': 'Canvas'
    }
    """
    # 列出文件夹下所有文件
    file_list = os.listdir(folder_absPath)
    # 正则，用于匹配 "taps_trial_<数字>_pose_<数字>_events_on.npy"
    pattern = re.compile(r'^taps_trial_(\d+)_pose_(\d+)_events_on\.npy$')
    for old_name in file_list:
        # 构建旧文件的完整路径
        old_path = os.path.join(folder_absPath, old_name)
        # 若是文件夹则跳过, 检查扩展名 
        if os.path.isdir(old_path) or not old_name.endswith('.npy'):
            continue
        # 用正则从开头开始匹配
        match = pattern.match(old_name)
        if not match:   # 不符合格式则跳过
            continue

        ### User can change this part to suit their own need #######
        x_str = match.group(1)
        y_str = match.group(2)
        new_name = f'{pose_mapping[y_str]}_trial_{x_str}_on.npy'
        ############################################################
        new_absPath = os.path.join(folder_absPath, new_name)
        os.rename(old_path, new_absPath)
        print(f"重命名: {old_name}  -->  {new_name}")

def batch_move_files(root: str, dst: str, type: int) -> None:
    """
    Copy files from root folder to dst folder.

    Params:
        type (int):
        * 1 means copy and paste every single file;
        * 2 means clip and paste every single file;
        * 3 means clip and paste the whole folder.
    """
    file_list = os.listdir(root)
    if type == 1:
        for file in file_list:
            file_absPath = os.path.join(root, file)
            shutil.copy(file_absPath, dst)
    elif type == 2:
        for file in file_list:
            file_absPath = os.path.join(root, file)
            shutil.move(file_absPath, dst)
    elif type == 3:
        shutil.move(root, dst)
    print("Finished.")

def plot_spike_raster(
        file_absPath: str,
        plot_method: int,
        gridsize: Tuple[int, int],
        leftTop: Tuple[int, int],
        rightBottom: Tuple[int, int],
        period: Tuple[int, int]
    ) -> None:
    """
    Easily plot the spike raster plot with only 1 function!

    Params:
        file_absPath (str): the absolute path of the file.
        plot_method (int): 1 means original data and plot it; 2 means\
                original data and plot after preprocess raster; 3\
                means preprocessed data and plot it.
        gridsize (int, int): Same as in PreprocessStream class.
        leftTop (int, int): Same as in PreprocessStream class.
        rightBottom (int, int): Same as in PreprocessStream class.
        period (int, int): Same as in PreprocessStream class.
    """
    processor = PreprocessStream(gridsize, leftTop, rightBottom, period)
    data = load_single_sample(file_absPath)

    if plot_method == 1:
        processor.plot_raster(data, True)
    elif plot_method == 2:
        stream_after_process = processor.crop_spatial_temporal(data)
        processor.plot_raster(stream_after_process, False)
    elif plot_method == 3:
        processor.plot_raster(data, False)
    else:
        raise ValueError("plot_method parameter can only be 1, 2 or 3.")
    
def preprocess_single_file_without_saving(
        file_absPath: str, 
        processor: PreprocessStream
    ) -> None:
    """
    preprocess single file and return it in xytp form in a 1D ndarray.
    """
    data_sample = load_single_sample(file_absPath)
    event_stream = processor.crop_spatial_temporal(data_sample)
    return event_stream
