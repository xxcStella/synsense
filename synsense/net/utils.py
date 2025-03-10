"""
This file contains some useful functions,
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(root_folder_dir: str, train_size: float, random_seed: int=None):
    """
    Split train and test data into 2 lists, each containing their absolute
    paths.

    If you want the whole folder to be training set (testing set), set train_size
    =1 or 0.

    root_folder_dir
    |-Wool_0_folder
    |  |-Wool_trial_0_on.npy
    |  |-Wool_trial_1_on.npy
    |-Canvas_1_folder
    |-...
    """
    folder_list = os.listdir(root_folder_dir)
    train_list = []
    test_list  = []

    for folder in folder_list:
        # All folders' absolute paths
        folder_absPath = os.path.join(root_folder_dir, folder)
        # If the path is a directory, go on
        if os.path.isdir(folder_absPath):
            # 获取文件夹下所有npy文件的绝对路径列表
            file_list = [
                item.path for item in os.scandir(folder_absPath) if item.is_file()
                ]
            xtrain, xtest, _, _ = train_test_split(
                file_list, 
                np.zeros(len(file_list)), 
                train_size, shuffle=True,
                random_state=random_seed
                )
            train_list.extend(xtrain)
            test_list.extend(xtest)
    
    return train_list, test_list