"""
This file is used to store users' own functions.
"""
import os
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def split_train_test(root_folder_dir: str, train_size: float, random_seed: int=None) -> Tuple[List, List]:
    """
    在root文件夹下，有好几个文件夹，每个文件夹下有更多子文件夹，
    每个子文件夹有两种材料，随机分成训练集和测试集

    root_folder_dir
    |-Acrylic_Fashion
    |  |-0123_bigdataset_white_01241942_depth_1_speed_30
    |    |-Acrylic_trial_0_on.npy
    |    |-Fashionfabric_trial_10_on.npy
    |  |-0123_bigdataset_white_01241957_depth_1_speed_40
    |-Cotton_Nylon
    |-...
    """
    train_list = []
    test_list  = []

    for sub_folder in os.listdir(root_folder_dir):  # Acrylic_Fashion
        sub_folder_path = os.path.join(root_folder_dir, sub_folder)

        for sub_sub_folder in os.listdir(sub_folder_path): # 0123_bigdataset_white_01241942_depth_1_speed_30
            # 去掉depth=0.5mm的文件
            if not "depth_2.5_speed_50" in sub_sub_folder:
                continue
            
            sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)

            material_files = {}

            for file_name in os.listdir(sub_sub_folder_path):
                file_path = os.path.join(sub_sub_folder_path, file_name)

                if os.path.isfile(file_path):
                    # 判断文件属于哪个材料 (以材料名开头)
                    material = file_name.split("_")[0]  # 假设材料名是文件名的第一部分

                    if material not in material_files:
                        material_files[material] = []

                    material_files[material].append(file_path)

            for material, file_list in material_files.items():
                x_train, xtest, _, _ = train_test_split(
                    file_list,
                    np.zeros(len(file_list)),
                    train_size=train_size,
                    shuffle=True,
                    random_state=random_seed
                )
                train_list.extend(x_train)
                test_list.extend(xtest)
    
    return train_list, test_list