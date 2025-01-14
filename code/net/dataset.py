
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def split_data(root_folder_dir: str, train_size: float):
    """
    Split train and test data into 2 lists, each containing their absolute
    paths.

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
                train_size, shuffle=True
                )
            train_list.extend(xtrain)
            test_list.extend(xtest)
    
    return train_list, test_list

class Dataset_Texture_Stream(Dataset):
    def __init__(
            self,
            root_folder_dir: str,
            bin_timestep: int,
            gridsize: int,
            phase: str,
            train_size: float
        ) -> None:
        super().__init__()
        self.root_folder_dir = root_folder_dir
        self.bin_timestep = bin_timestep
        self.gridsize = gridsize
        self.phase = phase
        self.train_size = train_size

