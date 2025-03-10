"""
This file contains dataset class for organize your own data.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tonic.transforms import ToFrame
from typing import Tuple, Optional

class Dataset_Stream_Base(Dataset):
    """
    Base class for stream-based dataset.
    This dataset is used to process event streams (xytp) into frames for PC training
    and testing and into streams (xytp) for synsense speck2f board to infer.

    Params:
        data_list: list. Contains a list of data sample absolute path. All data are \
                either training data or testing data. Cannot mix them.
        platform: str={'pc', 'speck'}. Dataset deplyed on PC or speck board.
        gridsize: (int, int). The size of grid you want after dividing in grid. gs_x\
                means row num, gs_y means column num.
        after_crop_size: Optional(int, int, int). For ToFrame function. It is the size of\
                frames after cropping from the robot data. eg. (260, 260, 1)
        n_time_bins: Optional(int). The param is for ToFrame function in tonic. Controls how\
                many frames (slices) you want when deploying on PC. This param is\
                optional, needed only when using PC.
        sort_time (Optional[bool]): whether sort the time of the original data before\
                training and testing.
    """
    def __init__(
            self,
            data_list: list,
            platform: str,
            gridsize: Tuple[int, int],
            after_crop_size: Optional[Tuple[int, int, int]]=None,
            n_time_bins: Optional[int]=None,
            sort_time: Optional[bool]=False
        ) -> None:
        super().__init__()
        self.data_list = data_list
        self.platform = platform
        self.gs_x, self.gs_y = gridsize
        self.after_crop_size = after_crop_size
        self.n_time_bins = n_time_bins
        self.sort_time = sort_time

    def _divide_to_grid(self, data: np.ndarray) -> np.ndarray:
        """
        Divide a single sample into gridsize matrix. Used in training phase (frames).
        If the data is not divisible, it will round to the closest integer.

        Params:
            data: ndarry frame, (T, C, H, W).
        
        Returns:
            trimmed_mat: ndarray frame. (T, C, gs_x, gs_y)
        """
        T, C, H, W = data.shape
        H_trimmed = (H // self.gs_x) * self.gs_x
        W_trimmed = (W // self.gs_y) * self.gs_y

        trimmed_mat = data[:, :, :H_trimmed, :W_trimmed]
        trimmed_mat = trimmed_mat.reshape(
            T, C,
            self.gs_x, H_trimmed//self.gs_x,
            self.gs_y, W_trimmed//self.gs_y
        )
        trimmed_mat = trimmed_mat.sum(axis=(3, 5))
        return trimmed_mat
    # np.allclose(mat_a, mat_b) # 用于测试两个矩阵是否相等,在大约1e-6误差内

    def get_label(self, file_absPath: str) -> torch.Tensor:
        """
        Choose part of the file name to be the label.
        This function can be modify according to user.

        Params:
            file_absPath: str. single sample's absolute path.

        Return:
            label: torch.long type. represent the label index.

        eg. Wool_trial_x_on.npy will be Wool.
        """
        pass

class Dataset_Texture_Stream(Dataset_Stream_Base):
    """
    This dataset is used to process event streams (xytp) into frames for PC training
    and testing and into streams (xytp) for synsense speck board to infer.

    Params:
        data_list: list. Contains a list of data sample absolute path. All data are \
                either training data or testing data. Cannot mix them.
        platform: str={'pc', 'speck'}. Dataset deplyed on PC or speck board.
        gridsize: (int, int). The size of grid you want after dividing in grid. gs_x\
                means row num, gs_y means column num.
        after_crop_size: Optional(int, int, int). For ToFrame function. It is the size of\
                frames after cropping from the robot data. eg. (260, 260, 1)
        n_time_bins: Optional(int). The param is for ToFrame function in tonic. Controls how\
                many frames (slices) you want when deploying on PC. This param is\
                optional, needed only when using PC.
        sort_time (Optional[bool]): whether sort the time of the original data before\
                training and testing.
    """
    def __init__(
            self,
            data_list: list,
            platform: str,
            gridsize: Tuple[int, int],
            after_crop_size: Optional[Tuple[int, int, int]]=None,
            n_time_bins: Optional[int]=None,
            sort_time: Optional[bool]=False
        ) -> None:
        super().__init__(
            data_list,
            platform,
            gridsize,
            after_crop_size,
            n_time_bins,
            sort_time
        )

    def get_label(self, file_absPath: str) -> torch.Tensor:
        """
        Choose part of the file name to be the label.
        This function can be modify according to user.

        Params:
            file_absPath: str. single sample's absolute path.

        Return:
            label: torch.long type. represent the label index.

        eg. Wool_trial_x_on.npy will be Wool.
        """
        base_name = os.path.splitext(os.path.basename(file_absPath))[0]
        material = base_name.split('_')[0]

        if material == "Acrylic":
            label = torch.tensor(0, dtype=torch.long)
        elif material == "Canvas":
            label = torch.tensor(1, dtype=torch.long)
        elif material == "Cotton":
            label = torch.tensor(2, dtype=torch.long)
        elif material == "Fashionfabric":
            label = torch.tensor(3, dtype=torch.long)
        elif material == "Felt":
            label = torch.tensor(4, dtype=torch.long)
        elif material == "Fur":
            label = torch.tensor(5, dtype=torch.long)
        elif material == "Mesh":
            label = torch.tensor(6, dtype=torch.long)
        elif material == "Nylon":
            label = torch.tensor(7, dtype=torch.long)
        elif material == "Wood":
            label = torch.tensor(8, dtype=torch.long)
        elif material == "Wool":
            label = torch.tensor(9, dtype=torch.long)
        return label

    def __getitem__(self, idx: int):
        single_sample_absPath = self.data_list[idx]
        with open(single_sample_absPath, 'rb') as f:
            data = np.load(f) # data should be event stream (xytp) in ascending order of 't'
        
        if self.sort_time:
            data = np.sort(data, order='t')
        
        # get the label of this sample
        label = self.get_label(single_sample_absPath)

        # get the data of this sample
        if self.platform == "pc":
            # transform stream data into frames to train
            frame_transform = ToFrame(
                sensor_size=self.after_crop_size,
                n_time_bins=self.n_time_bins
            )
            frames = frame_transform(data)
            # divide frames into gridsize data
            grid_frames = self._divide_to_grid(frames)
            grid_frames = torch.from_numpy(grid_frames).float()   # dtype converted into torch.float32

            return grid_frames, label
        
        elif self.platform == "speck":
            H, W = self.after_crop_size[:2]
            box_x_size, box_y_size = H // self.gs_x, W // self.gs_y
            new_x = data['x'] // box_x_size
            new_y = data['y'] // box_y_size
            events = np.array(
                list(zip(new_x, new_y, data['t'], data['p'])),
                dtype=[('x', '<i4'), ('y', '<i4'), ('t', '<i4'), ('p', '<i4')]
            )
            events.sort(order='t')

            return events, label

        else:
            raise Exception("Platform type error. Legal platforms are: pc, speck.")

    def __len__(self):
        return len(self.data_list)