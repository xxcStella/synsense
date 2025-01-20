"""
This file contains basic functiion for setting parameters and parallel preprocessing.
"""

import os
import numpy as np
from .preprocessors import PreprocessStream
from typing import Tuple, Dict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def make_meta(
        gridsize: Tuple[int, int],
        leftTop: Tuple[int, int],
        rightBottom: Tuple[int, int],
        period: Tuple[int, int],
) -> Dict:
    """
    You need to define all parameters in this function!!!
    The meaning of params refers to PreprocessStream class in preprocessors.py
    """
    meta = locals().copy()
    return meta

def load_single_sample(file_absPath: str) -> np.ndarray:
    """
    Load single numpy file. file should be the event stream (xytp) in
        numpy form dirctly from the robot.
    """
    with open(file_absPath, 'rb') as f:
        data_sample = np.load(f)
    return data_sample

def preprocess_single_file(
        file_absPath: str, 
        output_folder: str, 
        processor: PreprocessStream) -> None:
    """
    preprocess single file and save it in xytp form in a 1D ndarray.
    技巧: processor在外部实例化后,并行处理只需要实例化一次,减少时间开销
    """
    data_sample = load_single_sample(file_absPath)
    event_stream = processor.crop_spatial_temporal(data_sample)

    # 获取纯文件名，并丢弃扩展名
    base_name = os.path.splitext(os.path.basename(file_absPath))[0]
    output_path = os.path.join(output_folder, f"{base_name}.npy")
    np.save(output_path, event_stream)

def parallel_preprocessing(input_folder: str, output_folder: str, meta: Dict):
    """
    parallelly preprocess files in a single folder.
    You can use a loop to traverse all input folders to realize process
        multiple folders.
    """
    all_files_absPath = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
    ]
    processor = PreprocessStream(**meta)
    with ProcessPoolExecutor() as executor:
        func = partial(preprocess_single_file, output_folder=output_folder, processor=processor)
        # executor.map 返回一个迭代器，按顺序返回各文件处理后的结果
        for saved_path in executor.map(func, all_files_absPath):
            print("Saved:", saved_path)


if __name__ == "__main__":
    input_folder = r"F:\input_folder"   # 存放原始 .npy 数据, from robot(xytp)
    output_folder = r"F:\output_folder" # 存放处理后的 events (event stream, xytp)
    meta = make_meta()  # remember to modify the params in this function!!!
    parallel_preprocessing(input_folder, output_folder, meta)