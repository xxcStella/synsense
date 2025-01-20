import os
import re
import shutil

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

def batch_move_files(root: str, dst: str) -> None:
    """
    Copy files from root folder to dst folder.
    """
    file_list = os.listdir(root)
    for file in file_list:
        file_absPath = os.path.join(root, file)
        shutil.copy(file_absPath, dst)
    print("Finished.")