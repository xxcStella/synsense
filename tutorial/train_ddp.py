import argparse
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset

from synsense.net.auto_ddp import AutoSNN, ddp_setup, cleanup
from synsense.net.dataset import Dataset_Texture_Stream
from user import split_train_test

def main(rank, world_size, args):
    ddp_setup(rank, world_size)

    # 准备数据集、模型等，这里仅示例
    root = r'/media/xingchen/T7 Shield/Neuromorphic Data/Xingchen/processed/0220_dataset_iros/general_rotate'
    train_size = 0.7
    random_seed = 42
    train_list, test_list = split_train_test(root, train_size, random_seed)

    platform = "pc"
    # gridsize = (20, 20)
    # gridsize = (26, 26)
    # gridsize = (52, 52)
    gridsize = (130, 130)
    after_crop_size = (260, 260, 1)
    n_time_bins = 100
    train_dataset = Dataset_Texture_Stream(train_list, platform, gridsize, after_crop_size, n_time_bins)
    test_dataset = Dataset_Texture_Stream(test_list, platform, gridsize, after_crop_size, n_time_bins)

    init_lr = 1e-3
    step_size = 9

    cnn_20 = nn.Sequential(
        nn.Conv2d(1, 8, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(8, 16, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(144, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 72, bias=False),
        nn.ReLU(),
        nn.Linear(72, 10, bias=False)
    )

    cnn_26 = nn.Sequential(
        nn.Conv2d(1, 8, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(8, 16, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(400, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 10, bias=False)
    )

    cnn_52 = nn.Sequential(
        nn.Conv2d(1, 4, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(4, 4, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(4, 4, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 72, bias=False),
        nn.ReLU(),
        nn.Linear(72, 10, bias=False)
    )

    cnn_130 = nn.Sequential(
        nn.Conv2d(1, 4, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(4, 8, 3, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(7688, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 72, bias=False),
        nn.ReLU(),
        nn.Linear(72, 27, bias=False)
    )
        
    device = f"cuda:{rank}"
    
    trainer = AutoSNN(batch_size=args.batch_size)
    # k_folds, epochs, ...
    fold_train_acc, fold_test_acc = trainer.train(
        model_path=args.model_path,
        model=cnn_130,
        train_dataset=train_dataset,
        device=device,
        k_folds=args.k_folds,
        epochs=args.epochs,
        init_lr=init_lr,
        step_size = step_size,
        is_ddp=True,       # 多卡
        rank=rank,
        world_size=world_size,
        # kwargs={"spike_threshold": 0.8}
    )
    
    # 只有 rank=0 的返回不是 None；其他rank返回 None
    if rank == 0:
        print("fold_train_acc:", fold_train_acc)
        print("fold_test_acc:", fold_test_acc)

    # 训练结束
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--k_folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--model_path", type=str, default=r'/home/xingchen/PhD/synsense/test/0117-xytp/models/general_rotate-gs130-tl100-ddp.pt')
    # 根据需要再添加

    args = parser.parse_args()

    # 读取 env 里的 local_rank 等
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    start = time.perf_counter()
    main(local_rank, world_size, args)
    end = time.perf_counter()

    # 只让 rank=0 打印总时间
    if local_rank == 0:
        print(f"total time: {end - start: .2f} s.")

    # 在bash中运行下面的程度以启动多GPU训练
    # torchrun --nproc_per_node=3 --master_port=29501 "/home/xingchen/PhD/synsense/test/0117-xytp/train_ddp.py"