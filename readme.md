This folder contains files to preprocess and train, test the net on CPU, GPU and \
Speck2f board.

In net folder, dataset.py contains the basic class for data preprocessing. All \
future data class should inherit this class.

The auto.py file is to train the net on a single gpu. The auto_ddp.py file can train \
the net on a single GPU and multi-GPUs. For simple test, we recommend to use auto.py, \
for large scale test, use auto_ddp.py. If you find the performance on single GPU \
and on multi GPUs are different, try to make the batch size on each GPU from the \
multi GPU the same as the batch size on the single GPU.