import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import ListedColormap

def main(
    file_abspath: str, root_dir: str=os.path.dirname(__file__),
    disp_heatmap: bool=True, disp_spikes: bool=True, ssim_calc: bool=True
) -> None:
    """
    This function is used to check the heatmap, spike raster and the ssim map of the data.

    Parameters:
        file_abspath: absolute path of the pickle file. format: ndarray with lists inside.
        root_dir: where to store your images. default: in this folder.
    """

    # Font sizes
    TITLE_SIZE = 20
    AXES_SIZE = 16
    TICKS_SIZE = 12

    # file_name = os.path.basename(file_abspath)
    fig_save_dir = os.path.join(root_dir, "figures")

    # Test whether the dir exists. If not, make one.
    if not os.path.exists(fig_save_dir):
        os.mkdir(fig_save_dir)

    with open(file_abspath, 'rb') as f:
        data = pickle.load(f)

    ############### heatmap ################
    if disp_heatmap:
        # DVS and the pickle/numpy have inverted x and y axis. In numpy/pickle, rows are X, columns are Y. In DVS, rows are Y, columns are X.
        x_size = int(data.shape[1])
        y_size = int(data.shape[0])

        #create array for heatmap
        data_heatmap = np.empty((y_size, x_size))

        cmap = plt.cm.viridis.reversed()  # 选择一个基础的渐变色，例如viridis
        cmap_colors = cmap(np.arange(cmap.N))  # 获取 colormap 的颜色表
        cmap_colors[0] = [1, 1, 1, 1]  # 将第一个颜色（最小值0对应的颜色）设置为白色
        custom_cmap = ListedColormap(cmap_colors)

        # moving along x in each y count number of event timestamps in list
        for y in range(y_size):
            for x in range(x_size): 
                data_heatmap[y][x] = (len(data[y][x]))
        #reshape data to original shape
        data_heatmap = np.reshape(data_heatmap,(y_size,-1))

        plt.imshow(data_heatmap, cmap=custom_cmap, vmin=0, vmax=np.max(data_heatmap))
        plt.colorbar()
        plt.savefig(os.path.join(fig_save_dir, "heatmap.png"))
        plt.show()

    ############### spike raster ################
    if disp_spikes:
        spikes_data = data.flatten()
        neuron_idx = 0

        for spike_train in spikes_data:
            if spike_train != []:
                y = np.ones_like(spike_train) * neuron_idx
                plt.plot(spike_train, y, 'k|', markersize=0.7)
            neuron_idx +=1

        # #可以加入垂直的红线以标记起始和结束位置
        # plt.axvline(x=215, color='r', linestyle='-', linewidth=2)
        # plt.axvline(x=866, color='r', linestyle='-', linewidth=2)
        # ###############

        plt.ylim = (0, len(spikes_data))
        plt.ylabel("neuron", fontsize=AXES_SIZE)
        plt.xlabel("time (ms)", fontsize=AXES_SIZE)
        plt.title("Spike Raster Plot", fontsize=TITLE_SIZE)
        plt.setp(plt.gca().get_xticklabels(), visible=True)
        plt.savefig(os.path.join(fig_save_dir, 'spikes.png'))
        plt.show()


    ############### SSIM ################
    if disp_heatmap & ssim_calc:
        zeros_heatmap = np.zeros((y_size,x_size))
        mssim_score, mssim_local_map = ssim(data_heatmap, zeros_heatmap, gaussian_weights = True, full = True, data_range=(max(np.max(data_heatmap),np.max(zeros_heatmap))-min(np.min(data_heatmap),np.min(zeros_heatmap))), use_sample_covariance = False)
        print(mssim_score)

        #########################
        fig, ax = plt.subplots()
        # cax = ax.imshow(mssim_local_map, cmap=cmap, vmin=0, vmax=1)
        inverted_mssim_local_map = 1 - mssim_local_map
        cax = ax.imshow(inverted_mssim_local_map, cmap=cmap, vmin=0, vmax=1)
        # rect1 = plt.Rectangle((185, 105), 270, 270, fill=False, edgecolor='r', linewidth=5)
        rect2 = plt.Rectangle((205, 107), 260, 260, fill=False, edgecolor='r', linewidth=5)
        
        # ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.set_xlabel('X pixel', fontdict={'fontsize':AXES_SIZE})
        ax.set_ylabel('Y pixel', fontdict={'fontsize':AXES_SIZE})
        ax.set_title(' Heatmap of Spikes', fontdict={'fontsize':TITLE_SIZE})
        fig.colorbar(cax)
        ########################
        
        # plt.imshow(mssim_local_map)
        plt.savefig(os.path.join(fig_save_dir, 'mssim_local_map.png'))
        plt.show()

if __name__ == '__main__':
    file_abspath = r'f:\Files\PhD\coop\Data\PreTest\1223_us\ABB_AOLI_12231254_cottonnylon\events\taps_trial_0_pose_0_events_on'
    main(file_abspath=file_abspath)