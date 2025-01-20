# import os
import numpy as np
# import pickle
import matplotlib.pyplot as plt
from typing import Tuple

class PreprocessStream:
    """
    preprocess the stream event directly from the robot.
    The data from the robot is a 1D ndarray with tuples containing data type of xytp.
    We crop them in spatial and temporal spaces.
    Return a new ndarray containing xytp. (sequence in time, ascending)
    """
    def __init__(
            self,
            gridsize: Tuple[int, int],
            leftTop: Tuple[int, int],
            rightBottom: Tuple[int, int],
            period: Tuple[int, int]
        ) -> None:
        """
        Parameters:
            data: ndarray with tuple inside, xytp format. data directly from robot in event stream (xytp)
            gridsize: (xlength, ylength). compatible with numpy format. x is row, y is column.
            leftTop: (xmin, ymin), left top corner coordinate. xy are values compatible with 
                    numpy, where x is row and y is column.
        """
        self.gs_x, self.gs_y = gridsize

        self.ymin, self.xmin = leftTop  # 这里的xmin等都是相对于numpy格式的,即x对应行数,y对应列数
        self.ymax, self.xmax = rightBottom

        self.tstart, self.tend = period

    def crop_spatial_temporal(self, data) -> np.ndarray:
        """
        crop in spatial and temporal spaces. Only remains data in the square and within the time length.
        
        Return:
            event_stream: ndarray containing tuple with data type xytp.
        """
        event_stream = []

        # 限制只有在区域范围以及时间范围内的xytp能被加入新列表
        for i in range(len(self.data)):
            if data[i]['x'] >= self.xmin and data[i]['x'] <= self.xmax \
                and data[i]['y'] >= self.ymin and data[i]['y'] >= self.ymax and \
                    data[i]['t'] >= self.tstart and data[i]['t'] <= self.tend:
                
                event_stream.append((
                    data[i]['x'] - self.xmin,
                    data[i]['y'] - self.ymin,
                    data[i]['t'] - self.tstart,
                    data[i]['p']
                    ))

        event_stream = np.array(
            event_stream, 
            dtype=[('x', '<i4'), ('y', '<i4'), ('t', '<i4'), ('p', '<i4')]
            )
        event_stream.sort(order='t')
        return event_stream
    
    def plot_raster(self, event_stream):
        for event in event_stream:
            neuron_idx = self.gs_y * event['x'] + event['y']
            plt.plot(neuron_idx, event['t'], 'k|', markersize=0.7)
        
        plt.ylabel("neuron", fontsize=12)
        plt.xlabel("time (ms)", fontsize=12)
        plt.title("Spike Raster Plot", fontsize=14)
        plt.setp(plt.gca().get_xticklabels(), visible=True)
        plt.show()
