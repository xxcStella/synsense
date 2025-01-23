"""
This file contains preprocessor class. They are used to preprocess different
data initial format.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class PreprocessStream:
    """
    preprocess the stream event directly from the robot.
    The data from the robot is a 1D ndarray with tuples containing data type of xytp.
    We crop them in spatial and temporal spaces.
    Return a new ndarray containing xytp. (sequence in time, ascending)

    Parameters:
        gridsize: (xlength, ylength). compatible with numpy format. x is row, y is column.
        leftTop: (xmin, ymin), left top corner coordinate. xy are values compatible with 
                numpy, where x is row and y is column.
        period (int, int): start time and end time to crop in temporal space.\
                    Need to pay attention to the unit, ms or us.
    """
    def __init__(
            self,
            gridsize: Tuple[int, int],
            leftTop: Tuple[int, int],
            rightBottom: Tuple[int, int],
            period: Tuple[int, int]
        ) -> None:
        self.gs_x, self.gs_y = gridsize

        self.ymin, self.xmin = leftTop  # 这里的xmin等都是相对于numpy格式的,即x对应行数,y对应列数
        self.ymax, self.xmax = rightBottom

        self.tstart, self.tend = period

    def crop_spatial_temporal(self, data) -> np.ndarray:
        """
        crop in spatial and temporal spaces. Only remains data in the square and within the time length.
        
        Params:
            data (ndarray): event stream data. xytp, x means column, y means row.

        Return:
            event_stream: ndarray containing tuple with data type xytp.
        """
        event_stream = []

        # 限制只有在区域范围以及时间范围内的xytp能被加入新列表
        for i in range(len(data)):
            if data[i]['x'] >= self.xmin and data[i]['x'] <= self.xmax \
                and data[i]['y'] >= self.ymin and data[i]['y'] <= self.ymax and \
                    data[i]['t'] >= self.tstart and data[i]['t'] < self.tend:
                
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
    
    def plot_raster(self, event_stream, is_origin_data: bool) -> None:
        """
        Plot spike raster.

        Params:
            event_stream: event stream data in xytp format. ndarray.
            is_origin_data (bool): whether to plot original data from camera or 
                        data after preprocess.
        """
        if is_origin_data:
            x = event_stream['t']
            # Need to change the param when using another camera!!
            y = 640 * event_stream['x'] + event_stream['y']
            plt.plot(x, y, 'k|', markersize=0.7)
        else:
            x = event_stream['t']
            y = self.gs_y * event_stream['x'] + event_stream['y']
            plt.plot(x, y, 'k|', markersize=0.7)
            
        plt.ylabel("neuron", fontsize=12)
        plt.xlabel("time", fontsize=12)
        plt.title("Spike Raster Plot", fontsize=14)
        plt.setp(plt.gca().get_xticklabels(), visible=True)
        plt.show()
