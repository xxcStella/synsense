"""
Functions and classes to deploy SNN on Speck board from Synsense.
"""
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sinabs.backend.dynapcnn import DynapcnnNetwork
# from sinabs.backend.dynapcnn.chip_factory import ChipFactory
import samna
from collections import Counter
from typing import Tuple

class DeploySpeck:
    """
    Organize and deploy net on the Speck board.
    
    Params:
        input_shape (Tuple[int, int, int]): Channel, H, W. shape of the input\
                    except time dimension.
    """
    def __init__(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape

    def open_device(self, device_name: str="Speck2fDevKit:0"):
        # devices = samna.device.get_unopened_devices()
        # board = samna.device.open_device(devices[0])
        board = samna.device.open_device(device_name)
        print(board)
        print("Board opened successfully.")

        return board

    def close_device(self, board) -> None:
        samna.device.close_device(board)
        print("Board closed successfully.")

    def deploy_net_on_Speck(
            self, 
            model_path: str, 
            devkit_name: str="speck2fdevkit:0"
            ) -> DynapcnnNetwork:
        """
        Deploy net on the speck board.

        Params:
            model_path (str): absolute path of the trained snn model.
            devkit_name (str): no need to change if you are using this board.

        Return:
            dynapcnn (DynapcnnNetwork): model on board. use it to infer on board.
        """
        model = torch.load(model_path, map_location=torch.device('cpu'))

        # prepare a dynapcnn-compatible network based on the given sinabs net.
        dynapcnn = DynapcnnNetwork(
            snn=model, 
            input_shape=self.input_shape, 
            discretize=True, 
            dvs_input=False
        )
        # deploy the net on the board
        dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
        print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")

        return dynapcnn

    def infer(self, dataset: Dataset, dynapcnn: DynapcnnNetwork) -> None:
        """
        Infer on board.
        """
        test_samples = 0
        correct_samples = 0

        inferece_p_bar = tqdm(dataset)
        for events, label in inferece_p_bar:
            samna_event_stream = []
            # convert AER format (xytp) to Samna format (x y timestamp feature...)
            for ev in events:
                spk = samna.speck2f.event.Spike()
                spk.x = ev['x']
                spk.y = ev['y']
                spk.timestamp = ev['t'] - events['t'][0]
                spk.feature = ev['p']
                # Spikes will be sent to layer/core #0, since the SNN is deployed on core: [0, 1, 2, 3]
                spk.layer = 0
                samna_event_stream.append(spk)
            output_events = dynapcnn(samna_event_stream)

            # use the most frequent output neruon index as the final prediction
            neuron_index = [each.feature for each in output_events]

            if len(neuron_index) != 0:
                frequent_counter = Counter(neuron_index)
                prediction = frequent_counter.most_common(1)[0][0]
            else:
                prediction = -1
            inferece_p_bar.set_description(f"label: {label}, prediction: {prediction}, output spikes num: {len(output_events)}") 

            if prediction == label:
                correct_samples += 1

            test_samples += 1

        print(f"On chip inference accuracy: {correct_samples / test_samples}")