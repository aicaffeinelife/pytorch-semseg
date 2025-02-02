"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np

from collections import OrderedDict
from tabulate import tabulate


class Table:
    def __init__(self, header):
        self.header = header
        self.data = []
    def update(self, data_dict):
        if not self._checkeys(data_dict):
            raise KeyError("Improper data")
        self.data.append([v for k,v in data_dict.items()])

    def dump_to_file(self, fname):
        with open(fname, 'w') as wfile:
            wfile.write(tabulate(self.data, headers=self.header))

    def print_table(self):
        print(tabulate(self.data, headers=self.header))

    def print_recent(self):
        print(tabulate(self.data[-1], headers=self.header))

    def print_upto(self, history=10):
        """Prints the data from the end"""
        data = self.data[history:]
        print(tabulate(data, headers=self.header))

    def flush(self, keepheader=True):
        """Flush the entire contents of the data"""
        del self.data[:] # inplace flush
        if not keepheader:
            self.header = None

    def flush_upto(self, history=100):
        """Flush data in the given history"""
        assert(history < len(self.data))
        del self.data[:history]

    def _checkeys(self, data_dict):
        keys = [k for k in data_dict.keys()]
        if len(keys) != len(self.header):
            return False
        isok = all([k == h for k, h in zip(keys, self.header)])
        if isok:
            return True
        else:
            return False



def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


