import inspect
from types import FrameType
from pathlib import Path
from datetime import datetime

from numpy import ndarray
import matplotlib.pyplot as plt

def parent_dir(currentframe: FrameType) -> Path:
    """Find parent directory path in runtime"""
    return Path(inspect.getabsfile(currentframe)).parent

class FileSave:
    @classmethod
    def fig_state(cls, arr: ndarray, name: str):
        plt.imshow(arr) 
        plt.savefig(f"{parent_dir(inspect.currentframe()).parent}/figures/state"
            f"{datetime.now().strftime('%H_%M_%S')}_{name}.png")
    
    @classmethod
    def fig_metrics(cls, name: str):
        plt.savefig(f"{parent_dir(inspect.currentframe()).parent}/figures/metrics/"
            f"{datetime.now().strftime('%H_%M_%S')}_{name}.png")
