import inspect
from types import FrameType
from pathlib import Path

def parent_dir(currentframe: FrameType) -> Path:
    """Find parent directory path in runtime"""
    return Path(inspect.getabsfile(currentframe)).parent