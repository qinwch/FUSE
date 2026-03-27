import logging
import time
import json
import h5py
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union
import numpy as np


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to set up file handler: {e}")
    
    return logger


@contextmanager
def measure_time(logger: Optional[logging.Logger] = None, description: str = "Operation"):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        message = f"{description} completed in {elapsed_time:.4f} seconds"
        if logger:
            logger.info(message)
        else:
            print(message)


def save_to_hdf5(
    data: Dict[str, Union[np.ndarray, Any]],
    file_path: str,
    overwrite: bool = False,
    compression: Optional[str] = "gzip"
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    mode = 'w' if overwrite else 'a'
    
    try:
        with h5py.File(file_path, mode) as f:
            for key, value in data.items():
                if key in f:
                    if overwrite:
                        del f[key]
                    else:
                        raise ValueError(f"Dataset '{key}' already exists. Set overwrite=True to replace.")
                
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression=compression)
                elif isinstance(value, str):
                    # 保存字符串为变长字符串类型
                    f.create_dataset(key, data=value, dtype=h5py.string_dtype())
                elif isinstance(value, (list, tuple)):
                    # 检查列表中的元素类型
                    if len(value) > 0 and isinstance(value[0], str):
                        # 字符串列表
                        f.create_dataset(key, data=value, dtype=h5py.string_dtype())
                    else:
                        f.create_dataset(key, data=np.array(value), compression=compression)
                else:
                    f.create_dataset(key, data=np.array(value))
    except Exception as e:
        raise IOError(f"Failed to save HDF5 file: {e}")


def load_from_hdf5(file_path: str, keys: Optional[list] = None) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    try:
        data = {}
        with h5py.File(file_path, 'r') as f:
            if keys is None:
                keys = list(f.keys())
            
            for key in keys:
                if key not in f:
                    raise KeyError(f"Key '{key}' not found in HDF5 file")
                
                # 处理标量数据集
                dataset = f[key]
                if dataset.shape == ():
                    # 标量数据
                    data[key] = dataset[()]
                else:
                    # 数组数据
                    data[key] = dataset[:]
        
        return data
    except Exception as e:
        raise IOError(f"Failed to load HDF5 file: {e}")


def save_json_metadata(
    metadata: Dict[str, Any],
    file_path: str,
    overwrite: bool = False,
    indent: int = 4
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"Metadata file already exists: {file_path}. Set overwrite=True to replace.")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save JSON metadata: {e}")


def load_json_metadata(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load JSON metadata: {e}")
