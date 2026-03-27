import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch.nn import Module

try:
    from sbibm.scripts.utils import (
        setup_logger,
        save_to_hdf5,
        load_from_hdf5,
        save_json_metadata,
        load_json_metadata,
        measure_time,
    )
except ImportError:
    from utils import (
        setup_logger,
        save_to_hdf5,
        load_from_hdf5,
        save_json_metadata,
        load_json_metadata,
        measure_time,
    )


class DataCollector:
    def __init__(
        self,
        method: str,
        task: str,
        base_dir: str = ".",
        log_level: int = logging.INFO,
    ):
        self.method = method
        self.task = task
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints", method, task)
        self.results_dir = os.path.join(base_dir, "results", method, task)
        
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger = setup_logger(
            "DataCollector",
            os.path.join(self.results_dir, f"data_collector_{self.timestamp}.log"),
            level=log_level,
        )
        
        self.logger.info(f"DataCollector initialized for {method} on {task}")
        self.logger.info(f"Checkpoints directory: {self.checkpoints_dir}")
        self.logger.info(f"Results directory: {self.results_dir}")
    
    def save_checkpoint(
        self,
        model: Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        if filename is None:
            filename = f"{self.method}_{self.task}_{self.timestamp}.ckpt"
        
        checkpoint_path = os.path.join(self.checkpoints_dir, filename)
        
        try:
            self.logger.info(f"Saving checkpoint to: {checkpoint_path}")
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "method": self.method,
                "task": self.task,
                "timestamp": self.timestamp,
                "datetime": datetime.now().isoformat(),
            }
            
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            
            if additional_metadata is not None:
                checkpoint["additional_metadata"] = additional_metadata
            
            temp_path = f"{checkpoint_path}.tmp"
            torch.save(checkpoint, temp_path)
            shutil.move(temp_path, checkpoint_path)
            
            self._validate_checkpoint(checkpoint_path)
            self.logger.info(f"✓ Checkpoint saved successfully: {checkpoint_path}")
            
            return checkpoint_path
        
        except Exception as e:
            self.logger.error(f"✗ Failed to save checkpoint: {e}")
            if os.path.exists(f"{checkpoint_path}.tmp"):
                os.remove(f"{checkpoint_path}.tmp")
            raise IOError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(
        self,
        filename: str,
        model: Optional[Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> Dict[str, Any]:
        checkpoint_path = os.path.join(self.checkpoints_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            if model is not None and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                self.logger.info("✓ Model state loaded")
            
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.logger.info("✓ Optimizer state loaded")
            
            return checkpoint
        
        except Exception as e:
            self.logger.error(f"✗ Failed to load checkpoint: {e}")
            raise IOError(f"Failed to load checkpoint: {e}")
    
    def _validate_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise IOError("Checkpoint file was not created")
        
        file_size = os.path.getsize(checkpoint_path)
        if file_size == 0:
            raise IOError("Checkpoint file is empty")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            required_keys = ["model_state_dict", "method", "task", "timestamp"]
            for key in required_keys:
                if key not in checkpoint:
                    raise IOError(f"Checkpoint missing required key: {key}")
        except Exception as e:
            raise IOError(f"Checkpoint validation failed: {e}")
    
    def save_samples(
        self,
        samples: np.ndarray,
        additional_data: Optional[Dict[str, Union[np.ndarray, Any]]] = None,
        filename: str = "samples.h5",
        overwrite: bool = False,
    ) -> str:
        if not isinstance(samples, np.ndarray):
            raise TypeError("Samples must be a numpy array")
        
        if samples.ndim != 2:
            raise ValueError(f"Samples must be 2D array, got {samples.ndim}D")
        
        h5_path = os.path.join(self.results_dir, filename)
        
        data = {
            "samples": samples,
        }
        
        if additional_data is not None:
            data.update(additional_data)
        
        try:
            self.logger.info(f"Saving samples to: {h5_path}")
            
            temp_path = f"{h5_path}.tmp"
            save_to_hdf5(data, temp_path, overwrite=True)
            shutil.move(temp_path, h5_path)
            
            self._validate_hdf5(h5_path)
            self.logger.info(f"✓ Samples saved successfully: {h5_path}")
            
            return h5_path
        
        except Exception as e:
            self.logger.error(f"✗ Failed to save samples: {e}")
            if os.path.exists(f"{h5_path}.tmp"):
                os.remove(f"{h5_path}.tmp")
            raise IOError(f"Failed to save samples: {e}")
    
    def load_samples(
        self,
        filename: str = "samples.h5",
    ) -> Dict[str, Any]:
        h5_path = os.path.join(self.results_dir, filename)
        return load_from_hdf5(h5_path)
    
    def _validate_hdf5(self, h5_path: str) -> None:
        if not os.path.exists(h5_path):
            raise IOError("HDF5 file was not created")
        
        file_size = os.path.getsize(h5_path)
        if file_size == 0:
            raise IOError("HDF5 file is empty")
        
        data = load_from_hdf5(h5_path)
        if "samples" not in data:
            raise IOError("HDF5 file missing 'samples' dataset")
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "metadata.json",
        overwrite: bool = False,
    ) -> str:
        metadata_path = os.path.join(self.results_dir, filename)
        
        full_metadata = {
            "method": self.method,
            "task": self.task,
            "timestamp": self.timestamp,
            "datetime": datetime.now().isoformat(),
            **metadata,
        }
        
        try:
            self.logger.info(f"Saving metadata to: {metadata_path}")
            save_json_metadata(full_metadata, metadata_path, overwrite=overwrite)
            self.logger.info(f"✓ Metadata saved successfully: {metadata_path}")
            
            return metadata_path
        
        except Exception as e:
            self.logger.error(f"✗ Failed to save metadata: {e}")
            raise IOError(f"Failed to save metadata: {e}")
    
    def load_metadata(
        self,
        filename: str = "metadata.json",
    ) -> Dict[str, Any]:
        metadata_path = os.path.join(self.results_dir, filename)
        return load_json_metadata(metadata_path)
    
    def generate_readme(
        self,
        description: str = "",
        additional_info: Optional[Dict[str, str]] = None,
        filename: str = "README.md",
    ) -> str:
        readme_path = os.path.join(self.results_dir, filename)
        
        readme_content = f"""# {self.method} - {self.task}

## Overview
This directory contains results and checkpoints for the {self.method} method on the {self.task} task.

{description}

## Directory Structure
```
{self.base_dir}/
├── checkpoints/
│   └── {self.method}/
│       └── {self.task}/
│           └── {self.method}_{self.task}_{self.timestamp}.ckpt
└── results/
    └── {self.method}/
        └── {self.task}/
            ├── samples.h5
            ├── metadata.json
            ├── README.md
            └── data_collector_{self.timestamp}.log
```

## Files
- `samples.h5`: HDF5 file containing samples
- `metadata.json`: JSON file with metadata about the run
- `README.md`: This file
- `data_collector_*.log`: Log file from the data collector

## Metadata
- Method: {self.method}
- Task: {self.task}
- Timestamp: {self.timestamp}
- Generated: {datetime.now().isoformat()}

"""
        
        if additional_info:
            readme_content += "\n## Additional Information\n"
            for key, value in additional_info.items():
                readme_content += f"- **{key}**: {value}\n"
        
        try:
            self.logger.info(f"Generating README at: {readme_path}")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            self.logger.info(f"✓ README generated successfully: {readme_path}")
            
            return readme_path
        
        except Exception as e:
            self.logger.error(f"✗ Failed to generate README: {e}")
            raise IOError(f"Failed to generate README: {e}")
    
    def get_paths(self) -> Dict[str, str]:
        return {
            "checkpoints_dir": self.checkpoints_dir,
            "results_dir": self.results_dir,
            "timestamp": self.timestamp,
        }
