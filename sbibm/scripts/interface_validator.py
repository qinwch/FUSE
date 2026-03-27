import inspect
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import torch
from sbibm.tasks.task import Task

try:
    from sbibm.scripts.utils import (
        setup_logger,
        save_json_metadata,
        load_json_metadata,
        measure_time,
    )
except ImportError:
    from utils import (
        setup_logger,
        save_json_metadata,
        load_json_metadata,
        measure_time,
    )


class InterfaceValidator:
    def __init__(
        self,
        output_dir: str = "./validation_results",
        log_level: int = logging.INFO,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger(
            "InterfaceValidator",
            os.path.join(output_dir, f"validation_{timestamp}.log"),
            level=log_level,
        )
        
        self.results: List[Dict[str, Any]] = []
    
    def extract_parameters(
        self, func: Callable
    ) -> Dict[str, Dict[str, Any]]:
        signature = inspect.signature(func)
        params = {}
        
        for name, param in signature.parameters.items():
            params[name] = {
                "name": name,
                "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                "default": self._serialize_value(param.default) if param.default != inspect.Parameter.empty else None,
                "kind": str(param.kind),
            }
        
        return params
    
    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif hasattr(value, "__dict__"):
            return f"<{type(value).__name__} object>"
        elif isinstance(value, (list, tuple)):
            try:
                return json.loads(json.dumps(value))
            except (TypeError, OverflowError):
                return [str(x) for x in value]
        elif isinstance(value, dict):
            try:
                return json.loads(json.dumps(value))
            except (TypeError, OverflowError):
                return {k: str(v) for k, v in value.items()}
        else:
            try:
                json.dumps(value)
                return value
            except (TypeError, OverflowError):
                return str(value)
    
    def validate_input_output_format(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        validation_result = {
            "success": True,
            "input_format_valid": True,
            "output_format_valid": True,
            "error": None,
            "input_info": {},
            "output_info": None,
        }
        
        try:
            validation_result["input_info"] = {
                "args_count": len(args),
                "args_types": [self._serialize_type(arg) for arg in args],
                "kwargs_keys": list(kwargs.keys()),
                "kwargs_types": {k: self._serialize_type(v) for k, v in kwargs.items()},
            }
            
            with measure_time(self.logger, "Function execution"):
                result = func(*args, **kwargs)
            
            validation_result["output_info"] = {
                "type": self._serialize_type(result),
                "is_tuple": isinstance(result, tuple),
                "tuple_length": len(result) if isinstance(result, tuple) else None,
                "tuple_types": [self._serialize_type(item) for item in result] if isinstance(result, tuple) else None,
            }
            
            if isinstance(result, tuple):
                if len(result) < 2:
                    validation_result["output_format_valid"] = False
                    validation_result["error"] = f"Expected tuple with at least 2 elements, got {len(result)}"
                else:
                    if not isinstance(result[0], torch.Tensor):
                        validation_result["output_format_valid"] = False
                        validation_result["error"] = f"First element should be torch.Tensor, got {type(result[0])}"
                    if not isinstance(result[1], int):
                        validation_result["output_format_valid"] = False
                        validation_result["error"] = f"Second element should be int, got {type(result[1])}"
            
            validation_result["success"] = validation_result["input_format_valid"] and validation_result["output_format_valid"]
            
        except Exception as e:
            validation_result["success"] = False
            validation_result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
        
        return validation_result
    
    def _serialize_type(self, obj: Any) -> str:
        return f"{type(obj).__module__}.{type(obj).__name__}"
    
    def validate_algorithm(
        self,
        algorithm_name: str,
        algorithm_func: Callable,
        task: Task,
        num_samples: int = 10,
        num_simulations: int = 100,
        num_observation: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.logger.info(f"Validating algorithm: {algorithm_name}")
        self.logger.info(f"Task: {task.name_display}")
        
        result = {
            "algorithm": algorithm_name,
            "task": task.name,
            "task_display_name": task.name_display,
            "timestamp": datetime.now().isoformat(),
            "parameters": self.extract_parameters(algorithm_func),
            "validation": None,
            "status": "pending",
        }
        
        try:
            observation = task.get_observation(num_observation)
            self.logger.info(f"Observation shape: {observation.shape}")
            
            validation = self.validate_input_output_format(
                algorithm_func,
                args=(),
                kwargs={
                    "task": task,
                    "num_samples": num_samples,
                    "num_simulations": num_simulations,
                    "num_observation": num_observation,
                    **kwargs,
                },
            )
            
            result["validation"] = validation
            result["status"] = "success" if validation["success"] else "failed"
            
            if validation["success"]:
                self.logger.info(f"✓ Algorithm {algorithm_name} validation succeeded")
            else:
                self.logger.error(f"✗ Algorithm {algorithm_name} validation failed: {validation['error']}")
                
        except Exception as e:
            result["status"] = "error"
            result["validation"] = {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            }
            self.logger.error(f"✗ Error validating {algorithm_name}: {str(e)}")
        
        self.results.append(result)
        return result
    
    def save_results(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_validations": len(self.results),
            "success_count": sum(1 for r in self.results if r["status"] == "success"),
            "failed_count": sum(1 for r in self.results if r["status"] == "failed"),
            "error_count": sum(1 for r in self.results if r["status"] == "error"),
            "results": self.results,
        }
        
        save_json_metadata(summary, filepath, overwrite=True)
        self.logger.info(f"Results saved to: {filepath}")
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        return load_json_metadata(filepath)
    
    def print_summary(self):
        total = len(self.results)
        success = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        error = sum(1 for r in self.results if r["status"] == "error")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total validations: {total}")
        self.logger.info(f"Success: {success}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Error: {error}")
        self.logger.info("=" * 80)
        
        for result in self.results:
            status_icon = {
                "success": "✓",
                "failed": "✗",
                "error": "!",
                "pending": "?",
            }.get(result["status"], "?")
            
            self.logger.info(f"{status_icon} {result['algorithm']} on {result['task']} - {result['status']}")
