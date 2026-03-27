import argparse
import logging
import sys
import os
from typing import Dict, List, Optional, Any

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))
sys.path.insert(0, script_dir)

import sbibm
from sbibm.algorithms import smc_abc as smc_abc_sbi
from sbibm.algorithms.pyabc.smcabc import run as smc_abc_pyabc
from sbibm.tasks.task import Task

from interface_validator import InterfaceValidator
from utils import setup_logger, save_json_metadata


TASKS: List[str] = [
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "lotka_volterra",
    "sir",
    "slcp",
    "slcp_distractors",
    "two_moons",
    "bernoulli_glm_raw",
]

ALGORITHMS: Dict[str, callable] = {
    "SMC-ABC (sbi)": smc_abc_sbi,
    "SMC-ABC (pyabc)": smc_abc_pyabc,
}

ALGORITHM_DEFAULTS: Dict[str, Dict] = {
    "SMC-ABC (sbi)": {
        "num_rounds": 1,
    },
    "SMC-ABC (pyabc)": {
        "num_rounds": 1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SMC-ABC interface survey on 10 standard tasks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./smcabc_survey_results",
        help="Directory to save survey results",
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="List of tasks to validate",
    )
    
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
        help="List of algorithms to validate",
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="Number of simulations per validation",
    )
    
    parser.add_argument(
        "--num-observation",
        type=int,
        default=1,
        help="Observation number to use",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    
    return parser.parse_args()


def get_log_level(level_str: str) -> int:
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }[level_str]


def extract_default_parameters(func) -> Dict[str, Any]:
    import inspect
    signature = inspect.signature(func)
    params = {}
    
    for name, param in signature.parameters.items():
        params[name] = {
            "name": name,
            "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "kind": str(param.kind),
        }
    
    return params


def main():
    args = parse_args()
    log_level = get_log_level(args.log_level)
    
    logger = setup_logger("SMC-ABC Survey", level=log_level)
    logger.info("=" * 80)
    logger.info("SMC-ABC INTERFACE SURVEY")
    logger.info("=" * 80)
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Algorithms: {', '.join(args.algorithms)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    validator = InterfaceValidator(
        output_dir=args.output_dir,
        log_level=log_level,
    )
    
    all_params = {}
    for algo_name, algo_func in ALGORITHMS.items():
        all_params[algo_name] = extract_default_parameters(algo_func)
    
    params_file = f"{args.output_dir}/smcabc_default_parameters.json"
    save_json_metadata(all_params, params_file, overwrite=True)
    logger.info(f"Default parameters saved to: {params_file}")
    
    total_validations = len(args.tasks) * len(args.algorithms)
    current_validation = 0
    
    for task_name in args.tasks:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PROCESSING TASK: {task_name}")
        logger.info(f"{'=' * 80}")
        
        try:
            task = sbibm.get_task(task_name)
            logger.info(f"✓ Loaded task: {task.name_display}")
            
            for algorithm_name in args.algorithms:
                current_validation += 1
                logger.info(f"\n--- Validation {current_validation}/{total_validations} ---")
                
                algorithm_func = ALGORITHMS[algorithm_name]
                algorithm_kwargs = ALGORITHM_DEFAULTS.get(algorithm_name, {})
                
                try:
                    validator.validate_algorithm(
                        algorithm_name=algorithm_name,
                        algorithm_func=algorithm_func,
                        task=task,
                        num_samples=args.num_samples,
                        num_simulations=args.num_simulations,
                        num_observation=args.num_observation,
                        **algorithm_kwargs,
                    )
                except Exception as e:
                    logger.error(f"✗ Failed to validate {algorithm_name} on {task_name}: {str(e)}")
                    import traceback
                    validator.results.append({
                        "algorithm": algorithm_name,
                        "task": task_name,
                        "task_display_name": task.name_display,
                        "timestamp": None,
                        "parameters": all_params.get(algorithm_name),
                        "validation": {
                            "success": False,
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                                "traceback": traceback.format_exc(),
                            }
                        },
                        "status": "error",
                    })
        
        except Exception as e:
            logger.error(f"✗ Failed to load task {task_name}: {str(e)}")
            import traceback
            for algorithm_name in args.algorithms:
                current_validation += 1
                validator.results.append({
                    "algorithm": algorithm_name,
                    "task": task_name,
                    "task_display_name": task_name,
                    "timestamp": None,
                    "parameters": all_params.get(algorithm_name),
                    "validation": {
                        "success": False,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    },
                    "status": "error",
                })
    
    logger.info("\n" + "=" * 80)
    logger.info("SMC-ABC SURVEY COMPLETE")
    logger.info("=" * 80)
    
    validator.print_summary()
    
    results_file = validator.save_results("smcabc_survey_results.json")
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    success_count = sum(1 for r in validator.results if r["status"] == "success")
    if success_count == len(validator.results):
        logger.info("\n✓ All validations succeeded!")
        return 0
    else:
        logger.warning(f"\n⚠️ Some validations failed ({len(validator.results) - success_count}/{len(validator.results)})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
