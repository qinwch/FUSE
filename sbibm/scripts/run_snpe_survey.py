import argparse
import logging
import sys
from typing import Dict, List, Optional

import sbibm
from sbibm.algorithms import snpe
from sbibm.tasks.task import Task

from sbibm.scripts.interface_validator import InterfaceValidator
from sbibm.scripts.utils import setup_logger, save_json_metadata


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SNPE survey on sbibm tasks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./snpe_survey_results",
        help="Directory to save survey results",
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="List of tasks to survey",
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
        help="Number of simulations per survey",
    )
    
    parser.add_argument(
        "--num-observation",
        type=int,
        default=1,
        help="Observation number to use",
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of rounds for SNPE",
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


def save_snpe_default_parameters(output_dir: str, logger: logging.Logger) -> str:
    import inspect
    from sbibm.algorithms.sbi.snpe import run as snpe_run
    
    signature = inspect.signature(snpe_run)
    params = {}
    
    for name, param in signature.parameters.items():
        param_info = {
            "name": name,
            "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "kind": str(param.kind),
        }
        params[name] = param_info
    
    filename = "snpe_default_parameters.json"
    filepath = f"{output_dir}/{filename}"
    
    save_json_metadata(
        {
            "algorithm": "SNPE",
            "timestamp": str(__import__('datetime').datetime.now().isoformat()),
            "parameters": params,
        },
        filepath,
        overwrite=True,
    )
    
    logger.info(f"SNPE default parameters saved to: {filepath}")
    return filepath


def main():
    args = parse_args()
    log_level = get_log_level(args.log_level)
    
    logger = setup_logger("RunSNPEsurvey", level=log_level)
    logger.info("=" * 80)
    logger.info("SNPE SURVEY ON SBIBM TASKS")
    logger.info("=" * 80)
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_snpe_default_parameters(args.output_dir, logger)
    
    validator = InterfaceValidator(
        output_dir=args.output_dir,
        log_level=log_level,
    )
    
    algorithm_name = "SNPE"
    algorithm_func = snpe
    algorithm_kwargs = {
        "num_rounds": args.num_rounds,
    }
    
    total_surveys = len(args.tasks)
    current_survey = 0
    
    for task_name in args.tasks:
        current_survey += 1
        logger.info(f"\n{'=' * 80}")
        logger.info(f"SURVEYING TASK {current_survey}/{total_surveys}: {task_name}")
        logger.info(f"{'=' * 80}")
        
        try:
            task = sbibm.get_task(task_name)
            logger.info(f"✓ Loaded task: {task.name_display}")
            
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
                logger.error(f"✗ Failed to survey {algorithm_name} on {task_name}: {str(e)}")
                validator.results.append({
                    "algorithm": algorithm_name,
                    "task": task_name,
                    "task_display_name": task.name_display,
                    "timestamp": str(__import__('datetime').datetime.now().isoformat()),
                    "parameters": None,
                    "validation": {
                        "success": False,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": __import__('traceback').format_exc(),
                        }
                    },
                    "status": "error",
                })
        
        except Exception as e:
            logger.error(f"✗ Failed to load task {task_name}: {str(e)}")
            current_survey += 1
            validator.results.append({
                "algorithm": algorithm_name,
                "task": task_name,
                "task_display_name": task_name,
                "timestamp": str(__import__('datetime').datetime.now().isoformat()),
                "parameters": None,
                "validation": {
                    "success": False,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": __import__('traceback').format_exc(),
                    }
                },
                "status": "error",
            })
    
    logger.info("\n" + "=" * 80)
    logger.info("SNPE SURVEY COMPLETE")
    logger.info("=" * 80)
    
    validator.print_summary()
    
    results_file = validator.save_results("snpe_survey_results.json")
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    success_count = sum(1 for r in validator.results if r["status"] == "success")
    if success_count == len(validator.results):
        logger.info("\n✓ All surveys succeeded!")
        return 0
    else:
        logger.warning(f"\n⚠️ Some surveys failed ({len(validator.results) - success_count}/{len(validator.results)})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
