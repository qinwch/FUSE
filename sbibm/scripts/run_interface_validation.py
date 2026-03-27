import argparse
import logging
import sys
from typing import Dict, List, Optional

import sbibm
from sbibm.algorithms import smc_abc, snle, snpe
from sbibm.tasks.task import Task

from sbibm.scripts.interface_validator import InterfaceValidator
from sbibm.scripts.utils import setup_logger


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
    "SMC-ABC": smc_abc,
    "SNPE": snpe,
    "SNLE": snle,
}

ALGORITHM_DEFAULTS: Dict[str, Dict] = {
    "SMC-ABC": {
        "num_rounds": 1,
    },
    "SNPE": {
        "num_rounds": 1,
    },
    "SNLE": {
        "num_rounds": 1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run interface validation for sbibm algorithms and tasks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_results",
        help="Directory to save validation results",
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


def main():
    args = parse_args()
    log_level = get_log_level(args.log_level)
    
    logger = setup_logger("RunInterfaceValidation", level=log_level)
    logger.info("=" * 80)
    logger.info("SBIBM INTERFACE VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Algorithms: {', '.join(args.algorithms)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    validator = InterfaceValidator(
        output_dir=args.output_dir,
        log_level=log_level,
    )
    
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
                    validator.results.append({
                        "algorithm": algorithm_name,
                        "task": task_name,
                        "task_display_name": task.name_display,
                        "timestamp": validator.results[-1]["timestamp"] if validator.results else None,
                        "parameters": None,
                        "validation": {
                            "success": False,
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                            }
                        },
                        "status": "error",
                    })
        
        except Exception as e:
            logger.error(f"✗ Failed to load task {task_name}: {str(e)}")
            for algorithm_name in args.algorithms:
                current_validation += 1
                validator.results.append({
                    "algorithm": algorithm_name,
                    "task": task_name,
                    "task_display_name": task_name,
                    "timestamp": None,
                    "parameters": None,
                    "validation": {
                        "success": False,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        }
                    },
                    "status": "error",
                })
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    
    validator.print_summary()
    
    results_file = validator.save_results()
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
