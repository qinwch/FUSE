import argparse
import logging
import sys
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import sbibm
from sbibm.algorithms import snle
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

ALGORITHM_NAME: str = "SNLE"
ALGORITHM_FUNC = snle
ALGORITHM_DEFAULTS: Dict[str, Dict] = {
    "num_rounds": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SNLE survey for sbibm tasks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./snle_survey_results",
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


def save_default_parameters(output_dir: str, logger: logging.Logger) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_file = os.path.join(output_dir, f"snle_default_parameters_{timestamp}.json")
    
    default_params = {
        "algorithm": ALGORITHM_NAME,
        "timestamp": datetime.now().isoformat(),
        "default_kwargs": ALGORITHM_DEFAULTS,
        "num_samples": 10,
        "num_simulations": 100,
        "num_observation": 1,
    }
    
    save_json_metadata(default_params, params_file, overwrite=True)
    logger.info(f"Default parameters saved to: {params_file}")
    return params_file


def main():
    args = parse_args()
    log_level = get_log_level(args.log_level)
    
    logger = setup_logger("RunSNLESurvey", level=log_level)
    logger.info("=" * 80)
    logger.info("SBIBM SNLE SURVEY")
    logger.info("=" * 80)
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    validator = InterfaceValidator(
        output_dir=args.output_dir,
        log_level=log_level,
    )
    
    default_params_file = save_default_parameters(args.output_dir, logger)
    
    total_surveys = len(args.tasks)
    current_survey = 0
    
    for task_name in args.tasks:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PROCESSING TASK: {task_name}")
        logger.info(f"{'=' * 80}")
        
        try:
            task = sbibm.get_task(task_name)
            logger.info(f"✓ Loaded task: {task.name_display}")
            
            current_survey += 1
            logger.info(f"\n--- Survey {current_survey}/{total_surveys} ---")
            
            try:
                validator.validate_algorithm(
                    algorithm_name=ALGORITHM_NAME,
                    algorithm_func=ALGORITHM_FUNC,
                    task=task,
                    num_samples=args.num_samples,
                    num_simulations=args.num_simulations,
                    num_observation=args.num_observation,
                    **ALGORITHM_DEFAULTS,
                )
            except Exception as e:
                logger.error(f"✗ Failed to survey {ALGORITHM_NAME} on {task_name}: {str(e)}")
                validator.results.append({
                    "algorithm": ALGORITHM_NAME,
                    "task": task_name,
                    "task_display_name": task.name_display,
                    "timestamp": datetime.now().isoformat(),
                    "parameters": None,
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
            current_survey += 1
            validator.results.append({
                "algorithm": ALGORITHM_NAME,
                "task": task_name,
                "task_display_name": task_name,
                "timestamp": datetime.now().isoformat(),
                "parameters": None,
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
    logger.info("SURVEY COMPLETE")
    logger.info("=" * 80)
    
    validator.print_summary()
    
    results_file = validator.save_results(filename=f"snle_survey_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info(f"Default parameters saved to: {default_params_file}")
    
    success_count = sum(1 for r in validator.results if r["status"] == "success")
    if success_count == len(validator.results):
        logger.info("\n✓ All surveys succeeded!")
        return 0
    else:
        logger.warning(f"\n⚠️ Some surveys failed ({len(validator.results) - success_count}/{len(validator.results)})")
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main())
