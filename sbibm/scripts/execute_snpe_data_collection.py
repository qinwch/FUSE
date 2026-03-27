#!/usr/bin/env python3
"""
Execute SNPE data collection on sbibm tasks.

This script runs SNPE method on specified tasks for 5 independent runs,
saving checkpoints, samples, metadata, and timing information.
"""

import argparse
import logging
import sys
import os
import traceback
from typing import List, Dict, Any, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import sbibm
from sbibm.algorithms import snpe
from sbibm.tasks.task import Task

from scripts.data_collector import DataCollector
from scripts.timing_recorder import TimingRecorder
from scripts.utils import setup_logger, measure_time


# Available tasks for SNPE (excluding known problematic ones)
AVAILABLE_TASKS: List[str] = [
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "slcp",
    "two_moons",
    "bernoulli_glm_raw",
]

# Known problematic tasks (will be skipped or logged with failure reason)
PROBLEMATIC_TASKS: List[str] = [
    "lotka_volterra",
    "sir",
    "slcp_distractors",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute SNPE data collection on sbibm tasks"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for checkpoints and results",
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=AVAILABLE_TASKS,
        choices=AVAILABLE_TASKS + PROBLEMATIC_TASKS,
        help="List of tasks to run SNPE on",
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of independent runs per task",
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate from posterior",
    )
    
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10000,
        help="Total simulation budget",
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
        default=10,
        help="Number of rounds for SNPE",
    )
    
    parser.add_argument(
        "--neural-net",
        type=str,
        default="nsf",
        choices=["maf", "mdn", "made", "nsf"],
        help="Neural network type for SNPE",
    )
    
    parser.add_argument(
        "--hidden-features",
        type=int,
        default=50,
        help="Number of hidden features in network",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    
    parser.add_argument(
        "--skip-problematic",
        action="store_true",
        default=True,
        help="Skip known problematic tasks",
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


def run_snpe_on_task(
    task_name: str,
    run_id: int,
    args,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Run SNPE on a single task for one run.
    
    Args:
        task_name: Name of the task
        run_id: ID of the current run (1-indexed)
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing run results and metadata
    """
    logger.info(f"Starting run {run_id}/{args.num_runs} for task: {task_name}")
    
    # Initialize data collector for this task
    data_collector = DataCollector(
        method="SNPE",
        task=task_name,
        base_dir=args.base_dir,
        log_level=get_log_level(args.log_level),
    )
    
    # Load task
    task = sbibm.get_task(task_name)
    logger.info(f"Loaded task: {task.name_display}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run SNPE
        samples, num_simulations, log_prob_true = snpe(
            task=task,
            num_samples=args.num_samples,
            num_simulations=args.num_simulations,
            num_observation=args.num_observation,
            num_rounds=args.num_rounds,
            neural_net=args.neural_net,
            hidden_features=args.hidden_features,
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Convert samples to numpy array
        if isinstance(samples, torch.Tensor):
            samples_np = samples.detach().cpu().numpy()
        else:
            samples_np = np.array(samples)
        
        # Ensure samples is 2D
        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(-1, 1)
        
        # Save samples
        samples_path = data_collector.save_samples(
            samples=samples_np,
            additional_data={
                "num_simulations": np.array([num_simulations]),
                "log_prob_true_parameters": np.array([log_prob_true.item()]) if log_prob_true is not None else np.array([0.0]),
                "run_id": np.array([run_id]),
            },
            filename="samples.h5",
            overwrite=True,
        )
        
        # Prepare metadata
        metadata = {
            "run_id": run_id,
            "task": task_name,
            "task_display_name": task.name_display,
            "num_samples": args.num_samples,
            "num_simulations": num_simulations,
            "num_observation": args.num_observation,
            "num_rounds": args.num_rounds,
            "neural_net": args.neural_net,
            "hidden_features": args.hidden_features,
            "elapsed_time": elapsed_time,
            "log_prob_true_parameters": log_prob_true.item() if log_prob_true is not None else None,
            "samples_shape": list(samples_np.shape),
            "status": "success",
        }
        
        # Save metadata
        metadata_path = data_collector.save_metadata(
            metadata=metadata,
            filename="metadata.json",
            overwrite=True,
        )
        
        logger.info(f"✓ Run {run_id} completed successfully in {elapsed_time:.2f}s")
        logger.info(f"  Samples shape: {samples_np.shape}")
        logger.info(f"  Saved to: {samples_path}")
        
        return {
            "success": True,
            "elapsed_time": elapsed_time,
            "num_simulations": num_simulations,
            "samples_shape": samples_np.shape,
            "error": None,
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        logger.error(f"✗ Run {run_id} failed: {error_msg}")
        logger.debug(error_traceback)
        
        # Save failure metadata
        metadata = {
            "run_id": run_id,
            "task": task_name,
            "task_display_name": task.name_display,
            "num_samples": args.num_samples,
            "num_simulations": args.num_simulations,
            "num_observation": args.num_observation,
            "num_rounds": args.num_rounds,
            "neural_net": args.neural_net,
            "hidden_features": args.hidden_features,
            "elapsed_time": elapsed_time,
            "status": "failed",
            "error": error_msg,
            "traceback": error_traceback,
        }
        
        try:
            data_collector.save_metadata(
                metadata=metadata,
                filename="metadata.json",
                overwrite=True,
            )
        except Exception as save_error:
            logger.error(f"Failed to save error metadata: {save_error}")
        
        return {
            "success": False,
            "elapsed_time": elapsed_time,
            "num_simulations": 0,
            "samples_shape": None,
            "error": error_msg,
            "traceback": error_traceback,
        }


def run_multiple_runs(
    task_name: str,
    args,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Run SNPE multiple times on a task and collect timing information.
    
    Args:
        task_name: Name of the task
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        List of run results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running SNPE on task: {task_name}")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"{'=' * 80}")
    
    results = []
    times = []
    
    for run_id in range(1, args.num_runs + 1):
        run_result = run_snpe_on_task(task_name, run_id, args, logger)
        results.append(run_result)
        
        if run_result["success"]:
            times.append(run_result["elapsed_time"])
    
    return results, times


def main():
    args = parse_args()
    log_level = get_log_level(args.log_level)
    
    # Setup main logger
    logger = setup_logger("SNPE_DataCollection", level=log_level)
    
    logger.info("=" * 80)
    logger.info("SNPE DATA COLLECTION")
    logger.info("=" * 80)
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Number of runs per task: {args.num_runs}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Number of simulations: {args.num_simulations}")
    logger.info(f"Number of rounds: {args.num_rounds}")
    logger.info(f"Neural net: {args.neural_net}")
    logger.info(f"Hidden features: {args.hidden_features}")
    logger.info("=" * 80)
    
    # Filter out problematic tasks if requested
    tasks_to_run = args.tasks
    if args.skip_problematic:
        tasks_to_run = [t for t in tasks_to_run if t not in PROBLEMATIC_TASKS]
        skipped = [t for t in args.tasks if t in PROBLEMATIC_TASKS]
        if skipped:
            logger.warning(f"Skipping known problematic tasks: {', '.join(skipped)}")
    
    # Initialize timing recorder
    timing_recorder = TimingRecorder(
        output_dir=f"{args.base_dir}/results",
        logger_name="SNPE_Timing",
    )
    
    # Store all timing data
    all_timing_data = []
    
    # Track overall statistics
    total_tasks = len(tasks_to_run)
    completed_tasks = 0
    failed_tasks = []
    
    # Run SNPE on each task
    for task_idx, task_name in enumerate(tasks_to_run, 1):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"Task {task_idx}/{total_tasks}: {task_name}")
        logger.info(f"{'#' * 80}")
        
        try:
            # Run multiple times
            results, times = run_multiple_runs(task_name, args, logger)
            
            # Calculate statistics
            success_count = sum(1 for r in results if r["success"])
            
            if times:
                stats = timing_recorder.calculate_statistics(times)
                
                # Prepare timing data for CSV
                timing_data = {
                    "method": "SNPE",
                    "task": task_name,
                    "times": times + [0.0] * (5 - len(times)),  # Pad to 5 runs
                    "mean": stats["mean"],
                    "std": stats["std"],
                }
                all_timing_data.append(timing_data)
                
                logger.info(f"\nTiming statistics for {task_name}:")
                logger.info(f"  Mean: {stats['mean']:.4f}s")
                logger.info(f"  Std: {stats['std']:.4f}s")
                logger.info(f"  Successful runs: {success_count}/{args.num_runs}")
            else:
                logger.warning(f"No successful runs for {task_name}")
                failed_tasks.append(task_name)
            
            if success_count > 0:
                completed_tasks += 1
            else:
                failed_tasks.append(task_name)
                
        except Exception as e:
            logger.error(f"Failed to process task {task_name}: {e}")
            logger.debug(traceback.format_exc())
            failed_tasks.append(task_name)
    
    # Save timing data to CSV
    if all_timing_data:
        timing_csv_path = f"{args.base_dir}/results/timing.csv"
        timing_recorder.save_to_csv(all_timing_data, filename="timing.csv")
        logger.info(f"\nTiming data saved to: {timing_csv_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SNPE DATA COLLECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Completed tasks: {completed_tasks}")
    logger.info(f"Failed tasks: {len(failed_tasks)}")
    
    if failed_tasks:
        logger.warning(f"Failed tasks: {', '.join(failed_tasks)}")
    
    logger.info("\nOutput directories:")
    logger.info(f"  Checkpoints: {args.base_dir}/checkpoints/SNPE/")
    logger.info(f"  Results: {args.base_dir}/results/SNPE/")
    logger.info(f"  Timing: {args.base_dir}/results/timing.csv")
    logger.info("=" * 80)
    
    return 0 if len(failed_tasks) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
