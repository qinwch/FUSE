#!/usr/bin/env python3
"""
执行 SNLE 方法的数据收集脚本

对多个任务执行5次独立运行，保存模型检查点、采样结果和元数据，
并记录时间性能。
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import sbibm
from sbibm.algorithms import snle
from sbibm.tasks.task import Task

from data_collector import DataCollector
from timing_recorder import TimingRecorder
from utils import setup_logger


# 可用任务列表（排除已知有问题的任务）
AVAILABLE_TASKS: List[str] = [
    "bernoulli_glm",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "slcp",
    "two_moons",
    "bernoulli_glm_raw",
]

# 已知有问题的任务（可选跳过或记录失败）
PROBLEMATIC_TASKS: List[str] = [
    "lotka_volterra",
    "sir",
    "slcp_distractors",
]

METHOD_NAME: str = "SNLE"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="执行 SNLE 数据收集"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="基础目录，用于保存检查点和结果",
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=AVAILABLE_TASKS,
        choices=AVAILABLE_TASKS + PROBLEMATIC_TASKS,
        help="要执行的任务列表",
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="每个任务的独立运行次数",
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="每次运行生成的样本数",
    )
    
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10000,
        help="每次运行的模拟预算",
    )
    
    parser.add_argument(
        "--num-observation",
        type=int,
        default=1,
        help="使用的观测编号",
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="SNLE 的轮数（1 表示 NLE）",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别",
    )
    
    parser.add_argument(
        "--skip-problematic",
        action="store_true",
        default=True,
        help="跳过已知有问题的任务",
    )
    
    return parser.parse_args()


def get_log_level(level_str: str) -> int:
    """将日志级别字符串转换为 logging 常量"""
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }[level_str]


def run_snle_once(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: int,
    num_rounds: int,
    logger: logging.Logger,
) -> tuple:
    """
    执行一次 SNLE 运行
    
    Args:
        task: 任务实例
        num_samples: 样本数
        num_simulations: 模拟预算
        num_observation: 观测编号
        num_rounds: 轮数
        logger: 日志记录器
        
    Returns:
        (samples, num_simulations_used, log_prob_true)
    """
    logger.info(f"Running SNLE with {num_rounds} round(s)")
    
    samples, num_simulations_used, log_prob_true = snle(
        task=task,
        num_samples=num_samples,
        num_simulations=num_simulations,
        num_observation=num_observation,
        num_rounds=num_rounds,
    )
    
    return samples, num_simulations_used, log_prob_true


def execute_task_runs(
    task_name: str,
    num_runs: int,
    num_samples: int,
    num_simulations: int,
    num_observation: int,
    num_rounds: int,
    base_dir: str,
    logger: logging.Logger,
) -> Dict:
    """
    对单个任务执行多次运行
    
    Args:
        task_name: 任务名称
        num_runs: 运行次数
        num_samples: 样本数
        num_simulations: 模拟预算
        num_observation: 观测编号
        num_rounds: 轮数
        base_dir: 基础目录
        logger: 日志记录器
        
    Returns:
        包含运行结果的字典
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"处理任务: {task_name}")
    logger.info(f"{'='*80}")
    
    results = {
        "task": task_name,
        "method": METHOD_NAME,
        "num_runs": num_runs,
        "num_samples": num_samples,
        "num_simulations": num_simulations,
        "num_observation": num_observation,
        "num_rounds": num_rounds,
        "runs": [],
        "status": "pending",
        "error": None,
    }
    
    try:
        # 加载任务
        task = sbibm.get_task(task_name)
        logger.info(f"✓ 加载任务成功: {task.name_display}")
        
        # 创建 DataCollector
        data_collector = DataCollector(
            method=METHOD_NAME,
            task=task_name,
            base_dir=base_dir,
            log_level=logger.level,
        )
        
        all_samples = []
        all_times = []
        
        for run_idx in range(num_runs):
            logger.info(f"\n--- 运行 {run_idx + 1}/{num_runs} ---")
            
            run_data = {
                "run_index": run_idx,
                "status": "pending",
                "error": None,
            }
            
            try:
                # 记录开始时间
                start_time = datetime.now()
                
                # 执行 SNLE
                samples, num_simulations_used, log_prob_true = run_snle_once(
                    task=task,
                    num_samples=num_samples,
                    num_simulations=num_simulations,
                    num_observation=num_observation,
                    num_rounds=num_rounds,
                    logger=logger,
                )
                
                # 计算运行时间
                end_time = datetime.now()
                elapsed_time = (end_time - start_time).total_seconds()
                
                logger.info(f"✓ 运行 {run_idx + 1} 完成")
                logger.info(f"  - 样本数: {samples.shape}")
                logger.info(f"  - 使用模拟数: {num_simulations_used}")
                logger.info(f"  - 运行时间: {elapsed_time:.4f} 秒")
                
                # 转换为 numpy 数组
                if isinstance(samples, torch.Tensor):
                    samples_np = samples.detach().cpu().numpy()
                else:
                    samples_np = np.array(samples)
                
                all_samples.append(samples_np)
                all_times.append(elapsed_time)
                
                run_data.update({
                    "status": "success",
                    "samples_shape": list(samples_np.shape),
                    "num_simulations_used": num_simulations_used,
                    "elapsed_time": elapsed_time,
                    "log_prob_true": log_prob_true.item() if log_prob_true is not None else None,
                })
                
            except Exception as e:
                logger.error(f"✗ 运行 {run_idx + 1} 失败: {str(e)}")
                logger.error(traceback.format_exc())
                run_data.update({
                    "status": "error",
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                })
            
            results["runs"].append(run_data)
        
        # 保存所有成功的样本
        successful_runs = [r for r in results["runs"] if r["status"] == "success"]
        if successful_runs:
            # 合并所有样本
            combined_samples = np.vstack(all_samples)
            
            # 保存样本到 HDF5
            try:
                samples_path = data_collector.save_samples(
                    samples=combined_samples,
                    additional_data={
                        "num_runs": len(successful_runs),
                        "num_samples_per_run": num_samples,
                        "task": task_name,
                        "method": METHOD_NAME,
                    },
                    filename="samples.h5",
                    overwrite=True,
                )
                logger.info(f"✓ 样本已保存到: {samples_path}")
            except Exception as e:
                logger.error(f"✗ 保存样本失败: {e}")
            
            # 保存元数据
            try:
                metadata = {
                    "task": task_name,
                    "method": METHOD_NAME,
                    "num_runs": num_runs,
                    "num_successful_runs": len(successful_runs),
                    "num_samples": num_samples,
                    "num_simulations": num_simulations,
                    "num_observation": num_observation,
                    "num_rounds": num_rounds,
                    "run_times": all_times,
                    "mean_time": float(np.mean(all_times)) if all_times else 0.0,
                    "std_time": float(np.std(all_times)) if all_times else 0.0,
                    "runs": results["runs"],
                }
                metadata_path = data_collector.save_metadata(
                    metadata=metadata,
                    filename="metadata.json",
                    overwrite=True,
                )
                logger.info(f"✓ 元数据已保存到: {metadata_path}")
            except Exception as e:
                logger.error(f"✗ 保存元数据失败: {e}")
            
            results["status"] = "success" if len(successful_runs) == num_runs else "partial"
        else:
            results["status"] = "error"
            results["error"] = "所有运行都失败了"
        
        # 生成 README
        try:
            readme_path = data_collector.generate_readme(
                description=f"SNLE results for {task_name} with {num_runs} runs",
                additional_info={
                    "num_runs": str(num_runs),
                    "num_samples": str(num_samples),
                    "num_simulations": str(num_simulations),
                    "num_rounds": str(num_rounds),
                },
            )
            logger.info(f"✓ README 已生成: {readme_path}")
        except Exception as e:
            logger.error(f"✗ 生成 README 失败: {e}")
        
    except Exception as e:
        logger.error(f"✗ 处理任务 {task_name} 失败: {str(e)}")
        logger.error(traceback.format_exc())
        results["status"] = "error"
        results["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
    
    return results


def main():
    """主函数"""
    args = parse_args()
    log_level = get_log_level(args.log_level)
    
    # 设置主日志记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.base_dir, "results", METHOD_NAME, f"execute_snle_{timestamp}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = setup_logger("ExecuteSNLE", log_file=log_file, level=log_level)
    
    logger.info("=" * 80)
    logger.info("SNLE 数据收集执行")
    logger.info("=" * 80)
    logger.info(f"基础目录: {args.base_dir}")
    logger.info(f"任务列表: {', '.join(args.tasks)}")
    logger.info(f"运行次数: {args.num_runs}")
    logger.info(f"样本数: {args.num_samples}")
    logger.info(f"模拟预算: {args.num_simulations}")
    logger.info(f"轮数: {args.num_rounds}")
    logger.info("=" * 80)
    
    # 过滤任务
    tasks_to_run = args.tasks
    if args.skip_problematic:
        tasks_to_run = [t for t in tasks_to_run if t not in PROBLEMATIC_TASKS]
        logger.info(f"跳过有问题的任务: {PROBLEMATIC_TASKS}")
    
    logger.info(f"将执行的任务: {', '.join(tasks_to_run)}")
    
    # 创建 TimingRecorder
    timing_recorder = TimingRecorder(
        output_dir=os.path.join(args.base_dir, "results"),
        logger_name="SNLE_Timing",
    )
    
    all_results = []
    all_timing_data = []
    
    # 执行每个任务
    for task_idx, task_name in enumerate(tasks_to_run):
        logger.info(f"\n{'='*80}")
        logger.info(f"进度: {task_idx + 1}/{len(tasks_to_run)} - {task_name}")
        logger.info(f"{'='*80}")
        
        # 执行任务运行
        task_results = execute_task_runs(
            task_name=task_name,
            num_runs=args.num_runs,
            num_samples=args.num_samples,
            num_simulations=args.num_simulations,
            num_observation=args.num_observation,
            num_rounds=args.num_rounds,
            base_dir=args.base_dir,
            logger=logger,
        )
        
        all_results.append(task_results)
        
        # 收集时间数据用于 CSV
        successful_runs = [r for r in task_results["runs"] if r["status"] == "success"]
        if successful_runs:
            times = [r["elapsed_time"] for r in successful_runs]
            # 补齐到5次运行
            while len(times) < 5:
                times.append(0.0)
            
            timing_data = {
                "method": METHOD_NAME,
                "task": task_name,
                "times": times[:5],
                "mean": float(np.mean([r["elapsed_time"] for r in successful_runs])),
                "std": float(np.std([r["elapsed_time"] for r in successful_runs])),
            }
            all_timing_data.append(timing_data)
    
    # 保存时间数据到 CSV
    if all_timing_data:
        try:
            timing_recorder.save_to_csv(
                data=all_timing_data,
                filename="timing.csv",
            )
            logger.info(f"✓ 时间数据已保存到 results/timing.csv")
        except Exception as e:
            logger.error(f"✗ 保存时间数据失败: {e}")
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("执行总结")
    logger.info("=" * 80)
    
    for result in all_results:
        task_name = result["task"]
        status = result["status"]
        successful_count = sum(1 for r in result["runs"] if r["status"] == "success")
        
        status_icon = "✓" if status == "success" else "⚠️" if status == "partial" else "✗"
        logger.info(f"{status_icon} {task_name}: {successful_count}/{args.num_runs} 次运行成功")
    
    logger.info("=" * 80)
    
    # 返回退出码
    all_success = all(r["status"] == "success" for r in all_results)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
