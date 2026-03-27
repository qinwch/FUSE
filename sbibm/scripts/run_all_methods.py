#!/usr/bin/env python3
"""
统一运行所有 SBI 方法调研任务的脚本
支持自选方法、任务和重复运行次数
"""

import argparse
import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback

# 添加脚本目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils import setup_logger, save_json_metadata, measure_time, save_to_hdf5
from timing_recorder import TimingRecorder

import sbibm
from sbibm.algorithms.sbi import snpe, snle, smcabc


# 所有可用的任务
ALL_TASKS = [
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

# 所有可用的方法
ALL_METHODS = {
    "snpe": snpe,
    "snle": snle,
    "smcabc": smcabc,
}

# 方法默认参数
METHOD_DEFAULT_PARAMS = {
    "snpe": {
        "num_samples": 1000,
        "num_simulations": 10000,
        "num_rounds": 10,
        "neural_net": "nsf",
        "hidden_features": 50,
    },
    "snle": {
        "num_samples": 1000,
        "num_simulations": 10000,
        "num_rounds": 10,
        "neural_net": "maf",
        "hidden_features": 50,
    },
    "smcabc": {
        "num_samples": 1000,
        "num_simulations": 10000,
        "num_observation": 1,
    },
}


class UnifiedRunner:
    """统一运行器，管理所有方法的调研任务"""

    def __init__(
        self,
        output_dir: str = "./unified_results",
        log_level: int = logging.INFO,
    ):
        self.output_dir = output_dir
        self.logger = setup_logger("UnifiedRunner", log_level)
        os.makedirs(output_dir, exist_ok=True)

        # 初始化时间记录器
        self.timing_recorder = TimingRecorder(output_dir=output_dir)

        # 运行记录
        self.run_history: List[Dict[str, Any]] = []

    def run_single_task(
        self,
        method_name: str,
        task_name: str,
        run_id: int,
        num_observation: int = 1,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """运行单个任务的一次迭代"""

        run_info = {
            "method": method_name,
            "task": task_name,
            "run_id": run_id,
            "num_observation": num_observation,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "error": None,
            "duration": 0.0,
        }

        try:
            self.logger.info(
                f"[{method_name}] Running {task_name} - Run {run_id}"
            )

            # 加载任务
            task = sbibm.get_task(task_name)

            # 获取方法函数和默认参数
            method_func = ALL_METHODS.get(method_name)
            if not method_func:
                raise ValueError(f"Unknown method: {method_name}")

            params = METHOD_DEFAULT_PARAMS.get(method_name, {}).copy()
            if custom_params:
                params.update(custom_params)

            # 记录开始时间
            start_time = time.time()

            # 运行方法
            if method_name == "smcabc":
                # SMC-ABC 特殊处理
                samples, num_simulations, log_prob = method_func(
                    task=task,
                    num_samples=params.get("num_samples", 1000),
                    num_simulations=params.get("num_simulations", 10000),
                    num_observation=num_observation,
                )
            else:
                # SNPE/SNLE
                samples, num_simulations, log_prob = method_func(
                    task=task,
                    num_samples=params.get("num_samples", 1000),
                    num_simulations=params.get("num_simulations", 10000),
                    num_observation=num_observation,
                    num_rounds=params.get("num_rounds", 10),
                    neural_net=params.get("neural_net", "nsf"),
                    hidden_features=params.get("hidden_features", 50),
                )

            # 计算运行时间
            duration = time.time() - start_time

            # 保存结果
            result_data = {
                "samples": samples,
                "num_simulations": num_simulations,
                "log_prob_true_parameters": log_prob,
                "method": method_name,
                "task": task_name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "params": params,
            }

            # 使用 DataCollector 保存
            save_dir = os.path.join(
                self.output_dir, method_name, task_name, f"run_{run_id}"
            )
            os.makedirs(save_dir, exist_ok=True)

            # 保存采样结果到 HDF5
            hdf5_path = os.path.join(save_dir, "samples.h5")
            save_to_hdf5({"samples": samples}, hdf5_path)

            # 保存元数据
            metadata_path = os.path.join(save_dir, "metadata.json")
            metadata = {
                "method": method_name,
                "task": task_name,
                "run_id": run_id,
                "num_observation": num_observation,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "num_simulations": num_simulations,
                "samples_shape": list(samples.shape) if hasattr(samples, 'shape') else None,
                "parameters": params,
            }
            save_json_metadata(metadata, metadata_path)

            # 更新运行信息
            run_info.update({
                "status": "success",
                "duration": duration,
                "samples_shape": list(samples.shape) if hasattr(samples, 'shape') else None,
                "save_dir": save_dir,
            })

            self.logger.info(
                f"[{method_name}] {task_name} Run {run_id} completed in {duration:.2f}s"
            )

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            run_info.update({
                "status": "failed",
                "error": error_msg,
            })
            self.logger.error(
                f"[{method_name}] {task_name} Run {run_id} failed: {str(e)}"
            )

        return run_info

    def run_method_tasks(
        self,
        method_name: str,
        tasks: List[str],
        num_runs: int = 5,
        num_observation: int = 1,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """运行一个方法在多个任务上的多次迭代"""

        results = []

        self.logger.info(f"=" * 60)
        self.logger.info(f"Running method: {method_name}")
        self.logger.info(f"Tasks: {tasks}")
        self.logger.info(f"Number of runs per task: {num_runs}")
        self.logger.info(f"=" * 60)

        for task_name in tasks:
            self.logger.info(f"\n{'-' * 40}")
            self.logger.info(f"Task: {task_name}")
            self.logger.info(f"{'-' * 40}")

            for run_id in range(1, num_runs + 1):
                run_info = self.run_single_task(
                    method_name=method_name,
                    task_name=task_name,
                    run_id=run_id,
                    num_observation=num_observation,
                    custom_params=custom_params,
                )
                results.append(run_info)
                self.run_history.append(run_info)

        return results

    def run_all(
        self,
        methods: List[str],
        tasks: List[str],
        num_runs: int = 5,
        num_observation: int = 1,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """运行所有选定的方法在选定的任务上"""

        overall_start = time.time()

        all_results = {}

        for method_name in methods:
            if method_name not in ALL_METHODS:
                self.logger.warning(f"Skipping unknown method: {method_name}")
                continue

            method_results = self.run_method_tasks(
                method_name=method_name,
                tasks=tasks,
                num_runs=num_runs,
                num_observation=num_observation,
                custom_params=custom_params.get(method_name) if custom_params else None,
            )
            all_results[method_name] = method_results

        overall_duration = time.time() - overall_start

        # 保存总体结果
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": overall_duration,
            "methods": methods,
            "tasks": tasks,
            "num_runs_per_task": num_runs,
            "total_runs": len(methods) * len(tasks) * num_runs,
            "successful_runs": sum(
                1 for r in self.run_history if r["status"] == "success"
            ),
            "failed_runs": sum(
                1 for r in self.run_history if r["status"] == "failed"
            ),
            "results": all_results,
        }

        # 保存汇总报告
        summary_path = os.path.join(self.output_dir, "unified_run_summary.json")
        save_json_metadata(summary, summary_path)

        # 生成 CSV 时间报告
        self._generate_timing_csv()

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"All runs completed!")
        self.logger.info(f"Total duration: {overall_duration:.2f}s")
        self.logger.info(f"Successful: {summary['successful_runs']}/{summary['total_runs']}")
        self.logger.info(f"Failed: {summary['failed_runs']}/{summary['total_runs']}")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"{'=' * 60}")

        return summary

    def _generate_timing_csv(self):
        """生成时间统计 CSV 文件"""

        import csv

        csv_path = os.path.join(self.output_dir, "timing_summary.csv")

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "method", "task", "run_id", "status", "duration_seconds",
                "timestamp", "error"
            ])

            for run_info in self.run_history:
                writer.writerow([
                    run_info.get("method", ""),
                    run_info.get("task", ""),
                    run_info.get("run_id", ""),
                    run_info.get("status", ""),
                    f"{run_info.get('duration', 0):.4f}",
                    run_info.get("timestamp", ""),
                    str(run_info.get("error", ""))[:100] if run_info.get("error") else "",
                ])

        self.logger.info(f"Timing summary saved to: {csv_path}")


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description="统一运行所有 SBI 方法调研任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有方法在所有任务上，每个任务5次
  python run_all_methods.py

  # 只运行 SNPE 和 SNLE 方法
  python run_all_methods.py --methods snpe snle

  # 只运行特定任务
  python run_all_methods.py --tasks bernoulli_glm gaussian_linear

  # 每个任务运行10次
  python run_all_methods.py --num-runs 10

  # 自定义输出目录
  python run_all_methods.py --output-dir ./my_results
        """
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(ALL_METHODS.keys()) + ["all"],
        default=["all"],
        help="选择要运行的方法 (默认: all)",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS + ["all"],
        default=["all"],
        help="选择要运行的任务 (默认: all)",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="每个任务的重复运行次数 (默认: 5)",
    )

    parser.add_argument(
        "--num-observation",
        type=int,
        default=1,
        help="使用的观测值编号 (默认: 1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./unified_results",
        help="结果输出目录 (默认: ./unified_results)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 设置日志级别
    log_level = getattr(logging, args.log_level)

    # 处理方法选择
    if "all" in args.methods:
        selected_methods = list(ALL_METHODS.keys())
    else:
        selected_methods = args.methods

    # 处理任务选择
    if "all" in args.tasks:
        selected_tasks = ALL_TASKS
    else:
        selected_tasks = args.tasks

    # 创建运行器并执行
    runner = UnifiedRunner(output_dir=args.output_dir, log_level=log_level)

    summary = runner.run_all(
        methods=selected_methods,
        tasks=selected_tasks,
        num_runs=args.num_runs,
        num_observation=args.num_observation,
    )

    # 打印最终摘要
    print("\n" + "=" * 60)
    print("运行完成摘要")
    print("=" * 60)
    print(f"方法: {', '.join(selected_methods)}")
    print(f"任务: {', '.join(selected_tasks)}")
    print(f"每个任务运行次数: {args.num_runs}")
    print(f"总运行次数: {summary['total_runs']}")
    print(f"成功: {summary['successful_runs']}")
    print(f"失败: {summary['failed_runs']}")
    print(f"总耗时: {summary['total_duration_seconds']:.2f}s")
    print(f"结果目录: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
