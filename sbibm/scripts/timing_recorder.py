import csv
import time
import os
from typing import Callable, List, Dict, Any
import numpy as np

try:
    from .utils import setup_logger
except ImportError:
    from utils import setup_logger


class TimingRecorder:
    def __init__(
        self, output_dir: str = ".", logger_name: str = "TimingRecorder"
    ):
        self.output_dir = output_dir
        self.logger = setup_logger(logger_name)
        os.makedirs(output_dir, exist_ok=True)

    def measure_function(
        self, func: Callable, *args, num_runs: int = 5, **kwargs
    ) -> List[float]:
        times = []
        for i in range(num_runs):
            self.logger.info(f"Starting run {i+1}/{num_runs}")
            try:
                start_time = time.time()
                func(*args, **kwargs)
                elapsed = time.time() - start_time
                times.append(elapsed)
                self.logger.info(
                    f"Run {i+1}/{num_runs} completed in {elapsed:.4f} seconds"
                )
            except Exception as e:
                self.logger.error(f"Error during run {i+1}/{num_runs}: {e}")
                raise
        return times

    def calculate_statistics(self, times: List[float]) -> Dict[str, float]:
        if len(times) == 0:
            return {"mean": 0.0, "std": 0.0}
        mean_time = float(np.mean(times))
        std_time = float(np.std(times))
        return {"mean": mean_time, "std": std_time}

    def save_to_csv(
        self, data: List[Dict[str, Any]], filename: str = "timing.csv"
    ):
        file_path = os.path.join(self.output_dir, filename)

        if not data:
            self.logger.warning("No data to save to CSV")
            return

        try:
            fieldnames = [
                "method", "task", "run1", "run2", "run3",
                "run4", "run5", "mean", "std"
            ]
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    csv_row = {
                        "method": row["method"],
                        "task": row["task"]
                    }
                    for i, t in enumerate(row["times"]):
                        csv_row[f"run{i+1}"] = t
                    csv_row["mean"] = row["mean"]
                    csv_row["std"] = row["std"]
                    writer.writerow(csv_row)
            self.logger.info(f"Timing data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save CSV file: {e}")
            raise

    def record(
        self, method: str, task: str, func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        times = self.measure_function(func, *args, num_runs=5, **kwargs)
        stats = self.calculate_statistics(times)

        result = {
            "method": method,
            "task": task,
            "times": times,
            "mean": stats["mean"],
            "std": stats["std"]
        }

        self.logger.info(f"Method: {method}, Task: {task}")
        self.logger.info(
            f"Mean time: {stats['mean']:.4f}s, Std: {stats['std']:.4f}s"
        )

        return result
