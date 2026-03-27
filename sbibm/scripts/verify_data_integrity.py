#!/usr/bin/env python3
"""
数据完整性验证脚本

验证以下内容:
1. results/SNPE/ 目录下所有任务的数据文件（samples.h5, metadata.json）
2. results/SNLE/ 目录下所有任务的数据文件（samples.h5, metadata.json）
3. results/timing.csv 文件存在且格式正确
4. HDF5 文件可以正常读取
5. 目录结构符合规范

生成验证报告保存到 results/data_integrity_report.json
"""

import os
import sys
import json
import csv
import h5py
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

# 导入 utils.py 中的工具函数
try:
    from sbibm.scripts.utils import (
        setup_logger,
        load_from_hdf5,
        load_json_metadata,
        measure_time,
    )
except ImportError:
    from utils import (
        setup_logger,
        load_from_hdf5,
        load_json_metadata,
        measure_time,
    )


@dataclass
class VerificationResult:
    """单个文件的验证结果"""
    file_path: str
    file_type: str  # 'hdf5', 'json', 'csv', 'directory'
    status: str  # 'passed', 'failed', 'skipped'
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataIntegrityReport:
    """数据完整性验证报告"""
    timestamp: str
    base_dir: str
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    skipped_checks: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "base_dir": self.base_dir,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "skipped_checks": self.skipped_checks,
            "results": self.results,
            "summary": self.summary,
        }


class DataIntegrityVerifier:
    """数据完整性验证器"""

    def __init__(self, base_dir: str = ".", log_level: int = 20):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.logger = setup_logger(
            "DataIntegrityVerifier",
            os.path.join(self.results_dir, "verify_data_integrity.log") if os.path.exists(self.results_dir) else None,
            level=log_level,
        )
        self.report = DataIntegrityReport(
            timestamp=datetime.now().isoformat(),
            base_dir=os.path.abspath(base_dir),
        )
        self.verification_results: List[VerificationResult] = []

    def verify_all(self) -> DataIntegrityReport:
        """执行所有验证检查"""
        self.logger.info("=" * 60)
        self.logger.info("开始数据完整性验证")
        self.logger.info("=" * 60)

        with measure_time(self.logger, "数据完整性验证"):
            # 1. 验证 SNPE 目录
            self._verify_method_directory("SNPE")

            # 2. 验证 SNLE 目录
            self._verify_method_directory("SNLE")

            # 3. 验证 timing.csv
            self._verify_timing_csv()

            # 4. 生成报告摘要
            self._generate_summary()

        # 保存报告
        self._save_report()

        return self.report

    def _verify_method_directory(self, method: str) -> None:
        """验证方法目录（SNPE/SNLE）"""
        method_dir = os.path.join(self.results_dir, method)

        self.logger.info(f"\n验证 {method} 目录: {method_dir}")

        # 检查方法目录是否存在
        if not os.path.exists(method_dir):
            result = VerificationResult(
                file_path=method_dir,
                file_type="directory",
                status="failed",
                message=f"{method} 目录不存在",
            )
            self.verification_results.append(result)
            self.logger.warning(f"✗ {method} 目录不存在: {method_dir}")
            return

        result = VerificationResult(
            file_path=method_dir,
            file_type="directory",
            status="passed",
            message=f"{method} 目录存在",
        )
        self.verification_results.append(result)
        self.logger.info(f"✓ {method} 目录存在")

        # 获取所有任务子目录
        try:
            task_dirs = [
                d for d in os.listdir(method_dir)
                if os.path.isdir(os.path.join(method_dir, d))
            ]
        except Exception as e:
            self.logger.error(f"无法列出 {method} 目录内容: {e}")
            return

        if not task_dirs:
            self.logger.warning(f"{method} 目录下没有任务子目录")
            return

        self.logger.info(f"发现 {len(task_dirs)} 个任务: {', '.join(task_dirs)}")

        # 验证每个任务的数据文件
        for task in task_dirs:
            task_dir = os.path.join(method_dir, task)
            self._verify_task_data(method, task, task_dir)

    def _verify_task_data(self, method: str, task: str, task_dir: str) -> None:
        """验证单个任务的数据文件"""
        self.logger.info(f"\n  验证任务: {method}/{task}")

        # 验证 samples.h5
        h5_path = os.path.join(task_dir, "samples.h5")
        self._verify_hdf5_file(h5_path, method, task)

        # 验证 metadata.json
        json_path = os.path.join(task_dir, "metadata.json")
        self._verify_json_file(json_path, method, task)

    def _verify_hdf5_file(self, file_path: str, method: str, task: str) -> None:
        """验证 HDF5 文件"""
        if not os.path.exists(file_path):
            result = VerificationResult(
                file_path=file_path,
                file_type="hdf5",
                status="failed",
                message=f"samples.h5 文件不存在 ({method}/{task})",
            )
            self.verification_results.append(result)
            self.logger.error(f"    ✗ samples.h5 不存在: {file_path}")
            return

        try:
            # 尝试读取 HDF5 文件
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                file_size = os.path.getsize(file_path)

                # 检查是否包含 samples 数据集
                has_samples = "samples" in keys

                details = {
                    "keys": keys,
                    "file_size_bytes": file_size,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "has_samples_dataset": has_samples,
                }

                if not has_samples:
                    result = VerificationResult(
                        file_path=file_path,
                        file_type="hdf5",
                        status="failed",
                        message=f"HDF5 文件缺少 'samples' 数据集 ({method}/{task})",
                        details=details,
                    )
                    self.verification_results.append(result)
                    self.logger.error(f"    ✗ HDF5 文件缺少 'samples' 数据集: {file_path}")
                    return

                # 尝试加载数据验证完整性
                data = load_from_hdf5(file_path)
                samples_shape = data.get("samples", []).shape if "samples" in data else None

                details["samples_shape"] = str(samples_shape) if samples_shape else None

                result = VerificationResult(
                    file_path=file_path,
                    file_type="hdf5",
                    status="passed",
                    message=f"HDF5 文件验证通过 ({method}/{task})",
                    details=details,
                )
                self.verification_results.append(result)
                self.logger.info(f"    ✓ samples.h5 验证通过 (keys: {keys}, size: {details['file_size_mb']} MB)")

        except Exception as e:
            result = VerificationResult(
                file_path=file_path,
                file_type="hdf5",
                status="failed",
                message=f"HDF5 文件读取失败: {str(e)} ({method}/{task})",
                details={"error": str(e)},
            )
            self.verification_results.append(result)
            self.logger.error(f"    ✗ HDF5 文件读取失败: {e}")

    def _verify_json_file(self, file_path: str, method: str, task: str) -> None:
        """验证 JSON 元数据文件"""
        if not os.path.exists(file_path):
            result = VerificationResult(
                file_path=file_path,
                file_type="json",
                status="failed",
                message=f"metadata.json 文件不存在 ({method}/{task})",
            )
            self.verification_results.append(result)
            self.logger.error(f"    ✗ metadata.json 不存在: {file_path}")
            return

        try:
            # 尝试加载 JSON 文件
            metadata = load_json_metadata(file_path)
            file_size = os.path.getsize(file_path)

            # 检查必要的字段
            required_fields = ["method", "task", "timestamp"]
            missing_fields = [f for f in required_fields if f not in metadata]

            details = {
                "keys": list(metadata.keys()),
                "file_size_bytes": file_size,
                "missing_required_fields": missing_fields,
            }

            if missing_fields:
                result = VerificationResult(
                    file_path=file_path,
                    file_type="json",
                    status="failed",
                    message=f"metadata.json 缺少必要字段: {missing_fields} ({method}/{task})",
                    details=details,
                )
                self.verification_results.append(result)
                self.logger.error(f"    ✗ metadata.json 缺少必要字段: {missing_fields}")
                return

            result = VerificationResult(
                file_path=file_path,
                file_type="json",
                status="passed",
                message=f"metadata.json 验证通过 ({method}/{task})",
                details=details,
            )
            self.verification_results.append(result)
            self.logger.info(f"    ✓ metadata.json 验证通过 (keys: {list(metadata.keys())})")

        except json.JSONDecodeError as e:
            result = VerificationResult(
                file_path=file_path,
                file_type="json",
                status="failed",
                message=f"metadata.json JSON 格式错误: {str(e)} ({method}/{task})",
                details={"error": str(e)},
            )
            self.verification_results.append(result)
            self.logger.error(f"    ✗ metadata.json JSON 格式错误: {e}")

        except Exception as e:
            result = VerificationResult(
                file_path=file_path,
                file_type="json",
                status="failed",
                message=f"metadata.json 读取失败: {str(e)} ({method}/{task})",
                details={"error": str(e)},
            )
            self.verification_results.append(result)
            self.logger.error(f"    ✗ metadata.json 读取失败: {e}")

    def _verify_timing_csv(self) -> None:
        """验证 timing.csv 文件"""
        csv_path = os.path.join(self.results_dir, "timing.csv")

        self.logger.info(f"\n验证 timing.csv: {csv_path}")

        if not os.path.exists(csv_path):
            result = VerificationResult(
                file_path=csv_path,
                file_type="csv",
                status="failed",
                message="timing.csv 文件不存在",
            )
            self.verification_results.append(result)
            self.logger.error(f"✗ timing.csv 不存在: {csv_path}")
            return

        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)

            file_size = os.path.getsize(csv_path)

            # 检查必要的列
            expected_columns = ["method", "task", "run1", "run2", "run3", "run4", "run5", "mean", "std"]
            missing_columns = [c for c in expected_columns if c not in fieldnames]

            details = {
                "fieldnames": fieldnames,
                "row_count": len(rows),
                "file_size_bytes": file_size,
                "missing_columns": missing_columns,
            }

            if missing_columns:
                result = VerificationResult(
                    file_path=csv_path,
                    file_type="csv",
                    status="failed",
                    message=f"timing.csv 缺少必要列: {missing_columns}",
                    details=details,
                )
                self.verification_results.append(result)
                self.logger.error(f"✗ timing.csv 缺少必要列: {missing_columns}")
                return

            result = VerificationResult(
                file_path=csv_path,
                file_type="csv",
                status="passed",
                message="timing.csv 验证通过",
                details=details,
            )
            self.verification_results.append(result)
            self.logger.info(f"✓ timing.csv 验证通过 (rows: {len(rows)}, columns: {fieldnames})")

        except Exception as e:
            result = VerificationResult(
                file_path=csv_path,
                file_type="csv",
                status="failed",
                message=f"timing.csv 读取失败: {str(e)}",
                details={"error": str(e)},
            )
            self.verification_results.append(result)
            self.logger.error(f"✗ timing.csv 读取失败: {e}")

    def _generate_summary(self) -> None:
        """生成验证报告摘要"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("生成验证报告摘要")
        self.logger.info("=" * 60)

        # 统计结果
        total = len(self.verification_results)
        passed = sum(1 for r in self.verification_results if r.status == "passed")
        failed = sum(1 for r in self.verification_results if r.status == "failed")
        skipped = sum(1 for r in self.verification_results if r.status == "skipped")

        self.report.total_checks = total
        self.report.passed_checks = passed
        self.report.failed_checks = failed
        self.report.skipped_checks = skipped

        # 转换结果为字典列表
        self.report.results = [asdict(r) for r in self.verification_results]

        # 生成摘要
        self.report.summary = {
            "verification_status": "PASSED" if failed == 0 else "FAILED",
            "total_files_checked": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": round(passed / total * 100, 2) if total > 0 else 0,
            "failed_files": [
                r.file_path for r in self.verification_results if r.status == "failed"
            ],
        }

        self.logger.info(f"总检查数: {total}")
        self.logger.info(f"通过: {passed}")
        self.logger.info(f"失败: {failed}")
        self.logger.info(f"跳过: {skipped}")
        self.logger.info(f"通过率: {self.report.summary['pass_rate']}%")

        if failed > 0:
            self.logger.warning(f"\n失败的文件:")
            for r in self.verification_results:
                if r.status == "failed":
                    self.logger.warning(f"  - {r.file_path}: {r.message}")

    def _save_report(self) -> None:
        """保存验证报告到 JSON 文件"""
        report_path = os.path.join(self.results_dir, "data_integrity_report.json")

        try:
            os.makedirs(self.results_dir, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.report.to_dict(), f, indent=4, ensure_ascii=False)
            self.logger.info(f"\n✓ 验证报告已保存: {report_path}")
        except Exception as e:
            self.logger.error(f"\n✗ 保存验证报告失败: {e}")


def main():
    """主函数"""
    # 确定基础目录（从 sbibm 目录开始）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # sbibm 目录

    # 创建验证器并执行验证
    verifier = DataIntegrityVerifier(base_dir=base_dir)
    report = verifier.verify_all()

    # 打印最终结果
    print("\n" + "=" * 60)
    print("数据完整性验证完成")
    print("=" * 60)
    print(f"验证状态: {report.summary.get('verification_status', 'UNKNOWN')}")
    print(f"总检查数: {report.total_checks}")
    print(f"通过: {report.passed_checks}")
    print(f"失败: {report.failed_checks}")
    print(f"通过率: {report.summary.get('pass_rate', 0)}%")
    print(f"\n报告保存位置: {os.path.join(base_dir, 'results', 'data_integrity_report.json')}")

    # 根据验证结果返回退出码
    if report.failed_checks > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
