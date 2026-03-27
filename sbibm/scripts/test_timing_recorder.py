import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timing_recorder import TimingRecorder


def sample_function(seconds: float = 1.0):
    time.sleep(seconds)


def main():
    recorder = TimingRecorder(output_dir="/tmp/timing_test")

    results = []

    result1 = recorder.record(
        method="test_method_1",
        task="sample_task",
        func=sample_function,
        seconds=0.1
    )
    results.append(result1)

    result2 = recorder.record(
        method="test_method_2",
        task="sample_task",
        func=sample_function,
        seconds=0.2
    )
    results.append(result2)

    recorder.save_to_csv(results)


if __name__ == "__main__":
    main()
