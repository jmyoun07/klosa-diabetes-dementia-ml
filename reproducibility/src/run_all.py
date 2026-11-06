# -*- coding: utf-8 -*-
"""Run the whole reproduction in order and write a reproducibility record."""
import json
import os
import subprocess
import sys
import time

import common as C

STEPS = [
    ("Table 3  (single 70:30 split, secondary)", "run_01_single_split.py"),
    ("Table 5  (repeated cross-validation, PRIMARY)", "run_02_repeated_cv.py"),
    ("Table S8 (out-of-fold calibration)", "run_03_calibration.py"),
    ("Figure 2 (ROC curves)", "run_04_figure2.py"),
]

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    env = C.check_environment()
    record = {"environment": env, "seed": C.SEED, "data_path": C.DATA_PATH, "steps": []}

    for title, script in STEPS:
        print("=" * 78)
        print(title)
        print("=" * 78, flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
        elapsed = time.time() - t0
        record["steps"].append({"step": title, "script": script,
                                "returncode": result.returncode,
                                "seconds": round(elapsed, 1)})
        print("\n(%s finished in %.0fs, exit %d)\n" % (script, elapsed, result.returncode))
        if result.returncode != 0:
            print("STOPPING: %s failed." % script)
            break

    path = os.path.join(C.OUT_DIR, "reproducibility_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
