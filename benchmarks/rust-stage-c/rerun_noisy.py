"""Move every 100k job that started while the pinned CPU was busy (flagged noisy in raw/*.load.json) to
raw/discarded_noisy/ so that `run_stage_c.py --phase 1 --phase 2 --phase 4` runs it again. The first attempts are kept, not deleted.
Usage: python rerun_noisy.py && python -u run_stage_c.py --phase 1 --phase 2 --phase 4 >> run_rerun.log"""
import glob
import json
import os
import shutil

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
DIS = os.path.join(RAW, "discarded_noisy")
os.makedirs(DIS, exist_ok=True)
moved = []
for lp in sorted(glob.glob(os.path.join(RAW, "*.load.json"))):
    base = os.path.basename(lp)[:-len(".load.json")]
    if "_100000_" not in base:
        continue
    ld = json.load(open(lp))
    if not ld.get("noisy"):
        continue
    for ext in (".json", ".txt", ".load.json"):
        p = os.path.join(RAW, base + ext)
        if os.path.exists(p):
            shutil.move(p, os.path.join(DIS, base + ext))
    moved.append(base)
print(len(moved), "moved:", " ".join(moved))
