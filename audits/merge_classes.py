import os
import sys

BASE = r"<PROJECT_ROOT>\lowalt\sam3_runs\pklot_v1\merged\yolo_obb"
LABEL_DIR = os.path.join(BASE, "labels")

# 10-class -> 3-class mapping
REMAP = {
    0: 0,  # car in parking spot     -> occupied
    1: 1,  # empty parking stall     -> empty
    2: 1,  # empty spot              -> empty
    3: 0,  # occupied parking space  -> occupied
    4: 0,  # parked car              -> occupied
    5: 0,  # parked vehicle          -> occupied
    6: 2,  # parking area            -> parking_area
    7: 2,  # parking lot             -> parking_area
    8: 1,  # parking space           -> empty
    9: 0,  # vehicle in parking lot  -> occupied
}

labels = [0, 1, 2]  # valid new label set

for split in ["train", "val", "test"]:
    split_dir = os.path.join(LABEL_DIR, split)
    if not os.path.isdir(split_dir):
        continue
    count = 0
    for fname in os.listdir(split_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(split_dir, fname)
        lines = []
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_cls = int(parts[0])
                if old_cls not in REMAP:
                    continue
                parts[0] = str(REMAP[old_cls])
                lines.append(" ".join(parts))
        with open(fpath, "w") as f:
            for line in lines:
                f.write(line + "\n")
        count += 1
    print(f"[{split}] {count} files remapped")

print("Done.")
