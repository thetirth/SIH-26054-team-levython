"""Split the whole Reaper OBJ into airframe + prop (exact partition by group)."""
import numpy as np
from collections import defaultdict

SRC = "gcs-app/public/mq9-reaper.obj"
PROP_GROUPS = {"Propeller", "Blades"}

verts = []
cur = None
air, prop, header = [], [], []
with open(SRC, "r", errors="ignore") as fh:
    for line in fh:
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        if line.startswith("g "):
            cur = line[2:].strip() or "default"
        if cur is None:
            header.append(line)
        elif cur in PROP_GROUPS:
            prop.append(line)
        else:
            air.append(line)

with open("gcs-app/public/reaper-airframe.obj", "w", errors="ignore") as f:
    f.writelines(header + air)
with open("gcs-app/public/reaper-prop.obj", "w", errors="ignore") as f:
    f.writelines(header + prop)

# verify: hub center + disc plane of the prop part
V = np.array(verts)
pidx = set()
cur = None
with open(SRC, "r", errors="ignore") as fh:
    for line in fh:
        if line.startswith("g "):
            cur = line[2:].strip() or "default"
        elif line.startswith("f ") and cur in PROP_GROUPS:
            pidx.update(int(p.split("/")[0]) - 1 for p in line.split()[1:])
P = V[np.array(sorted(pidx))]
print("prop verts:", len(P), "center:", P.mean(0).round(3), "span:", (P.max(0) - P.min(0)).round(3))
print("thin axis (spin axis):", "XYZ"[int(np.argmin(P.max(0) - P.min(0)))])
print("airframe lines:", len(air), "| prop lines:", len(prop))
