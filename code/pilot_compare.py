"""Print KITTI/STF mIoU trajectory for all exp/pilot_*/history.json runs."""
import os, json, glob
rows = []
for d in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "exp", "pilot_*"))):
    h = os.path.join(d, "history.json")
    if not os.path.exists(h):
        continue
    try:
        hist = json.load(open(h))
    except Exception:
        continue
    name = os.path.basename(d)
    traj = {e["epoch"]: (e["kitti"]["mIoU"] * 100, e["stf"]["mIoU"] * 100) for e in hist}
    last = max(traj) if traj else None
    rows.append((name, traj, last))
print(f"{'config':22s} | " + " | ".join(f"ep{e}(K/S)" for e in (3, 7, 11)) + " | mean@last")
print("-" * 78)
base = None
for name, traj, last in sorted(rows):
    cells = []
    for e in (3, 7, 11):
        if e in traj:
            cells.append(f"{traj[e][0]:4.1f}/{traj[e][1]:4.1f}")
        else:
            cells.append("   -    ")
    m = (traj[last][0] + traj[last][1]) / 2 if last is not None else 0
    if name == "pilot_none":
        base = m
    print(f"{name:22s} | " + " | ".join(cells) + f" | {m:5.2f}")
if base:
    print(f"\n(source-only mean@last = {base:.2f}; positive delta = improvement)")
