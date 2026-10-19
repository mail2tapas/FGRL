"""Writes the small WfCommons-format test instances in data/test (3 per workflow type, 50 tasks)."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrl.workflows.generator import generate, to_wfformat  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "data/test"
os.makedirs(out, exist_ok=True)
rng = random.Random(2026)
for kind in ["montage", "cybershake", "epigenomics"]:
    for i in range(5):
        G = generate(kind, 50, rng)
        name = f"{kind}-50-{i}"
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump(to_wfformat(G, name), f, indent=1)
        print("wrote", name, G.number_of_nodes(), "tasks")
