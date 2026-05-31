"""
Spawn the two deployed Q2 training functions. Unlike `modal run --detach`,
`.spawn()` completely disconnects from the local process.
"""
import sys
from pathlib import Path
from modal import Function

call_ids = {}

for app, fn in [("genai-q2-mbart", "train"),
                ("genai-q2-scratch-aug", "train")]:
    f = Function.from_name(app, fn)
    handle = f.spawn()
    call_ids[app] = handle.object_id
    print(f"  spawned {app}/{fn}: call_id = {handle.object_id}")

# Persist so the user can poll later
out = Path(__file__).parent / "call_ids.txt"
out.write_text("\n".join(f"{k}={v}" for k, v in call_ids.items()), encoding="utf-8")
print(f"\nCall IDs saved to {out}")
