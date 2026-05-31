"""
Download artifacts from all three Modal Volumes into the local project tree.

Run AFTER the training jobs finish.
    python modal_apps/harvest.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import modal

Q1_APP = modal.App.lookup("genai-q1-cyclegan", create_if_missing=False)
Q2_MBART_APP = modal.App.lookup("genai-q2-mbart", create_if_missing=False)
Q2_AUG_APP = modal.App.lookup("genai-q2-scratch-aug", create_if_missing=False)


def pull(app_name, vol_paths, local_root):
    """Ask the app's read_file() function for each remote path, write locally."""
    from modal import Function
    fn_read = Function.from_name(app_name, "read_file")
    for vp in vol_paths:
        data = fn_read.remote(vp)
        local = local_root / Path(vp).relative_to("/vol")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        print(f"  {len(data):>10,} bytes  ->  {local}")


if __name__ == "__main__":
    # Q1: generator weights + a late-epoch sample
    from modal import Function
    q1_list = Function.from_name("genai-q1-cyclegan", "list_artifacts").remote()
    print("Q1 artifacts:", q1_list)
    wanted_q1 = ["/vol/weights/generators.pt"]
    for name, size in q1_list.get("samples", []):
        if name.endswith(".png"):
            wanted_q1.append(f"/vol/samples/{name}")
    # Also grab the final log
    wanted_q1.append("/vol/logs/train.log")
    pull("genai-q1-cyclegan", wanted_q1, PROJECT / "Q1_CycleGAN" / "modal_out")

    # Q2 mBART: last epoch folder's files
    q2m_list = Function.from_name("genai-q2-mbart", "list_artifacts").remote()
    print("Q2 mBART artifacts:", q2m_list)
    wanted_q2m = ["/vol/logs/train.log"]
    # mBART saves epoch_N/ directories. The inference cell reads them back
    # from the Volume so we only need the training log locally + a rendered
    # inference transcript.
    pull("genai-q2-mbart", wanted_q2m, PROJECT / "Q2_Translation" / "modal_out_mbart")

    # Q2 aug: weights + log
    q2a_list = Function.from_name("genai-q2-scratch-aug", "list_artifacts").remote()
    print("Q2 aug artifacts:", q2a_list)
    wanted_q2a = ["/vol/logs/train.log"]
    for name, size in q2a_list.get("weights_aug", []):
        if name.endswith("latest.pt") or name.endswith("bpe_en.json") or name.endswith("bpe_ur.json"):
            wanted_q2a.append(f"/vol/{name}")
    pull("genai-q2-scratch-aug", wanted_q2a, PROJECT / "Q2_Translation" / "modal_out_aug")

    print("\nAll done.")
