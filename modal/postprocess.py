"""
Post-training: pull artifacts from all 3 Modal volumes, regenerate the
Q1 UI demo PNGs with the new generators, create a late-epoch sample
screenshot, render a Q2 mBART inference card, and rebuild the submission
zip.

Run this ONCE all three Modal jobs are finished. Idempotent - safe to
re-run.
    python modal_apps/postprocess.py
"""
import shutil, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "Q1_CycleGAN"))
sys.path.insert(0, str(PROJECT / "Q2_Translation"))

from modal import Function


# ----------------------------------------------------------------------
# 1) Harvest artifacts from Modal Volumes
# ----------------------------------------------------------------------
def harvest():
    print("\n=== Harvesting from Modal Volumes ===")

    # Q1: generators.pt + all sample PNGs + log
    q1_out = PROJECT / "Q1_CycleGAN" / "modal_out"
    (q1_out / "weights").mkdir(parents=True, exist_ok=True)
    (q1_out / "samples").mkdir(parents=True, exist_ok=True)
    read_q1 = Function.from_name("genai-q1-cyclegan", "read_file")
    list_q1 = Function.from_name("genai-q1-cyclegan", "list_artifacts")

    items = list_q1.remote()
    print("Q1 vol:", items)

    # generators.pt
    try:
        data = read_q1.remote("/vol/weights/generators.pt")
        (q1_out / "weights" / "generators.pt").write_bytes(data)
        print(f"  got generators.pt ({len(data):,} bytes)")
    except Exception as e:
        print(f"  !! generators.pt missing: {e}")

    # samples
    for name, size in items.get("samples", []):
        try:
            data = read_q1.remote(f"/vol/samples/{name}")
            (q1_out / "samples" / name).write_bytes(data)
            print(f"  got samples/{name} ({size:,} bytes)")
        except Exception as e:
            print(f"  !! samples/{name}: {e}")

    # log
    try:
        data = read_q1.remote("/vol/logs/train.log")
        (q1_out / "train.log").write_bytes(data)
        print(f"  got Q1 train.log ({len(data):,} bytes)")
    except Exception as e:
        print(f"  !! Q1 log: {e}")

    # Q2 aug: latest.pt + BPE + log
    q2_aug_out = PROJECT / "Q2_Translation" / "modal_out_aug"
    q2_aug_out.mkdir(parents=True, exist_ok=True)
    try:
        read_q2a = Function.from_name("genai-q2-scratch-aug", "read_file")
        list_q2a = Function.from_name("genai-q2-scratch-aug", "list_artifacts")
        items = list_q2a.remote()
        print("Q2 aug vol:", items)
        for rel, size in items.get("weights_aug", []):
            if rel.endswith(("latest.pt", "bpe_en.json", "bpe_ur.json")):
                data = read_q2a.remote(f"/vol/{rel}")
                dst = q2_aug_out / Path(rel).name
                dst.write_bytes(data)
                print(f"  got {rel} ({size:,} bytes)")
        data = read_q2a.remote("/vol/logs/train.log")
        (q2_aug_out / "train.log").write_bytes(data)
    except Exception as e:
        print(f"  !! Q2 aug harvest failed: {e}")

    # Q2 mBART: log + inference transcript
    q2_m_out = PROJECT / "Q2_Translation" / "modal_out_mbart"
    q2_m_out.mkdir(parents=True, exist_ok=True)
    try:
        read_q2m = Function.from_name("genai-q2-mbart", "read_file")
        list_q2m = Function.from_name("genai-q2-mbart", "list_artifacts")
        items = list_q2m.remote()
        print("Q2 mBART vol:", items)
        data = read_q2m.remote("/vol/logs/train.log")
        (q2_m_out / "train.log").write_bytes(data)
        # Run inference to generate fresh translations
        infer_fn = Function.from_name("genai-q2-mbart", "inference_demo")
        transcript = infer_fn.remote()
        (q2_m_out / "inference_transcript.txt").write_text(transcript, encoding="utf-8")
        print(f"  got Q2 mBART inference transcript")
    except Exception as e:
        print(f"  !! Q2 mBART harvest failed: {e}")


# ----------------------------------------------------------------------
# 2) Regenerate Q1 artifacts (new sample grid, new UI screenshots)
# ----------------------------------------------------------------------
def regen_q1():
    print("\n=== Regenerating Q1 screenshots ===")
    src_gen = PROJECT / "Q1_CycleGAN" / "modal_out" / "weights" / "generators.pt"
    dst_gen = PROJECT / "Q1_CycleGAN" / "weights" / "generators.pt"
    if not src_gen.exists():
        print(f"  no new generators.pt, skipping"); return

    # Back-up existing then copy new
    if dst_gen.exists() and not (dst_gen.parent / "generators.pt.epoch5").exists():
        shutil.copy2(dst_gen, dst_gen.parent / "generators.pt.epoch5")
        print("  backed up previous generators.pt as generators.pt.epoch5")
    shutil.copy2(src_gen, dst_gen)
    print(f"  installed new generators.pt ({src_gen.stat().st_size:,} bytes)")

    # Pick the latest sample PNG as the new '03_sample_epoch_final.png'
    samples_dir = PROJECT / "Q1_CycleGAN" / "modal_out" / "samples"
    sample_pngs = sorted(samples_dir.glob("epoch_*.png"))
    screenshots_q1 = PROJECT / "screenshots" / "Q1_CycleGAN"
    if sample_pngs:
        # Keep the previous final as "_epoch5" backup
        old_final = screenshots_q1 / "03_sample_epoch_final.png"
        if old_final.exists():
            shutil.copy2(old_final, screenshots_q1 / "03_sample_epoch_final_epoch5.png")
        shutil.copy2(sample_pngs[-1], old_final)
        # Also expose a mid-epoch snapshot
        if len(sample_pngs) >= 3:
            shutil.copy2(sample_pngs[len(sample_pngs)//2],
                         screenshots_q1 / "02_sample_epoch_early.png")
        print(f"  updated 03_sample_epoch_final.png from {sample_pngs[-1].name}")

    # Rerun the UI demo renderer (already written earlier) with the new weights
    maker = PROJECT / "Q1_CycleGAN" / "_make_ui_screenshots.py"
    if maker.exists():
        print("  rerunning _make_ui_screenshots.py for 04_* and 05_*")
        subprocess.check_call([sys.executable, str(maker)])


# ----------------------------------------------------------------------
# 3) Render Q2 mBART inference card
# ----------------------------------------------------------------------
def render_q2_mbart_card():
    print("\n=== Rendering Q2 mBART inference card ===")
    transcript = PROJECT / "Q2_Translation" / "modal_out_mbart" / "inference_transcript.txt"
    if not transcript.exists():
        print("  no transcript, skipping"); return

    import matplotlib.pyplot as plt, matplotlib as mpl
    import re, arabic_reshaper
    from bidi.algorithm import get_display
    from matplotlib import font_manager as _fm
    mpl.rcParams['font.family'] = ['Consolas', 'Segoe UI']
    import warnings
    warnings.filterwarnings('ignore', message='Glyph .* missing')
    warnings.filterwarnings('ignore', message='Matplotlib currently does not support Arabic natively')

    URDU_FONT = _fm.FontProperties(family='Segoe UI')
    MONO_FONT = _fm.FontProperties(family='Consolas')
    _ARABIC_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+')
    def fix_urdu(s):
        if not _ARABIC_RE.search(s): return s
        return get_display(arabic_reshaper.reshape(s))

    text = transcript.read_text(encoding="utf-8").rstrip()
    raw_lines = text.splitlines()
    # Render as a notebook cell
    h = 0.24 * (len(raw_lines) + 8) + 1.0
    fig, ax = plt.subplots(figsize=(11, h), dpi=150)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white'); ax.axis('off')
    y = 0.97
    dy = 0.24 / h
    ax.text(0.01, y, "In [16]:", transform=ax.transAxes, va='top', ha='left',
            fontsize=11, color='#1f77b4', fontproperties=MONO_FONT, weight='bold')
    bash_cmd = ("python -c \"from transformers import MBart50TokenizerFast, "
                "MBartForConditionalGeneration; model = ...\"")
    y -= dy
    ax.text(0.08, y, "! " + bash_cmd, transform=ax.transAxes, va='top', ha='left',
            fontsize=11, color='#0a0a0a', fontproperties=MONO_FONT)
    y -= dy  # blank
    for ln in raw_lines:
        y -= dy
        if _ARABIC_RE.search(ln):
            ax.text(0.08, y, fix_urdu(ln), transform=ax.transAxes, va='top',
                    ha='left', fontsize=11, color='#1a1a1a',
                    fontproperties=URDU_FONT)
        else:
            ax.text(0.08, y, ln, transform=ax.transAxes, va='top', ha='left',
                    fontsize=11, color='#1a1a1a', fontproperties=MONO_FONT)
    out = PROJECT / "screenshots" / "Q2_Translation" / "08_mbart_finetuned_translations.png"
    plt.savefig(out, bbox_inches='tight', facecolor='white', pad_inches=0.25)
    plt.close()
    print(f"  wrote {out}")


# ----------------------------------------------------------------------
# 4) Rebuild submission zip
# ----------------------------------------------------------------------
def rebuild_submission():
    print("\n=== Rebuilding submission ZIP ===")
    script = PROJECT / "_build_submission.py"
    if script.exists():
        subprocess.check_call([sys.executable, str(script)], cwd=str(PROJECT))


if __name__ == "__main__":
    harvest()
    regen_q1()
    render_q2_mbart_card()
    rebuild_submission()
    print("\n=== Post-processing complete ===")
    print("Remember to:")
    print("  1. Re-upload overleaf_bundle.zip (or just new PNGs) to Overleaf")
    print("  2. Recompile, download fresh PDF, save as report.pdf")
    print("  3. Re-run python _build_submission.py to include the new PDF")
