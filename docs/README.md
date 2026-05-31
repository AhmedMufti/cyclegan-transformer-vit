# Report and documentation

This folder holds the written report and the supporting guides.

| File | What it is |
|------|------------|
| `report.pdf` | The compiled report covering all three projects, in Springer LNCS format. |
| `report.tex` | LaTeX source for the report. |
| `DEMO_GUIDE.md` | A walkthrough of what each result means, with likely questions and answers. Good for a live demo or a viva. |
| `SETUP.md` | Dependencies and how to move the project to another machine. |
| `COLAB_INSTRUCTIONS.md` | How to run the training on Colab if a local or cloud GPU is not available. |
| `PROJECT_NOTES.md` | Engineering decisions, the state of each task, and the gotchas worth knowing before extending the work. |

## Compiling the report

The report compiles on Overleaf with no extra setup.

1. Create a new blank project on Overleaf.
2. Upload `report.tex`.
3. Create a folder named `images` at the project root.
4. Copy everything from `../screenshots/` into `images/`, keeping the subfolder names:
   ```
   images/
     Q1_CycleGAN/...
     Q2_Translation/...
     Q3_ViT_CNN/...
   ```
5. Set the compiler to pdfLaTeX and recompile.

The figure paths in `report.tex` are case sensitive and already match the subfolder names under `screenshots/`, so the figures resolve without editing any paths.
