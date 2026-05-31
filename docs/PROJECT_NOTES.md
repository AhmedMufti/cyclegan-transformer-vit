# Project notes

Engineering decisions, the state of each task, and the things worth knowing before extending the work. This is the document to read first if you are picking the project up.

## Q1, CycleGAN for face and sketch

- Training ran in two stages: 5 epochs at 128x128 as a warm up, then 30 epochs at 256x256 on a Modal L4 GPU.
- The shipped `generators.pt` is a mixed checkpoint. The face to sketch generator comes from the 30 epoch run because it keeps improving, and the sketch to face generator comes from the 5 epoch run because it peaks early and then degrades.
- The degradation in the sketch to face direction comes from training at batch size 8 rather than the paper's batch size 1, which makes that direction train too fast and overshoot.
- The Flask UI runs on CPU and handles uploads, webcam capture, and automatic detection of whether the input is a face or a sketch.

## Q2, English to Urdu translation

- The from scratch Transformer trained for 40 epochs total, 20 plain and 20 with EDA augmentation. Final BLEU is around 0.43.
- The model mode collapses, which is expected on a 22.8k pair corpus. Augmentation shifted the collapse phrase but did not solve it. This is reported honestly as an observation about data scale.
- mBART-50 is included both zero shot and fine tuned for 3 epochs. Both produce fluent Urdu, with the fine tuned version a little more idiomatic.
- The augmentation module implements EDA from Wei and Zou 2019: random deletion, random swap, and synonym replacement via WordNet.

## Q3, Vision Transformer vs CNN

- CNN, ResNet-20 style: 92.78 percent validation accuracy, 2.78M parameters, 50 epochs.
- ViT from scratch: 82.19 percent, 2.69M parameters, 55 epochs with early stopping.
- Pretrained ViT-tiny from timm: 97.15 percent, 5.53M parameters, 3 epochs.
- All three trained on Colab. The CNN and scratch ViT are matched on parameter count so the comparison is about architecture, not capacity.

## Cloud training setup

The heavier runs used Modal for serverless GPU jobs, with persistent volumes so checkpoints survive container restarts. The shared helpers in `../modal/` launch and harvest those jobs, and each project has its own Modal app definition under its `modal/` folder.

When launching long jobs, prefer `modal deploy` together with `Function.spawn()` over the detached run flag. A detached run can be stopped if the local CLI loses its network connection, while a deployed function keeps running independently.

## Gotchas worth knowing

1. The mBART fast tokenizer changed its API. Use the `text_target=` keyword and set `tok.tgt_lang = "ur_PK"` before tokenising the target, otherwise the target side is tokenised with the wrong language.
2. Urdu in matplotlib needs `arabic_reshaper.reshape()` followed by the bidi algorithm before rendering, or the letters will not join. On Windows, use Segoe UI for the Arabic glyphs.
3. On Windows, set `PYTHONIOENCODING=utf-8` before any CLI call that prints check marks or other non ASCII characters, otherwise the console encoding raises an error.
4. The Kaggle notebook UI stops rendering new output after roughly 1 MB even though the job is still alive. Check the job status directly rather than assuming it died.
5. The CycleGAN sketch to face direction degrades after about epoch 5 at batch size 8, which is why the mixed checkpoint exists.

## What was intentionally left out

- No CIFAR retraining on cloud GPUs, since Q3 finished on Colab.
- No full CycleGAN retraining to completion at 256x256, since the mixed checkpoint approach was chosen instead.
- No mBART training beyond 3 epochs, which was enough to show that fine tuning improves the outputs.
- No attention weight visualisation for the ViT, noted as a possible extension.
