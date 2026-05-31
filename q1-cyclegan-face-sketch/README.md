# Q1: CycleGAN for face and sketch translation

An unpaired image to image translator that converts a face photo into a pencil sketch and back. The architecture follows the original CycleGAN paper closely, and the project comes with a small Flask web app for trying it interactively.

## Architecture

- Generator: 9 block ResNet, the same design used in the original paper for 256x256 inputs.
- Discriminator: 70x70 PatchGAN, which scores overlapping patches rather than the whole image.
- Losses: least squares GAN loss, cycle consistency, and identity loss, with an image replay buffer to stabilise the discriminators.

## Files

```
cyclegan_model.py     ResNet generator, PatchGAN discriminator, and weight init
dataset.py            Person Face Sketches unpaired sampler and the image transforms
cyclegan_train.py     Training loop with LSGAN, cycle, identity, replay buffer, and --resume
inference.py          load_generators, detect_sketch, and the translate helper
flask_app.py          Flask web UI for upload, webcam capture, and auto detection
templates/index.html  Front end for the web UI
modal/q1_cyclegan.py  Modal wrapper that pulls the Kaggle dataset and trains on an L4 GPU
notebooks/            Colab and Kaggle notebooks used to run and resume training
```

## Run the web UI locally

You need a trained `generators.pt` checkpoint placed at `weights/generators.pt`. The checkpoint is not in the repository because of its size, so either train one with the script below or copy one down from your training run.

```bash
pip install torch torchvision pillow flask opencv-python
python flask_app.py --weights weights/generators.pt
# then open http://127.0.0.1:5000
```

The UI accepts an upload or a webcam capture, auto detects whether the input is a face or a sketch, and shows the translated result. You can override the detected direction if it guesses wrong.

## How training was run

Training happened in two stages. A short warm up of 5 epochs at 128x128 on Colab, then 30 epochs at 256x256 on a Modal L4 GPU. The Modal wrapper spawns the training script like this:

```bash
python cyclegan_train.py --image_size 256 --batch_size 8 --n_epochs 30 --decay_epoch 15 --resume
```

The persistent volume keeps checkpoints across container restarts, so the run can be stopped and resumed without losing progress.

## The mixed checkpoint

The face to sketch generator keeps improving across all 30 epochs, but the sketch to face generator looks best around epoch 5 and degrades after that, a side effect of training at batch size 8 rather than the paper's batch size 1. The shipped checkpoint therefore takes the face to sketch generator from the 30 epoch run and the sketch to face generator from the 5 epoch run. This is treated as a methodological observation rather than a bug, and it is explained in `../docs/report.pdf`.
