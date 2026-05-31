# Figures

Every figure referenced by the report, organised so the paths match what `docs/report.tex` expects. To recompile the report, copy these subfolders into an `images/` folder on Overleaf, keeping the names below.

## Q1_CycleGAN

- `03_sample_epoch_final.png` training sample grid: real photos, generated sketches, real sketches, generated faces.
- `04_ui_face_to_sketch.png` web UI translating faces into sketches.
- `05_ui_sketch_to_face.png` web UI translating sketches into faces.
- `03_sample_epoch5_backup.png` backup of the early epoch sample.

## Q2_Translation

- `01_training_header.png` training configuration summary.
- `02_corpus_alignment.png` corpus alignment check on the parallel data.
- `03_final_bleu_and_collapse.png` final epoch log with BLEU and sample outputs.
- `04_from_scratch_mode_collapse.png` from scratch inference showing identical outputs.
- `05_mbart_zeroshot_translations.png` mBART zero shot translations.
- `06_comparison_card.png` side by side comparison of from scratch and mBART.
- `07_data_augmentation_samples.png` EDA augmentation examples.
- `08_mbart_finetuned_translations.png` mBART fine tuned translations.

## Q3_ViT_CNN

- `01_cnn_curves.png`, `02_vit_curves.png`, `03_pretrained_curves.png` loss and accuracy curves per model.
- `04_cnn_confusion.png`, `05_vit_confusion.png`, `06_pretrained_confusion.png` confusion matrices.
- `07_example_preds.png` grid of example predictions.
- `08_comparison_chart.png` accuracy and parameter bar chart.
- `09_val_acc_overlay.png` validation accuracy overlay across all three models.
- `10_comparison_table.png` summary comparison table.
- `11_cnn_training_log.png`, `12_vit_training_log.png`, `13_pretrained_training_log.png` training logs.
