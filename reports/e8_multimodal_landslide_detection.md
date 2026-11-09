# E8: Lightweight multimodal landslide detection

## Context and completed work before E8

- E7 is a patch-level **landslide detection** classifier, not an event forecast.
- E7 test results recorded in the GPU logs:
  - GPU 0: accuracy 0.8054, AUC 0.8895, F1 0.7802.
  - GPU 1: accuracy 0.8023, AUC 0.8872, F1 0.7342.
- Inspection found that E7's two MobileNet branches were fed channel groups from
  one S2 sample, not independent S2 and S1 data. Its S2 time series were averaged
  across acquisitions.
- The initial NASA POWER export contained 3 patches from one event (93 daily
  patch-records). Those rows were a pipeline smoke test, not a complete weather
  dataset; their coincident values are expected for a coarse-grid weather source.
- The aligned S12LS-AD release contains 6,549 matched patches:
  4,191 train, 1,047 validation, and 1,311 test across 6 inventories. Aligned
  S1 ascending, S1 descending, and S2 NetCDF files were confirmed present.
- Both GPUs were idle when this E8 run was prepared.

## E8 pipeline being run

1. Download NASA POWER daily temperature, precipitation, and relative humidity
   for every aligned patch's location and 30-day pre-event interval. A 0.1-degree
   representative point is used to avoid redundant queries; the service's native
   weather grid remains coarse (about 0.5 degrees), not patch-resolution.
2. Expand the retrieved daily series back to each patch ID, retain a per-patch
   download manifest, and build event-relative features using dates strictly
   earlier than the inventory event date. POWER missing-value sentinels are
   treated as missing data.
3. Train a compact classifier on S2, S1 ascending, S1 descending, and the weather
   feature vector. The two S1 orbits share one MobileNetV3-Small encoder, so they
   add a second view without adding a third image backbone. Train-only
   normalization is used for image and weather inputs.
4. Select the checkpoint by validation F1 at threshold 0.5, evaluate the test
   split once, and save the checkpoint, training history, configuration, and
   metrics under `outputs/lightweight/E8_s1_s2_weather_<run-id>/`.

The background runner is `scripts/run_e8_s1_s2_weather.sh`. It writes timestamped
logs and updates this report with final results. The official splits are random
patch splits, not held-out inventory splits; weather and neighboring patches can
therefore share event context across splits. Test scores are not an estimate of
unseen-event performance.

## Next steps after this run

- E8 did not improve the test metrics over the stronger E7 run: accuracy was
  effectively unchanged, while F1 and AUC were lower. This is not a controlled
  modality ablation because E7 and E8 use different aligned inputs/preprocessing.
- Next compare matched S2-only, S1+S2, and S1+S2+weather models on this exact
  aligned split and seed. Then assess generalization with inventory-held-out
  evaluation before making any claim that weather fusion helps.

## E8 run status and results

- Completed successfully on GPU 0. Weather coverage: 6,549/6,549 aligned
  patches; 203,019 daily records; 0 failed requests; 31 daily records per
  patch.
- Training finished after 144.1 minutes. Best checkpoint was epoch 10, selected
  by validation F1 (0.8125).
- Test: accuracy 0.8055, AUC 0.8824, F1 0.7606.
- E7 GPU 0 reference: accuracy 0.8054, AUC 0.8895, F1 0.7802. E8 accuracy is
  essentially unchanged; AUC is 0.0071 lower and F1 is 0.0196 lower. E7 GPU 1
  had accuracy 0.8023, AUC 0.8872, F1 0.7342.
- This comparison is descriptive, not a controlled ablation, because E7 and E8
  use different modality/input pipelines. Neither run tests unseen-event
  generalization.
- Run log: `../logs/E8_s1_s2_weather_20261108_101310.log`.
- Checkpoint and metrics: `../outputs/lightweight/E8_s1_s2_weather_20261108_101310/`.

## E9 controlled ablation: planned and running

The E8 score does not isolate a modality effect, so E9 uses the same aligned
S12LS-AD train/validation/test split and seed (42) in three arms:

1. **S2 only** — pretrained MobileNetV3-Small optical image encoder.
2. **S1 + S2** — the same S2 branch plus one MobileNetV3-Small encoder shared
   by ascending and descending S1; orbit features are averaged.
3. **S1 + S2 + weather** — same image architecture plus a weather sequence and
   weather-summary encoder, fused with image tokens through a transformer
   attention layer and learned sigmoid gate.

Weather is more than a static classification head: the model reads the 30 daily
observations strictly before the inventory date through a small bidirectional
GRU, learns temporal attention over days, and combines that representation
with the existing pre-event rainfall/temperature/humidity aggregates. Sequence
inputs contain five NASA POWER variables, missing-value indicators, and
relative day position. Precipitation uses log1p; all normalization parameters
are fitted on training patches only. The image branches use cached pretrained
torchvision ImageNet MobileNetV3-Small weights; no Hugging Face model download
is needed.

All arms use seed 42, batch size 32, the same optimizer, 15-epoch maximum,
patience 5, and checkpoint selection by validation F1 at threshold 0.5. Each
reports accuracy, AUC, and F1 on the test split after checkpoint selection.
This is a controlled comparison of these model variants on the existing random
patch split, not a held-out-inventory test. The complete launch, progress, ETA,
test results, parameter counts, and run artifact paths are appended below.

### E9 verification before training

- `python -m unittest tests.test_environmental_features -v`: 3 tests passed,
  including strict pre-event feature construction, missing POWER values, and
  chronological 30-day sequence/event-day exclusion.
- Model forward smoke tests passed for all three ablations with the expected
  logit shape. Parameter counts: S2 1,144,481; S1+S2 2,146,497;
  S1+S2+weather 2,212,226.
- Python compilation, runner shell syntax, trainer CLI, and `git diff --check`
  passed.

## E8 run result

- Finished (UTC): 2026-11-08T12:43:21.271752+00:00
- Best epoch: 10; best validation F1: 0.8125
- Test accuracy: 0.8055; AUC: 0.8824; F1: 0.7606
- Training duration: 144.1 minutes
- Parameters: 1,930,497
- Artifacts: `/users/home/b23241/Sen12Landslides/outputs/lightweight/E8_s1_s2_weather_20261108_101310`
- Caveat: Official aligned random patch splits include the same inventories across splits; test scores are not an unseen-event estimate.

- Runner status: completed at 2026-11-08T18:13:25+05:30; wall time 150m 15s; log: /home/b23241/Sen12Landslides/logs/E8_s1_s2_weather_20261108_101310.log.

## E9 controlled ablation suite status

- Started: 2026-11-09T16:13:47+05:30; run id: 20261109_104347; output: /home/b23241/Sen12Landslides/outputs/lightweight/E9_ablation_20261109_104347
- Arms: S2 only; S1+S2; S1+S2+temporal weather and gated cross-modal attention.
- Both GPUs are used for the first two arms; the final multimodal arm starts when one GPU becomes free.
- Progress and ETA are logged per arm in logs/E9_20261109_104347_<ablation>.log.

### s2 arm progress

- Started: 2026-11-09T10:45:02.626621+00:00; parameters: 1,144,481; output: `/users/home/b23241/Sen12Landslides/outputs/lightweight/E9_ablation_20261109_104347/s2`.

### s1_s2 arm progress

- Started: 2026-11-09T10:46:20.793813+00:00; parameters: 2,146,497; output: `/users/home/b23241/Sen12Landslides/outputs/lightweight/E9_ablation_20261109_104347/s1_s2`.
- s2 epoch 1/15: train F1 0.6513, AUC 0.7880; validation F1 0.5028, AUC 0.7633; elapsed 7.0 min, ETA 98.2 min.
- s2 epoch 2/15: train F1 0.7608, AUC 0.8816; validation F1 0.3362, AUC 0.7600; elapsed 14.9 min, ETA 97.1 min.
- s1_s2 epoch 1/15: train F1 0.7118, AUC 0.8301; validation F1 0.0227, AUC 0.7458; elapsed 15.3 min, ETA 214.3 min.
- s2 epoch 3/15: train F1 0.8278, AUC 0.9284; validation F1 0.5160, AUC 0.8008; elapsed 22.2 min, ETA 88.9 min.
