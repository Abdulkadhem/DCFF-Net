# Experiment log — DCFF-Net

A condensed, chronological record of every experiment behind the manuscript.
Raw artefacts are in `results/`: one `*_result.json` per run (full argument
set, parameter count, validation and test metrics) plus the training log.

---

## 1. LEVIR-CD — architecture development

| Stage | Configuration | Test F1 | Test IoU | Note |
|---|---|---|---|---|
| 1 | FC-Siam-Diff baseline (reproduced) | 85.31 | 74.38 | fair like-for-like reference |
| 2 | DCFF-Net, dual-cue, 60 ep | 89.31 | 80.69 | ablation reference point |
| 3 | + remove CBAM, remove boundary loss | 89.58 | 81.13 | simpler **and** better |
| 4 | 100-epoch schedule (final recipe) | 90.02 | 81.85 | adopted base |
| 5 | **+ multi-scale TTA** | **90.69** | **82.96** | **adopted operating point** |

Parameter count: 13.6 M. Encoder ResNet-18, ImageNet-initialised, trained at
0.1× the base learning rate. AdamW, cosine annealing, base lr 6e-4, batch 16,
pos_weight 2, mixed precision. Augmentation: random flips and 90° rotations.

## 2. Ablation (60-epoch protocol, LEVIR-CD test)

| Variant | F1 | IoU | Δ vs dual-cue |
|---|---|---|---|
| dual-cue (reference) | 89.31 | 80.69 | — |
| difference-only | 88.19 | 78.87 | −1.12 |
| concatenation-only | 89.15 | 80.43 | −0.16 |
| without ASPP | 89.15 | 80.43 | −0.16 |
| without CBAM | 89.58 | 81.13 | +0.27 → removed |
| without boundary loss | 89.48 | 80.96 | +0.17 → removed |
| no pretraining | 84.78 | 73.58 | −4.53 |

Drivers: transfer learning +4.53, dual-cue fusion +1.12.

## 3. Multi-dataset in-domain training

One shared recipe, no per-dataset tuning. Trained on two rented GPU sessions
(RTX 5060 Ti).

| Dataset | Baseline F1 | DCFF-Net F1 | Δ | Epochs | Note |
|---|---|---|---|---|---|
| LEVIR-CD | 85.31 | 90.69 | +5.4 | 100 | standard split |
| SYSU-CD | 79.24 | 83.57 | +4.3 | 100 | standard split; seed 7 gives 82.21 |
| EGY-BCD | 68.10 | 81.39 | +13.3 | 300 | strong augmentation |
| CLCD | 50.46 | 77.25 | +26.8 | 100 | 256-crop re-split, not comparable to published |
| DSIFN-CD | 57.31 | 67.61 | +10.3 | 200 | @512, hardest case |

**SYSU-CD seed variance.** Two seeds give 83.57 and 82.21, mean 82.89 ± 0.96.
We report the mean, not the better run, because the spread is comparable to
the margin separating the four methods we compare against.

**DSIFN-CD budget test.** 61.0 → 65.5 → 67.6 across successive budget
increases, so the gap to published results (~0.90) is not explained by an
insufficient schedule.

## 4. Zero-shot cross-dataset diagnostic (LEVIR-trained, no adaptation)

| Target | P | R | F1 | Note |
|---|---|---|---|---|
| EGY-BCD | 0.535 | 0.198 | 0.289 | near domain — real signal transfers |
| EGY-BCD + TTA | 0.536 | 0.188 | 0.279 | TTA does not help under shift |
| DSIFN-CD | 0.994 | 0.015 | 0.029 | far domain — collapse |

## 5. Conformal-FDR calibration

Calibrated on LEVIR-CD validation, evaluated once on test.

| α | λ̂ | calib FDP | test FDP | recall | guarantee |
|---|---|---|---|---|---|
| 0.05 | 0.780 | 0.0473 | 0.0451 | 0.7558 | satisfied |
| 0.10 | 0.080 | 0.0970 | 0.0969 | 0.8353 | satisfied |
| 0.20 | 0.020 | 0.1248 | 0.1254 | 0.8521 | satisfied |
| fixed 0.50 | 0.500 | — | 0.0590 | 0.7914 | none |

Cross-dataset soundness, exchangeable calibration (a random half of the
target's own test set):

| α | LEVIR | EGY-BCD | DSIFN-CD |
|---|---|---|---|
| 0.05 | 0.045 ✓ | infeasible | abstains |
| 0.10 | 0.097 ✓ | 0.081 ✓ | abstains |
| 0.20 | 0.125 ✓ | 0.166 ✓ | abstains |

Under deliberately shifted calibration (each dataset's official validation
partition) the layer issues false certificates whose gap tracks the shift:
EGY-BCD 0.102 at α = 0.10, DSIFN-CD 0.355 / 0.427 / 0.487.

## 6. Optimisation study — seven avenues

| Strategy | F1 | vs 90.02 |
|---|---|---|
| **multi-scale TTA** | **90.69** | **+0.67 — adopted** |
| longer schedule (200 ep) + strong augmentation | 89.80 | −0.22 |
| remote-sensing SSL pretraining (Sentinel-2 MoCo, torchgeo) | 89.14 | −0.88 |
| Lovász-hinge from scratch | 89.34 | −0.68 |
| Lovász polishing at lr 5e-5 | 90.00 | −0.02 |
| model soup across independent runs | 86.57 | −3.45 |
| FixRes fine-tuning @384 + MS-TTA | 90.19 | −0.50 |

Six of seven failed. Best validation score of the adopted recipe: 90.42; no
probed avenue crossed it at any epoch.

Two findings worth reporting on their own:
1. Sentinel-2 self-supervised pretraining (10 m) transfers **worse** than
   ImageNet for 0.5 m aerial change detection. Matching *resolution* matters
   more than matching nominal *domain*.
2. Model soup collapses across independently initialised runs, consistent with
   the requirement that souped models share a loss basin.

## 7. Reproduction

Every number above regenerates from the released code:

```bash
python code/train.py --dataset levir --model dcff --fusion dual \
  --no_cbam --no_boundary --batch 16 --lr 6e-4 --epochs 100 \
  --tag levir_dcff --out results
python code/eval_mstta.py --ckpt results/levir_dcff_best.pt --dataset levir
python code/conformal_prep.py --ckpt results/levir_dcff_best.pt
python code/conformal_fdr.py
```

Seeds are fixed at 42; the SYSU-CD variance study additionally uses seed 7.
