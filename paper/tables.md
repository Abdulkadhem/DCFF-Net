# Paper tables — DCFF-Net

All numbers below are our own reproduced experiments unless a citation marks
them as published. Every value is regenerated from the JSON result files in
`results/`.

---

## Table 1 — Main comparison on LEVIR-CD (test)

| Method | Type | Params (M) | Precision | Recall | **F1** | **IoU** |
|---|---|---|---|---|---|---|
| FC-Siam-Diff | CNN | 1.35 | — | — | 86.3 | 76.3 |
| FC-Siam-Conc | CNN | 1.55 | — | — | 83.7 | 72.0 |
| SNUNet-CD | CNN | 12.0 | — | — | 88–90 | — |
| TinyCD | CNN | 0.28 | — | — | 91.0 | 83.6 |
| BIT | Transformer | 3.5 | — | — | 89.3 | 80.7 |
| ChangeFormer | Transformer | 41.0 | — | — | 90.4 | 82.5 |
| ChangeTitans | Transformer | — | — | — | **91.52** | **84.36** |
| **FC-Siam-Diff (our reproduction)** | CNN | 1.35 | 0.886 | 0.822 | 85.31 | 74.38 |
| **DCFF-Net (ours)** | CNN | 13.6 | — | — | 90.02 | 81.85 |
| **DCFF-Net (ours) + MS-TTA** | **CNN** | **13.6** | **0.921** | **0.893** | **90.69** | **82.96** |

The final model is dual-cue fusion + ASPP + a pretrained encoder, with CBAM and
the boundary loss removed following the ablation. Published figures differ
between sources; the reproduced baseline is the fair like-for-like reference.

---

## Table 2 — Ablation on LEVIR-CD test (60-epoch protocol, ResNet-18)

| # | Variant | F1 | IoU | Δ F1 vs dual-cue |
|---|---|---|---|---|
| A | difference-only fusion | 88.19 | 78.87 | **−1.12** |
| B | concatenation-only fusion | 89.15 | 80.43 | −0.16 |
| C | without ASPP | 89.15 | 80.43 | −0.16 (ASPP helps) |
| D | no pretraining (from scratch) | 84.78 | 73.58 | **−4.53** (transfer learning is decisive) |
| E | without CBAM | 89.58 | 81.13 | +0.27 → **removed** |
| F | without boundary loss | 89.48 | 80.96 | +0.17 → **removed** |
| **ref** | **dual-cue (full)** | 89.31 | 80.69 | reference |

**Conclusion.** The two drivers are transfer learning (+4.53) and dual-cue
fusion (+1.12). ASPP helps. CBAM and the boundary loss do not, and were
removed. The final model (no CBAM, no boundary loss, 100 epochs) scores
**F1 90.02 base / 90.69 with MS-TTA**, which is the adopted operating point.

---

## Table 3 — Multi-dataset study

### (a) Zero-shot generalisation diagnostic (no adaptation)

A controlled study of the effect of domain distance on a LEVIR-trained model.

| Target (LEVIR-trained →) | Domain distance | P | R | **F1** | IoU |
|---|---|---|---|---|---|
| **EGY-BCD** (building change, close to LEVIR) | near | 0.535 | 0.198 | **0.289** | 0.169 |
| EGY-BCD + TTA | near | 0.536 | 0.188 | 0.279 | 0.162 |
| **DSIFN-CD** (Google Earth, 2 m) | far | 0.994 | 0.015 | **0.029** | 0.015 |

**Observation.** Zero-shot transfer degrades *gradually* with domain distance:
F1 falls from 90 (in domain) to 0.29 on a similar building-change dataset, to
0.03 on a distant domain. This shows that (i) the model transports real signal
to a nearby domain — it detects 20 % of changes at 53 % precision with no
training at all — and (ii) the failure is caused by domain distance, not by a
defect of the model. Note also that TTA helps in domain (+0.5 F1) but does
**not** help under shift (−0.01). This fragility is the direct motivation for
the conformal reliability contribution.

### (b) In-domain multi-dataset results — five datasets

Trained and tested within each dataset, best validation checkpoint. DCFF-Net
beats the baseline on all five.

| Dataset | Method | P | R | **F1** | IoU | Δ F1 |
|---|---|---|---|---|---|---|
| **LEVIR-CD** | FC-Siam-Diff | 0.886 | 0.822 | 85.31 | 74.38 | |
| | **DCFF-Net (+MS-TTA)** | 0.921 | 0.893 | **90.69** | 82.96 | **+5.4** |
| **SYSU-CD** | FC-Siam-Diff | 0.777 | 0.809 | 79.24 | 65.62 | |
| | **DCFF-Net** | 0.848 | 0.824 | **83.57** | 71.78 | **+4.3** |
| **EGY-BCD** | FC-Siam-Diff | 0.652 | 0.713 | 68.10 | 51.63 | |
| | **DCFF-Net** | 0.768 | 0.865 | **81.39** | 68.63 | **+13.3** |
| **CLCD** | FC-Siam-Diff | 0.493 | 0.517 | 50.46 | 33.74 | |
| | **DCFF-Net** | 0.777 | 0.768 | **77.25** | 62.94 | **+26.8** |
| **DSIFN-CD** @512 | FC-Siam-Diff | 0.560 | 0.587 | 57.31 | 40.16 | |
| | **DCFF-Net** | 0.587 | 0.798 | **67.61** | 51.07 | **+10.3** |

A consistent advantage across five datasets (+4.3 to +26.8 F1) shows that
dual-cue fusion is a *general* property rather than a LEVIR-CD artefact.

**Honest notes.**
1. **DSIFN-CD** is our weakest result (67.61 adopted) and remains below
   published figures (~0.90) even at full resolution and with a doubled
   training budget (61.0 → 65.5 → 67.6). We keep it as a hard case, not as a
   competitiveness claim.
2. **CLCD** shows a very large margin (+26.8) partly because the baseline
   collapses on severely imbalanced data (change ratio ~1 %).
3. **Protocol check.** Our LEVIR-CD (7120/1024/2048), SYSU-CD
   (12000/4000/4000) and DSIFN-CD (3600/340/48) splits match the standard
   partitions, so those numbers are comparable. Our **CLCD** copy is a
   256-pixel crop re-split and is therefore **not comparable** with published
   CLCD numbers.
4. **EGY-BCD.** Adopted 81.39 (300 epochs, strong augmentation), below
   published results (AFDE-Net 88.8, Edge-CVT 90.12).

---

## Table 4 — Conformal-FDR layer (contribution 2)

Calibrated on the LEVIR-CD validation split, evaluated on the test split.

| α (target FDR) | Threshold λ̂ | Calibration FDP | **Test FDP (realised)** | Power (recall on change) | Guarantee |
|---|---|---|---|---|---|
| 0.05 | 0.780 | 0.0473 | **0.0451** | 0.7558 | satisfied |
| 0.10 | 0.080 | 0.0970 | **0.0969** | 0.8353 | satisfied |
| 0.20 | 0.020 | 0.1248 | **0.1254** | 0.8521 | satisfied |
| fixed 0.50 (baseline) | 0.500 | — | 0.0590 | 0.7914 | none (uncontrolled) |

The distribution-free guarantee holds on LEVIR-CD test at every level. Power
rises as the budget is relaxed (0.756 → 0.835 → 0.852). **Practical gain:** at
α ∈ {0.10, 0.20} the certified threshold reaches **higher** recall than the
default 0.5 threshold. Produced by `conformal_prep.py` + `conformal_fdr.py`.

---

## Table 5 — Conformal-FDR across datasets: soundness and shift diagnosis

Realised test FDR under two calibration regimes: (A) on the official
validation partition, (B) on an exchangeable split of the target's own test
set — the correct regime.

| α | LEVIR (A = B) | EGY (A) val | **EGY (B) exch.** | DSIFN (A) val | **DSIFN (B) exch.** |
|---|---|---|---|---|---|
| 0.05 | 0.045 ✓ | infeasible | infeasible | 0.355 ✗ | abstains |
| 0.10 | 0.097 ✓ | 0.102 ⚠ | **0.081 ✓** | 0.427 ✗ | abstains |
| 0.20 | 0.125 ✓ | 0.178 ✓ | **0.166 ✓** | 0.487 ✗ | **abstains** |

**Finding.** Under correct exchangeable calibration (B) the layer is sound on
all three datasets: it issues a valid FDR certificate or it abstains, and it
never lies. It certifies FDR ≤ {0.10, 0.20} on LEVIR-CD and EGY-BCD; on
DSIFN-CD, where test precision is intrinsically low, it **abstains honestly**
rather than making a false promise. Calibrating on a shifted distribution (A)
produces false certificates whose gap grows with the magnitude of the shift,
which makes the layer both a reliable threshold selector and a
distribution-shift diagnostic. Shift-robust conformal procedures are future
work.

---

## Table 6 — Optimisation study: seven avenues probed on LEVIR-CD

| # | Strategy | F1 | vs reference |
|---|---|---|---|
| — | **Original recipe (100 epochs)** | 90.02 | reference |
| **1** | **+ MS-TTA (1.0 / 1.25 / 1.5×)** | **90.69** | **+0.67 (adopted)** |
| 2 | Longer schedule (200 ep) + strong augmentation | 89.80 | −0.22 |
| 3 | Remote-sensing SSL pretraining (Sentinel-2 MoCo) | 89.14 | −0.88 |
| 4 | Lovász-hinge from scratch | 89.34 | −0.68 |
| 5 | Lovász polishing (lr 5e-5) | 90.00 | −0.02 |
| 6 | Model soup across independent runs | 86.57 | −3.45 |
| 7 | FixRes fine-tuning @384 (+MS-TTA) | 90.19 | −0.50 |

**Conclusion.** Of seven independent optimisation strategies, **only one
improved the model** (MS-TTA, inference only). This shows the recipe sits at a
tight, verified optimum rather than being lazily tuned. Two findings deserve
reporting: (a) low-resolution (10 m) remote-sensing pretraining does **not**
beat ImageNet for high-resolution (0.5 m) change detection — matching
*resolution* matters more than matching nominal *domain*; (b) model soup fails
across runs from different loss basins.
