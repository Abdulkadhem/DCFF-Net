# Equations — DCFF-Net

The 13 numbered equations of the manuscript, in the same order and with the
same numbering as `latex/dcffnet.tex`. Each is annotated with the file that
implements it.

---

## Part I — the detector

**(1) Siamese encoder** — `code/models.py::ResNetEncoder`

```
F¹ᵢ = 𝓔_θ(T₁)ᵢ ,   F²ᵢ = 𝓔_θ(T₂)ᵢ ,   i ∈ {1,2,3,4}
```

A weight-shared ResNet-18 maps the bitemporal pair to four scales with strides
{1/4, 1/8, 1/16, 1/32}. Sharing weights guarantees that identical content maps
to identical features, so any residual discrepancy is evidence of change.

**(2) The two cues** — `code/models.py::DualCueFusion`

```
diffᵢ = |F¹ᵢ − F²ᵢ|
concᵢ = ψ( W_c [ F¹ᵢ ; F²ᵢ ] ) ,   ψ = BatchNorm → ReLU
```

`[·;·]` is channel concatenation and `W_c` a 1×1 convolution restoring the
channel count. The difference cue is magnitude-sensitive but blind to the sign
and to the identity of the transition, since |a−b| = |b−a|; the concatenation
cue preserves relational context but carries no explicit change signal.

**(3) Dual-cue fusion (contribution 1)** — `code/models.py::DualCueFusion`

```
fusedᵢ = φ( [ diffᵢ ; concᵢ ] ) ,   φ = (Conv3×3 → BN → ReLU) × 2
```

**(4) ASPP context module** — `code/models.py::ASPP`

```
ASPP(x) = W_p [ { Conv3×3_r(x) }_{r ∈ {1,2,4}} ; GAP(x) ]
```

Dilation rates r chosen to cover the receptive field of a 256-pixel tile
without the gridding artefacts that larger rates introduce.

**(5) Output probability map** — `code/models.py::DCFFNet.forward`

```
p̂ = σ( 𝓤( Head( 𝓓( {fusedᵢ} ) ) ) ) ∈ [0,1]^{H×W}
```

Head is a 1×1 convolution to one channel, σ the logistic function and 𝓤
bilinear upsampling to the input resolution H×W.

**(6) Training objective** — `code/train.py::compute_loss`

```
𝓛 = Σ_{s ∈ 𝓢} w_s ( 𝓛^(s)_BCE(pos_weight = 2) + 𝓛^(s)_Dice )
w_main = 1 ,   w_aux = 0.4   (3 auxiliary heads)
```

Change pixels are rare (≈8.7 % on LEVIR-CD, ≈1 % on CLCD), so a plain
cross-entropy converges to a degenerate all-background solution.

**(7) Dice term** — `code/train.py::dice_loss`

```
𝓛_Dice = (1/B) Σ_b ( 1 − ( 2 Σ_x p̂_b(x)·y_b(x) + ε ) / ( Σ_x ( p̂_b(x) + y_b(x) ) + ε ) ) ,   ε = 1
```

Computed per image and averaged over the batch B. The smoothing constant ε
keeps images containing no change from producing an undefined loss.

---

## Part II — the reliability layer (contribution 2)

**(8) Per-image false-discovery proportion** — `code/conformal_prep.py`

```
Lᵢ(λ) = FDPᵢ(λ) = FPᵢ(λ) / ( TPᵢ(λ) + FPᵢ(λ) ) ∈ [0,1] ,
Lᵢ(λ) = 0  if  TPᵢ + FPᵢ = 0
```

Non-increasing in λ and bounded above by B = 1 — exactly the requirements of
conformal risk control.

**(9) Empirical risk over n calibration images** — `code/conformal_fdr.py`

```
R̂ₙ(λ) = (1/n) Σᵢ Lᵢ(λ)
```

**(10) Conformal threshold selection** — `code/conformal_fdr.py::crc_threshold`

```
λ̂ = inf { λ : ( n·R̂ₙ(λ) + B ) / ( n + 1 ) ≤ α }
```

The `+B` in the numerator is what pays for the finite calibration sample. If
no λ satisfies the inequality, the procedure **abstains** rather than issuing
an invalid certificate.

**(11) The guarantee**

```
E[ FDP_test(λ̂) ] ≤ α
```

Holds for any exchangeable test image, with no assumption whatsoever on the
data distribution or on the detector. The layer is post-hoc, model-agnostic
and training-free.

---

## Part III — evaluation

**(12) Metrics on the change class** — `code/eval_local.py`

```
P   = TP / (TP + FP)
R   = TP / (TP + FN)
F1  = 2PR / (P + R)
IoU = TP / (TP + FP + FN)
```

Computed over the accumulated test-set confusion matrix.

**(13) Multi-scale test-time augmentation** — `code/eval_mstta.py`

```
p̄ = 1/(|S||V|) Σ_{s ∈ S} Σ_{v ∈ V} 𝓡( v⁻¹( f_θ( v(𝓤_s T₁), v(𝓤_s T₂) ) ) )
S = {1.0, 1.25, 1.5} ,   V = 6 dihedral views
```

𝓤_s is the upsampling operator and 𝓡 the resampling back to the label grid.
Resampling *before* averaging matters: it prevents the coarser scales from
blurring the boundaries recovered by the finer ones. This is the only one of
the seven optimisation avenues we probed that improved the model.
