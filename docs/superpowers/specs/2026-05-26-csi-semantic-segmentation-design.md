# CSI Semantic Segmentation with LatentMIM — Design Spec

**Date:** 2026-05-26
**Goal:** Verify that patch-level latent representations learned by a pre-trained LatentMIM model on CSI data contain meaningful semantic structure corresponding to propagation paths. Validated by generating unsupervised segmentation maps and t-SNE visualisations side by side.

---

## Section 1 — Data Pipeline

### Channel parameters
- Channel model: CDL-A (3GPP TR38.901) via Sionna
- Carrier frequency: 7.0 GHz
- Subcarrier spacing: 30 kHz
- Delay spread: 100 ns
- UE speed: 30 km/h (8.3 m/s)
- Antenna: **1×32 ULA, single-polarisation** (32 TX antennas, 1 RX antenna)
- N_PRB: 48

### Transformations
- **Angular domain:** 1D FFT with N_FFT=64 on 32 antenna elements (2× zero-padding) → 64 azimuth bins
- **Delay domain:** Use Sionna's CIR output directly (already delay-domain); choose N_delay=64 delay taps as a grid parameter
- Final per-sample tensor shape: `[2, 64, 64]` — 2-channel (real, imaginary), 64 delay taps × 64 azimuth bins (square aspect ratio)

### Normalisation
Per-sample normalisation using joint complex magnitude:

$$P = \sqrt{\frac{1}{N}\sum_{i,j}(\text{real}_{ij}^2 + \text{imag}_{ij}^2)}$$

Divide **both** real and imaginary channels by $P$. This preserves phase and relative amplitudes across delay-azimuth bins while removing per-sample power variation.

### Dataset split
- Total samples: 80,000
- Training set: 70,000
- Test set: 10,000

### Path count recording
Record the number of dominant paths per sample (defined as CDL-A clusters with power exceeding −20 dB relative to the strongest cluster) in a separate file alongside the dataset.

### Verification & visualisation
After generation, pick 3 random samples and for each plot two 2D heatmaps:
- Delay × azimuth magnitude `|H|`
- Annotate with the number of dominant paths

---

## Section 2 — LatentMIM Modifications

### Code change
`PatchEmbed` in `vits.py` and `build_lmim` in `models_lmim.py`: change `in_chans=3` → `in_chans=2`.

Specifically in `build_lmim`:
```python
model = LMIM(
    in_chans=2,   # real + imaginary channels
    patch_size=cfg['patch_size'],
    ...
)
```

### Config changes (`configs/lmim.yaml`)

| Parameter | Default | New value | Reason |
|---|---|---|---|
| `encoder` | `vit_base` | `vit_small` | Better data-to-parameter ratio for 80K samples |
| `patch_size` | 16 | 4 | Analogue to original 16px/224px ratio; paths span 1–2 patches |
| `grid_size` | 14 | 16 | 16×16 = 256 patches, close to original 196 |
| `num_vis` | 20 | 25 | ~10% visible, ~90% masked — consistent with original masking ratio |
| `patch_gap` | 4 | 0 | Stochastic offset too aggressive for 64×64 image |
| `input_size` | 224 | 64 | CSI image size |
| `epochs` | 300 | 800 | Small dataset needs more epochs to converge |
| `warmup_epochs` | 30 | 40 | ~5% of 800 epochs |
| `accum_iter` | 4 | 1 | Single GPU, batch_size=128 sufficient |
| `target_depth` | 12 | 12 | Keep full-depth target encoder |
| `dataset` | `imagenet` | `csi_cdla` | Custom CSI dataset loader to be added to `datasets.py` in Step 1 |

All other parameters (`blr`, `weight_decay`, `tau`, `loss`, `decoder_depth`, `avg_sim_coeff`, `target_mom`) remain at their defaults.

### Data transforms
Replace ImageNet transforms with:
- Per-sample complex normalisation (see Section 1)
- No `RandomResizedCrop`, no `RandomHorizontalFlip` — physically meaningless for CSI
- Masking within LatentMIM is the primary augmentation

### Hardware estimate
- Model: ViT-Small (~22M trainable parameters)
- Estimated VRAM: ~2.3 GB on a single Tesla V100 16GB
- Comfortable headroom; batch_size can be increased to 512+ if needed

### Training monitoring
- Print loss to terminal once per epoch (`print_freq` = steps per epoch ≈ 547)
- KNN accuracy evaluated every 10 epochs (`eval_freq=10`)
- Stop manually when KNN accuracy plateaus for 20–30 consecutive evaluations

---

## Section 3 — Inference & Segmentation

### Latent representation extraction
- Feed all 10,000 test samples through the **online encoder** with no masking
- Extract patch-level output (exclude CLS token): shape `[10000, 256, 384]`
- Use `torch.no_grad()` throughout

### Unsupervised segmentation
- For each sample, run `AgglomerativeClustering` (sklearn) on its 256 patch vectors `[256, 384]`
- Linkage: `ward`
- Number of clusters `k`: sweep `k ∈ {2, 3, 4, 5}`; use the `k` equal to the recorded number of dominant paths for that sample (from the path count file)
- Result: 16×16 cluster label map per sample

### Upsampling
Resize 16×16 label map → 64×64 using **nearest-neighbour interpolation** to preserve hard cluster boundaries.

---

## Section 4 — Visualisation

### Figure layout
For 5 randomly selected test samples, produce one row of 3 side-by-side plots per sample:

```
Sample 1: [CSI magnitude] [Segmentation map] [t-SNE]    n_paths = X
Sample 2: [CSI magnitude] [Segmentation map] [t-SNE]    n_paths = X
...
Sample 5: [CSI magnitude] [Segmentation map] [t-SNE]    n_paths = X
```

### Plot 1 — CSI delay-azimuth image
- Display `|H| = sqrt(real² + imag²)` as a 64×64 heatmap
- Colormap: `viridis`
- X-axis: azimuth bin (0–63), Y-axis: delay tap (0–63)

### Plot 2 — Unsupervised segmentation map
- 64×64 cluster label map (upsampled from 16×16)
- Colormap: `tab10` (qualitative, one fixed colour per cluster label)
- Same spatial axes as Plot 1

### Plot 3 — t-SNE of patch representations
- Run t-SNE independently per sample on `[256, 384]` patch vectors
- Each of the 256 points coloured by its cluster label using the **same `tab10` mapping as Plot 2**
- 2D scatter plot

### Colour consistency
Both Plot 2 and Plot 3 derive colours from the same cluster label array using the same colormap. Cluster $k$ maps to the same colour in both plots — no additional alignment step needed.

### Annotation
Number of dominant paths written as a title above each row, taken from the recorded path count file generated in Section 1.
