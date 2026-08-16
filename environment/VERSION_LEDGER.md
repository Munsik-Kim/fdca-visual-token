# Exact technical stack

## Model-free
Python 3.12.12 (T0) and 3.11 (GPU analysis), NumPy 2.2.6, SciPy 1.15.2, pandas 3.0.5, PyArrow 25.0.1, pytest 8.4.2, PyYAML 6.0.3. T0 used SciPy HiGHS linear programming.

## Full GPU
PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, LPIPS 0.1.4, scalar-canonical FP32, `compile=False`, TF32 disabled, RTX 5080 (16,303 MiB), driver 610.43.02, CUDA UMD 13.3. H-MaskGIT upstream commit `f61b0a1314717004dc7487531fd16a8bb71e1888`. Asset revisions/hashes are in `EXTERNAL_ASSET_REGISTRY.md`.

## Paper
Frozen PDF and LaTeX/BibTeX source are included. The clean public command uses Tectonic 0.17.0 locally; CI uses an immutable-pinned TeX Live action and checks page count plus source hashes.

## Audit/packaging
GitHub CLI 2.96.0 was used for publication. ZIPs use Deflate and are CRC/manifest checked. GitHub Actions dependencies are pinned to immutable commits. WSL2/Linux was the experiment host.
