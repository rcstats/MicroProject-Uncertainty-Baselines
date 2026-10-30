# MicroProject-Uncertainty-Baselines

Small-scale reproduction of the 2D toy experiments (two moons / two ovals,
Section 5.1, Appendix C.1, Figures 1–3) from:

> Liu et al., *Simple and Principled Uncertainty Estimation with Deterministic
> Deep Learning via Distance Awareness* (SNGP), NeurIPS 2020.
> [arXiv:2006.10108](https://arxiv.org/abs/2006.10108)

The paper's large-scale experiments (CIFAR, CLINC) live in
[google/uncertainty-baselines](https://github.com/google/uncertainty-baselines),
but the toy experiments were never released. Our starting point is Google's
[SNGP tutorial](https://www.tensorflow.org/tutorials/understanding/sngp),
which covers two moons with Deterministic / MC Dropout / Deep Ensemble / SNGP.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`tf-models-official` (where the tutorial's SNGP layers live) cannot be
installed on macOS arm64, so the two layer files are vendored unchanged in
[`sngp_layers/`](sngp_layers/) from `tensorflow/models@v2.15.0`.

## Run

```sh
.venv/bin/python -m toy.tutorial_sngp   # ~45 s on CPU, writes figures/tutorial/
```

## Layout

| Path | Contents |
|---|---|
| `sngp_layers/` | `SpectralNormalization`, `RandomFeatureGaussianProcess` (vendored) |
| `toy/tutorial_sngp.py` | Faithful port of the TF tutorial (step 1) |
| `figures/` | Generated figures |

## Paper vs tutorial

| | Paper (App. C.1) | Tutorial |
|---|---|---|
| Data | 500 pts/class; two moons + two ovals | two moons only |
| Network | ResFFN-12-128, dropout 0.01 | 6 layers, 128 units, dropout 0.1 |
| Methods | GP, Deep Ensemble (10), MC Dropout (10), DNN-SN, DNN-GP, SNGP | Deterministic, MC Dropout (10), Deep Ensemble (10), SNGP |
| Uncertainty plotted | Fig 1: `var(logit)` (GP models), `1-2\|p-0.5\|` (others); Figs 2–3: `1-2\|p-0.5\|` | `p(1-p)`, max-normalised |
| Optimiser / epochs / SN bound | not specified | Adam 1e-4, 100 epochs, batch 128, c = 0.9 |
