# CHQ-SAV scheme

Code for simulating how a mixture of two materials separates into patches over time, when the material also has elastic (rubber-like) stress that resists the separation. This repo contains the simulation method itself, two ready-to-run notebooks that show it in action, and the saved results of the slower runs so you don't have to wait hours to see them.

Accompanies the paper: [arXiv:2610.30077](https://arxiv.org/abs/2610.30077)

## `chq_sav/` — the simulation code

- **`chq_sav.py`** — the main simulation method. Given a starting pattern, it works out how the material evolves over time.
- **`chqpy_coef.py`** — the specific numbers and settings used to describe the material in the paper (how strongly it mixes, how stiff it is, the starting pattern, etc).
- **`dealias.py`** — a small cleanup step that removes numerical noise the simulation can pick up, so the pictures come out smooth.
- **`chq_nonlinear.py`** — a second, completely independent way of solving the same problem, used only to double-check that the main method gives the right answer.
- **`mms.py`** — a helper used only for accuracy testing: it builds a made-up example where the exact answer is known in advance, so we can check how close the simulation gets to it.

## `data/` — saved results

Some of the runs shown in the paper take many hours to compute. Rather than re-running them every time, the key numbers and pictures from those runs are saved here so the notebooks can load them instantly.

- **`energy_coupled.npz` / `energy_noA.npz`** — how much energy is left in the material over time, with and without the elastic effect.
- **`snapshots_coupled.npz` / `snapshots_noA.npz`** — pictures of the material's pattern at a few points in time, with and without the elastic effect.
- **`headline_energy_dt*.npz` / `headline_mass_dt*.npz` / `headline_rdrift_dt*.npz`** — energy, "is anything being lost or created" checks, and an internal-accuracy check for the simulation method itself, run three times at different levels of time detail.
- **`snapshots_W5.npz`** — pictures of the material's pattern for a version with a stronger mixing tendency.

## `notebooks/` — see it in action

- **`chq_vs_ch_demo.ipynb`** — compares the material with and without the elastic effect. Runs a short live example in under a minute, then shows the full comparison (pictures and energy over time) from the saved data, because that full run took about 12 hours.
- **`other_figures_demo.ipynb`** — the rest of the checks and pictures from the paper: a handful of quick accuracy checks that run live in under a minute, plus a few longer runs (tens of hours combined) shown from the saved data.

## How to use

Open either notebook in Jupyter and run it top to bottom. Each one explains, in its own first cell, which parts are computed live and which parts are loaded from saved results, and why.

## Use of AI-assisted technologies

During the preparation of this work, the authors used
ChatGPT and Claude to support code development, assist in formulating mathematical proofs,
explore the relevant literature, and improve the clarity and readability of the manuscript.
The authors critically reviewed the AI-assisted contributions and independently verified the
mathematical arguments, proofs, and numerical results reported in this work. The authors take
full responsibility for the accuracy and integrity of the content of this publication.
