# Analysis code

Code behind the Central Asian study-abroad decision-making study. Turns the raw survey exports into every table and figure in the paper.

## Layout

```
config.py               paths, constants, scoring keys - change values here, not inside the scripts
src/
  data_prep.py           builds the analytic dataset from the three raw excel files
  data_quality.py        duplicate/range/straightlining checks
  reliability.py         cronbach's alpha, mcdonald's omega, item-total correlations
  balance.py             randomization balance checks
  stats_utils.py         cohen's d, eta squared, TOST - shared by several other files
  hypothesis_tests.py    H1, H2, H3, H5
  intervention_effects.py H6 - paired tests / ANOVA / ANCOVA / DiD / mixed models
  mediation.py           bootstrap mediation, multi-mediator horse race
  calibration.py         overconfidence / Dunning-Kruger
  choice_experiments.py  the five DCE tasks
  equivalence.py         TOST on every null result in the study
  segmentation.py        PCA and k-means
  make_figures.py        figures
tests/                  unit tests, run these before you trust anything above
data/                   put Pre-Test.xlsx, Post-Test.xlsx, RCT_Assignment.xlsx here (not committed)
output/                 everything the scripts write goes here (not committed)
```

## Running it

```bash
pip install -r requirements.txt
python run_all.py
```

Or run any `src/*.py` file on its own - most of them just need `output/ANALYTIC.pkl` to already exist, which `data_prep.py` builds.

## Tests

```bash
pytest tests/ -v
```

These don't touch the real data at all - they build small synthetic datasets where we already know the right answer (e.g. items with a known true correlation should give alpha close to what the math says it should be), and check the functions get there. If a test fails after you change something in `src/`, that's the signal to stop and look, not to skip it.

21 tests currently, covering reliability, the shared stats helpers, and the mediation bootstrap.

## A few things worth knowing before you touch anything

- `mcdonald_omega` uses first-eigenvector loadings, not an iteratively fit factor model. Good enough for what we need here but don't call it a real CFA.
- `bootstrap_indirect` reseeds per-resample with `rng.integers()`, not a fixed seed each time - if you need bit-for-bit reproducibility across runs, pass the same top-level `seed` and it'll reproduce, but don't expect two different `n_boot` values to give the same draws.
- multi-select survey columns (comma-separated strings like `"1,3"`) get turned into `NaN` by `numify()` in `data_prep.py`, on purpose - we don't currently use any multi-select items in the composite scores, but if you add one, you'll need to handle the string-splitting yourself before it hits `numify`.
- `config.SCALES` only has the 4-5 item psychological scales. The 2-3 item contextual scales (trust, econ, mobil) are built directly in `data_prep.build_wave` with `.mean(axis=1)`, they're not run through `reliability.py`'s scale loop by default - add them to the dict there if you want their alpha/omega printed too.

## Known limitation in the code itself

PCA in `segmentation.py` suggests the 21-item psychological block is closer to a 2-factor structure than the 5 sub-scales the instrument was designed around. We kept the 5-scale scoring in `data_prep.py` anyway, for comparability with the source instruments each sub-scale was adapted from. Worth remembering if you're extending this - the sub-scale boundaries may not hold up to a proper CFA.
