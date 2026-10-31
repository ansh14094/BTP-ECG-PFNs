# Gaussian context sampling for imbalanced regression

B.Tech. project, Department of Computer Science, IIT Roorkee.

Tabular in-context models such as TabPFN and TabICL predict by conditioning on a
*context set* drawn from the training data. That context is normally sampled
uniformly. This project asks what happens when it is sampled from a Gaussian over
the **target** instead, concentrating the context near a drawn centre:

    mu    = quantile(y, U(0.05, 0.95))
    sigma = std(y) * exp(U(log 0.15, log 1.5))
    w_i   proportional to exp(-0.5 * ((y_i - mu) / sigma)^2)

The motivation is imbalanced regression: if the context concentrates near a region
of target space, the model may become locally sharper there, including in the
sparse tails that uniform sampling under-represents.

## What was found

The intervention does one thing, consistently: **it moves accuracy from the bulk of
the target range into the tail.** On PTB-XL age estimation, patients over 60 improve
and patients aged 30-60 get worse, for every encoder tested. The two effects very
nearly cancel, so the method redistributes error rather than reducing it.

It does not generalise. Across abalone, communities-and-crime, wine quality and
AgeDB the Gaussian arm did not beat its matched control.

The one robust improvement in the study has nothing to do with Gaussian sampling:
training the ECG encoder end-to-end through a frozen *stock* TabICL improves mean
absolute error from 7.330 to 7.152 years, and improves every age band at once.

Full numbers: [`imbreg/results/matrix_age_bands.csv`](imbreg/results/matrix_age_bands.csv).
Narrative write-ups: [`TABICL_PRIOR_LOG.md`](TABICL_PRIOR_LOG.md) and
[`TABICL_TRANSFER_LOG.md`](TABICL_TRANSFER_LOG.md).

## Two levels of intervention

**Dataset level** (`imbreg/biased_split.py`) -- the context is Gaussian-sampled while
fine-tuning on the target dataset. The obvious objection is that this measures the
adaptation procedure, not the pre-trained model.

**Prior level** (`tabicl_prior/`) -- the released TabICL regressor is given 3000
further pre-training steps on *its own synthetic prior*, with the rows of each
synthetic dataset reordered so the context/query split lands on a
target-concentrated context. The resulting model has never seen any evaluation
dataset, so any effect is a property of the model rather than of fine-tuning.
TabPFN-2.5 cannot be tested this way: the checkpoint ships, the prior generator
does not.

Three prior variants are trained and compared: `gauss`, `mixed` (half Gaussian,
half uniform), and `uniform` -- a matched control performing a permutation of the
same shape and cost, so "the ordering helped" is separable from "3000 more steps
helped".

## Layout

| path | contents |
|---|---|
| `imbreg/` | evaluation harness: datasets, protocols, metrics, arms, plots |
| `tabicl_prior/` | prior-level continued pre-training of TabICL |
| `ecg_tabpfn/` | ECG encoders for PTB-XL age regression |
| `agedb_tabpfn/` | AgeDB image-age experiments |
| `third_party/` | vendored upstream source (not committed) |

## Reproducing

Checkpoints, datasets and raw logs are not in version control -- the TabICL prior
checkpoints are 218 MB each, and PTB-XL and AgeDB are redistributed under their own
licences. Obtain the datasets from their sources, then:

```bash
pip install -r requirements.txt

# prior-level continued pre-training (one run per arm, ~3.5 h each on one GPU)
cd tabicl_prior && ./run_prior.sh gauss

# the encoder x body matrix
./run_encoder_matrix.sh

# the table
cd imbreg && python3 make_matrix_csv.py
```

Jobs serialise on a GPU mutex (`gpu_lock.sh`); the workstation GPU is shared.

## Credit

The imbalanced-regression protocol, the DIR splits and the AgeDB pipeline follow
Yang et al., *Delving into Deep Imbalanced Regression*, ICML 2021. The upstream
AgeDB-DIR readme is preserved as `UPSTREAM_README_agedb_dir.md`. TabICL and TabPFN
are the work of their respective authors; this repository contains no copy of
either model's weights.
