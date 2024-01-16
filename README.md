## Preparing the reduced data for each model

run `data_prep_ham.py` by adjusting the corresponding model checkpoints and the argument specifying the model number. This file is modified from `data_prep_new.py` from the data preparation repo for the GTSRB traffic sign dataset. It produces tabular data outputs such as data_0_all.csv, data_1_all.csv, etc.

## Create a summary csv file

run `summary.py`. This file is modified from `summary.py` from the GTSRB repo. The original GTSRB `summary.py` reads in `data_[model#]_nat.csv` and `data_[model#]_adv.csv` individually, however in this repo the `summary.py` script reads in one file in the format of `data_[model#]_all.csv` in one run. The output file is `summary.csv`.

## Calculate the invariance between natural and adversarial embeddings

run `invariance.py`. Produce file is named inv.csv. Note: the script only write one row (per model) to inv.csv at a time.