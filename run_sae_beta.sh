#!/usr/bin/env bash
python read_dataset_beta.py -f 5 -p data/animals_sae/ -r results_tuning/animals_sae_results_tuningbeta &&\
python read_dataset_beta.py -f 5 -p data/animals_sae_pca/ -r results_tuning/animals_sae_pca_results_tuningbeta &&\
python read_dataset_beta.py -f 5 -p data/animals_sae_rf/ -r results_tuning/animals_sae_rf_results_tuningbeta
