#!/usr/bin/env bash
python train_test.py -p data/animals_sae/ -r results_train_test/animals_sae_5f_betaresults -b 0.01 &&\
python train_test.py -p data/animals_sae_pca/ -r results_train_test/animals_sae_pca_5f_betaresults -b 0.01 &&\
python train_test.py -p data/animals_sae_rf/ -r results_train_test/animals_sae_rf_5f_betaresults -b 0.01
