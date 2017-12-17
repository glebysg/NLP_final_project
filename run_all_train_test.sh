#!/usr/bin/env bash
python train_test.py -p data/animals_sae/ -r results_train_test/animals_sae_results &&\
python train_test.py -p data/animals_sae_pca/ -r results_train_test/animals_sae_pca_results &&\
python train_test.py -p data/animals_sae_rf/ -r results_train_test/animals_sae_rf_results  &&\
python train_test.py -p data/apascal_pca/ -r results_train_test/apascal_pca_results &&\
python train_test.py -p data/apascal_rf/ -r results_train_test/apascal_rf_results &&\
python train_test.py -p data/animals_pca/ -r results_train_test/animals_pca_results  &&\
python train_test.py -p data/animals_rf/ -r results_train_test/animals_rf_results &&\
python train_test.py -p data/sun_pca/ -r  results_train_test/sun_pca_results &&\
python train_test.py -p data/sun_rf/ -r results_train_test/sun_rf_results
