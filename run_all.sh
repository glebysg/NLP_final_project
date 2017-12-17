#!/usr/bin/env bash
python read_dataset.py -p data/animals_sae/ -r results_tuning/animals_sae_results_tuning &&\
python read_dataset.py -p data/animals_sae_pca/ -r results_tuning/animals_sae_pca_results_tuning &&\
python read_dataset.py -p data/animals_sae_rf/ -r results_tuning/animals_sae_rf_results_tuning  &&\
python read_dataset.py -p data/apascal_pca/ -r results_tuning/apascal_pca_results_tuning &&\
python read_dataset.py -p data/apascal_rf/ -r results_tuning/apascal_rf_results_tuning &&\
python read_dataset.py -p data/animals_pca/ -r results_tuning/animals_pca_results_tuning  &&\
python read_dataset.py -p data/animals_rf/ -r results_tuning/animals_rf_results_tuning &&\
python read_dataset.py -p data/sun_pca/ -r  results_tuning/sun_pca_results_tuning &&\
python read_dataset.py -p data/sun_rf/ -r results_tuning/sun_rf_results_tuning