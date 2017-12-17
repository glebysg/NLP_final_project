#!/usr/bin/env bash
python read_dataset.py -p data/apascal_pca/ -r apascal_pca_results &&\
python read_dataset.py -p data/animals/ -r animals_results &&\
python read_dataset.py -p data/animals_pca/ -r animals_pca_results  &&\
python read_dataset.py -p data/animals_rf/ -r animals_rf_results &&\
python read_dataset.py -p data/animals_sae/ -r animals_sae_results &&\
python read_dataset.py -p data/animals_sae_pca/ -r animals_sae_pca_results &&\
python read_dataset.py -p data/animals_sae_rf/ -r animals_sae_rf_results  &&\
python read_dataset.py -p data/apascal/ -r apascal_results &&\
python read_dataset.py -p data/apascal_rf/ -r apascal_rf_results &&\
python read_dataset.py -p data/sun/ -r sun_results &&\
python read_dataset.py -p data/sun_pca/ -r  sun_pca_results &&\
python read_dataset.py -p data/sun_rf/ -r sun_rf_results
