# Representation Learning for Documents in a Citation Network

This repository contains all code for reproducing the experiments presented in the thesis *Representation Learning for Documents in a Citation Network*.

To reproduce the experiments, please follow the following steps:

1) Clone this repository and install all dependencies listed in [the dependencies file](dependcies.txt).
2) Download the DBLP_V10 data from [here](https://aminer.org/citation).
3) Run the ```data_clean/clean_dblp_v10.py``` script to process the DBLP_V10 json data into abstracts and references.
4) Run the ```experiment.py``` script with different settings to reproduce and extend the research.
5) Analyse the results using the ```results.ipyn``` notebook.
