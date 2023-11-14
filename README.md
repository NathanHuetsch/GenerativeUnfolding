<h2 align="center"> Generative Unfolding</h2>

This a some code to do generative unfolding with INNs and CFMs.
## Installation

```sh
# clone the repository
git clone https://github.com/NathanHuetsch/GenerativeUnfolding
# then install in dev mode
cd GenerativeUnfolding
pip install --editable .
```

## The dataset

Downloading the dataset is done with the energyflow package. 
Open the datadownloader.py script and fill in your desired data path. Then run
```sh
pip install energyflow
python datadownloader.py
```
This does not work on Mac. If you are using a Mac, contact someone for the dataset or do the download on the cluster. 
Once you have the dataset move it to a subfolder GenerativeUnfolding/data

## Usage

Training a model:
```sh
src train params/paramcard.yaml
```
A new subfolder will be created in `output` which will contain log files, the run parameters,
the trained model and plots.

Re-running plots for a trained model:
```sh
src plot run_name
```
