# RAPO source code

This is the RAPO source code, which gives you easy access to the reproducibility of results.

## Installation 
To run the code, you will have some python packages installed. Under Linux system, you can import part of the packages from conda by running:
```sh
 python -m pip install -r requirements.txt

## Examples
To train RAPO with ppo:
```sh
python run.py --config-path config_halfcheetah_advV_ppo_meta.json --seed 0 --out-dir-prefix train
```