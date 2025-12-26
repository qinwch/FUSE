# Flow-Matching-Posterior-Estimation

# Set up
```sh
conda create -n fmpe python=3.10
conda activate fmpe
pip install dingo-gw sbibm
```

# Experiments

## SBI Benchmark 

Training and evaluation scripts available in `./sbi-benchmark`. To train an FMPE model,
run

```sh
cd flow-matching-posterior-estimation/sbi-benchmark/
python run_sbibm.py --train_dir </path/to/train_dir>
```

where the training directory contains a `settings.yaml` file. MLP example settings can be found in `./flow-matching-posterior-estimation/sbi-benchmark/settings.yaml`.
While MM-DiT settings can be found in 'flow-matching-posterior-estimation/mmdit_trainning/settings.yaml'
You can change the yaml to change the network.