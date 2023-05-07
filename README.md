# RL Project for learning STL

## Installation steps
```python
conda create -n rl_project python==3.7.6
conda activate rl_project
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge imageio
pip install xlsxwriter
pip install stable-baselines3
```

## Preparation
1. Create a directory `exps_rlp` in the father directory of this code repo: `mkdir -p ../exps_rlp`

## Training
(See `script_rl_project.sh` for more details)

`python train_1_car.py -e exp_1_car_rl_r --train_rl --epochs 400000 --num_workers 8`

`python train_2_ship.py -e exp_2_ship_il --train_il --epochs 50000 --num_workers 1 -P exp_4_ship_nn/models/model_49000.ckpt --il_mode all`

## Testing
(See `script_rl_project.sh` for more details)

`python test_1_car.py --test --rl -R exp_1_car_rl_r/models/model_last.zip --no_viz`

1. Remove `--no_viz` if you want to render the animations
2. The result will be saved in both the training directory `../exps_rlp/exp_1_car_rl_r/`, and a public directory `../exps_rlp/eval_result/`