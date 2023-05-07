# Train NN-STL (expert policy)
# Train RL-r (raw)
# Train RL-s (stl-score)
# Train RL-a (stl-accuracy)
# Train IL   (imitation learning from expert)
# Train IL-RL-r (pretrained from IL, and then keep RL-r)
# Train IL-RL-s (pretrained from IL, and then keep RL-s)
# Train IL-RL-a (pretrained from IL, and then keep RL-a)

# Test with each method above


# traffic
python train_1_car.py -e exp_1_car_rl_r --train_rl --epochs 400000 --num_workers 8
python train_1_car.py -e exp_1_car_rl_s --train_rl --epochs 400000 --num_workers 8 --stl_reward
python train_1_car.py -e exp_1_car_rl_a --train_rl --epochs 400000 --num_workers 8 --acc_reward

python test_1_car.py --test --rl -R exp_1_car_rl_r/models/model_last.zip --no_viz
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s/models/model_last.zip --no_viz
python test_1_car.py --test --rl --rl_acc -R exp_1_car_rl_a/models/model_last.zip --no_viz

# ship
python train_2_ship.py -e exp_2_ship_rl_r --train_rl --epochs 400000 --num_workers 8
python train_2_ship.py -e exp_2_ship_rl_s --train_rl --epochs 400000 --num_workers 8 --stl_reward
python train_2_ship.py -e exp_2_ship_rl_a --train_rl --epochs 400000 --num_workers 8 --acc_reward

python test_2_ship.py --test --rl --num_trials 20 -R exp_2_ship_rl_r/models/model_last.zip --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s/models/model_last.zip --no_viz
python test_2_ship.py --test --rl --rl_acc --num_trials 20 -R exp_2_ship_rl_a/models/model_last.zip --no_viz


python train_2_ship.py -e exp_2_ship_rl_r --train_il --epochs 50000 --num_workers 1 -P exp_4_ship_nn/models/model_49000.ckpt

# robot
python train_3_rover.py -e exp_3_rover_rl_r --train_rl --epochs 2000000 --num_workers 8
python train_3_rover.py -e exp_3_rover_rl_s --train_rl --epochs 2000000 --num_workers 8 --stl_reward
python train_3_rover.py -e exp_3_rover_rl_a --train_rl --epochs 2000000 --num_workers 8 --acc_reward

python test_3_rover.py --test --rl -R exp_3_rover_rl_r/models/model_last.zip --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --rl_stl -R exp_3_rover_rl_s/models/model_last.zip --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --rl_acc -R exp_3_rover_rl_a/models/model_last.zip --mpc_update_freq 10 --no_viz