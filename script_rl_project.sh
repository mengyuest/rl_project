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




# robot
python train_3_rover.py -e exp_3_rover_rl_r --train_rl --epochs 2000000 --num_workers 8
python train_3_rover.py -e exp_3_rover_rl_s --train_rl --epochs 2000000 --num_workers 8 --stl_reward
python train_3_rover.py -e exp_3_rover_rl_a --train_rl --epochs 2000000 --num_workers 8 --acc_reward

python test_3_rover.py --test --rl -R exp_3_rover_rl_r/models/model_last.zip --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --rl_stl -R exp_3_rover_rl_s/models/model_last.zip --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --rl_acc -R exp_3_rover_rl_a/models/model_last.zip --mpc_update_freq 10 --no_viz


# imitation learning
run_rlp python train_1_car.py -e exp_1_car_il --train_il --epochs 50000 --num_workers 1 -P e1_traffic/models/model_49000.ckpt --hiddens 64 64 64
run_rlp python train_2_ship.py -e exp_2_ship_il --train_il --epochs 50000 --num_workers 1 -P exp_4_ship_nn/models/model_49000.ckpt
run_rlp python train_3_rover.py -e exp_3_rover_il --train_il --epochs 50000 --num_workers 1 -P e5_rover_pret/models/model_249000.ckpt

# RL based on imitation learning
run_rlp python train_1_car.py -e exp_1_car_il_rl_r --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_r --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last
run_rlp python train_3_rover.py -e exp_3_rover_il_rl_r --train_rl --epochs 2000000 --num_workers 8 -R exp_3_rover_il/models/model_last

run_rlp python train_1_car.py -e exp_1_car_il_rl_s --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward
run_rlp python train_3_rover.py -e exp_3_rover_il_rl_s --train_rl --epochs 2000000 --num_workers 8 -R exp_3_rover_il/models/model_last --stl_reward

run_rlp python train_1_car.py -e exp_1_car_il_rl_a --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --acc_reward
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_a --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --acc_reward
run_rlp python train_3_rover.py -e exp_3_rover_il_rl_a --train_rl --epochs 2000000 --num_workers 8 -R exp_3_rover_il/models/model_last --acc_reward
ls



# RUN NEW
run_rlp python train_1_car.py -e exp_1_car_il_rl_r_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --learn
run_rlp python train_1_car.py -e exp_1_car_il_rl_s_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn
run_rlp python train_1_car.py -e exp_1_car_il_rl_a_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --acc_reward --learn

run_rlp python train_2_ship.py -e exp_2_ship_il_rl_r_new --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --learn
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_new --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --learn
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_a_new --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --acc_reward --learn
ls

run_rlp python train_3_rover.py -e exp_3_rover_il_rl_r_new --train_rl --epochs 2000000 --num_workers 8 -R exp_3_rover_il/models/model_last --learn
run_rlp python train_3_rover.py -e exp_3_rover_il_rl_s_new --train_rl --epochs 2000000 --num_workers 8 -R exp_3_rover_il/models/model_last --stl_reward --learn
run_rlp python train_3_rover.py -e exp_3_rover_il_rl_a_new --train_rl --epochs 2000000 --num_workers 8 -R exp_3_rover_il/models/model_last --acc_reward --learn
ls



run_rlp python train_2_ship.py -e exp_2_ship_il_rl_r_new2 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --learn --load_others
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_new2 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --learn --load_others
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_a_new2 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --acc_reward --learn --load_others
ls


## TEST RL
exp1_rl_raw/models/model_last.zip
exp1_rl_stl/models/model_last.zip
exp1_rl_acc/models/model_last.zip

exp4_rl_raw/models/model_last.zip
exp4_rl_stl/models/model_last.zip
exp4_rl_acc/models/model_last.zip

exp5_rl_raw/models/model_last.zip
exp5_rl_stl/models/model_last.zip
exp5_rl_acc/models/model_last.zip

g0327-182454_exp6_rl_stl/models/model_last.zip


python test_1_car.py --test --il -R exp_1_car_il/models/model_last
python test_1_car.py --test --rl --rl_acc -R exp_1_car_rl_a/models/model_last
python test_2_ship.py --test --il --num_trials 20 -R exp_2_ship_il/models/model_last
python test_2_ship.py --test --rl --rl_acc --num_trials 20 -R exp_2_ship_rl_a/models/model_last
python test_3_rover.py --test --il -R exp_3_rover_il/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --rl --rl_acc -R exp_3_rover_rl_a/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --rl --il --rl_acc -R exp_3_rover_il_rl_a/models/model_last --mpc_update_freq 10
ls

###### TEST ALL EXPERIMENTS (no viz)
python test_1_car.py --test -P e1_traffic/models/model_49000.ckpt --no_viz
python test_1_car.py --test --rl --rl_raw -R exp_1_car_rl_r/models/model_last --no_viz
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s/models/model_last --no_viz
python test_1_car.py --test --rl --rl_acc -R exp_1_car_rl_a/models/model_last --no_viz
python test_1_car.py --test --il -R exp_1_car_il/models/model_last --no_viz
python test_1_car.py --test --rl --il --rl_raw -R exp_1_car_il_rl_r/models/model_last --no_viz
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s/models/model_last --no_viz
python test_1_car.py --test --rl --il --rl_acc -R exp_1_car_il_rl_a/models/model_last --no_viz


python test_2_ship.py --test --num_trials 20 -P exp_4_ship_nn/models/model_49000.ckpt --no_viz
python test_2_ship.py --test --rl --rl_raw --num_trials 20 -R exp_2_ship_rl_r/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_acc --num_trials 20 -R exp_2_ship_rl_a/models/model_last --no_viz
python test_2_ship.py --test --il --num_trials 20 -R exp_2_ship_il/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_raw --num_trials 20 -R exp_2_ship_il_rl_r/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_acc --num_trials 20 -R exp_2_ship_il_rl_a/models/model_last --no_viz


python test_3_rover.py --test -P e5_rover_pret/models/model_249000.ckpt --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --rl_raw -R exp_3_rover_rl_r/models/model_last --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --rl_stl -R exp_3_rover_rl_s/models/model_last --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --rl_acc -R exp_3_rover_rl_a/models/model_last --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --il -R exp_3_rover_il/models/model_last --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --il --rl_raw -R exp_3_rover_il_rl_r/models/model_last --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --il --rl_stl -R exp_3_rover_il_rl_s/models/model_last --mpc_update_freq 10 --no_viz
python test_3_rover.py --test --rl --il --rl_acc -R exp_3_rover_il_rl_a/models/model_last --mpc_update_freq 10 --no_viz


###### TEST ALL EXPERIMENTS (viz)
python test_1_car.py --test -P e1_traffic/models/model_49000.ckpt
python test_1_car.py --test --rl --rl_raw -R exp_1_car_rl_r/models/model_last --hiddens 64 64 64
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s/models/model_last --hiddens 64 64 64
python test_1_car.py --test --rl --rl_acc -R exp_1_car_rl_a/models/model_last --hiddens 64 64 64
python test_1_car.py --test --il -R exp_1_car_il/models/model_last
python test_1_car.py --test --rl --il --rl_raw -R exp_1_car_il_rl_r/models/model_last --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s/models/model_last --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_acc -R exp_1_car_il_rl_a/models/model_last --hiddens 64 64 64


python test_2_ship.py --test --num_trials 20 -P exp_4_ship_nn/models/model_49000.ckpt
python test_2_ship.py --test --rl --rl_raw --num_trials 20 -R exp_2_ship_rl_r/models/model_last
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s/models/model_last
python test_2_ship.py --test --rl --rl_acc --num_trials 20 -R exp_2_ship_rl_a/models/model_last 
python test_2_ship.py --test --il --num_trials 20 -R exp_2_ship_il/models/model_last
python test_2_ship.py --test --rl --il --rl_raw --num_trials 20 -R exp_2_ship_il_rl_r/models/model_last
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s/models/model_last
python test_2_ship.py --test --rl --il --rl_acc --num_trials 20 -R exp_2_ship_il_rl_a/models/model_last


python test_3_rover.py --test -P e5_rover_pret/models/model_249000.ckpt --mpc_update_freq 10
python test_3_rover.py --test --rl --rl_raw -R exp_3_rover_rl_r/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --rl --rl_stl -R exp_3_rover_rl_s/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --rl --rl_acc -R exp_3_rover_rl_a/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --il -R exp_3_rover_il/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --rl --il --rl_raw -R exp_3_rover_il_rl_r/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --rl --il --rl_stl -R exp_3_rover_il_rl_s/models/model_last --mpc_update_freq 10
python test_3_rover.py --test --rl --il --rl_acc -R exp_3_rover_il_rl_a/models/model_last --mpc_update_freq 10
ls