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
run_rlp python train_2_ship.py -e exp_2_ship_rl_r_1008 --train_rl --epochs 400000 --num_workers 8 --seed 1008
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_1008 --train_rl --epochs 400000 --num_workers 8 --stl_reward --seed 1008 
run_rlp python train_2_ship.py -e exp_2_ship_rl_a_1008 --train_rl --epochs 400000 --num_workers 8 --acc_reward --seed 1008

run_rlp python train_2_ship.py -e exp_2_ship_rl_r_1009 --train_rl --epochs 400000 --num_workers 8 --seed 1009
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_1009 --train_rl --epochs 400000 --num_workers 8 --stl_reward --seed 1009 
run_rlp python train_2_ship.py -e exp_2_ship_rl_a_1009 --train_rl --epochs 400000 --num_workers 8 --acc_reward --seed 1009
ls


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



# TEST success
python test_2_ship.py --test --num_trials 20 -P exp_4_ship_nn/models/model_49000.ckpt --no_viz
python test_2_ship.py --test --rl --rl_raw --num_trials 20 -R exp_2_ship_rl_r/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_acc --num_trials 20 -R exp_2_ship_rl_a/models/model_last --no_viz
python test_2_ship.py --test --il --num_trials 20 -R exp_2_ship_il/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_raw --num_trials 20 -R exp_2_ship_il_rl_r_new/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_new/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_acc --num_trials 20 -R exp_2_ship_il_rl_a_new/models/model_last --no_viz
ls


# train RL
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_r1c.5 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 1.0 --stl_cap 0.5
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_r10c.5 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 10.0 --stl_cap 0.5
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_r100c.5 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 100.0 --stl_cap 0.5
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_r1 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 1.0 
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_r10 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 10.0
run_rlp python train_2_ship.py -e exp_2_ship_rl_s_r100 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 100.0
ls

# test
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r1c.5/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r10c.5/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r100c.5/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r1/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r10/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r100/models/model_last --no_viz
ls







# 
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_r1c.5 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --stl_ratio 1.0 --stl_cap 0.5 --learn
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_r10c.5 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --stl_ratio 10.0 --stl_cap 0.5 --learn
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_r100c.5 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --stl_ratio 100.0 --stl_cap 0.5 --learn
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_r1 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --stl_ratio 1.0 --learn
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_r10 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --stl_ratio 10.0 --learn
run_rlp python train_2_ship.py -e exp_2_ship_il_rl_s_r100 --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --stl_reward --stl_ratio 100.0 --learn
ls

run_rlp python train_2_ship.py -e exp_2_ship_il_rl_a_new --train_rl --epochs 400000 --num_workers 8 -R exp_2_ship_il/models/model_last --acc_reward --learn



# Traffic case
# rl_stl 6x

# il_rl_r_new
run_rlp python train_1_car.py -e exp_1_car_il_rl_r_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --learn --hiddens 64 64 64

# il_rl_s_new
# run_rlp python train_1_car.py -e exp_1_car_il_rl_s_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn --hiddens 64 64 64

# il_rl_a_new
run_rlp python train_1_car.py -e exp_1_car_il_rl_a_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --acc_reward --learn --hiddens 64 64 64

# il_rl_s_new 6x
run_rlp python train_1_car.py -e exp_1_car_il_rl_s_r1c.5_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn --hiddens 64 64 64 --stl_ratio 1.0 --stl_cap 0.5
run_rlp python train_1_car.py -e exp_1_car_il_rl_s_r10c.5_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn --hiddens 64 64 64 --stl_ratio 10.0 --stl_cap 0.5
run_rlp python train_1_car.py -e exp_1_car_il_rl_s_r100c.5_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn --hiddens 64 64 64 --stl_ratio 100.0 --stl_cap 0.5
run_rlp python train_1_car.py -e exp_1_car_il_rl_s_r1_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn --hiddens 64 64 64 --stl_ratio 1.0
run_rlp python train_1_car.py -e exp_1_car_il_rl_s_r10_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn --hiddens 64 64 64 --stl_ratio 10.0 
run_rlp python train_1_car.py -e exp_1_car_il_rl_s_r100_new --train_rl --epochs 400000 --num_workers 8 -R exp_1_car_il/models/model_last --stl_reward --learn --hiddens 64 64 64 --stl_ratio 100.0

ls


run_rlp python train_1_car.py -e exp_1_car_rl_s_r1 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 1.0 --hiddens 64 64 64
run_rlp python train_1_car.py -e exp_1_car_rl_s_r10 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 10.0 --hiddens 64 64 64
run_rlp python train_1_car.py -e exp_1_car_rl_s_r100 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 100.0 --hiddens 64 64 64
run_rlp python train_1_car.py -e exp_1_car_rl_s_r1c.5 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 1.0 --stl_cap 0.5 --hiddens 64 64 64
run_rlp python train_1_car.py -e exp_1_car_rl_s_r10c.5 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 10.0 --stl_cap 0.5 --hiddens 64 64 64
run_rlp python train_1_car.py -e exp_1_car_rl_s_r100c.5 --train_rl --epochs 400000 --num_workers 8 --stl_reward --stl_ratio 100.0 --stl_cap 0.5 --hiddens 64 64 64
ls


#### TESTING (traffic)
python test_1_car.py --test -P e1_traffic/models/model_49000.ckpt --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --rl_raw -R exp_1_car_rl_r/models/model_last --no_viz --hiddens 64 64 64
# python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r1/models/model_last --no_viz --hiddens 64 64 64
# python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r10/models/model_last --no_viz --hiddens 64 64 64
# python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r100/models/model_last --no_viz --hiddens 64 64 64
# python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r1c.5/models/model_last --no_viz --hiddens 64 64 64
# python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r10c.5/models/model_last --no_viz --hiddens 64 64 64
# python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r100c.5/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --rl_acc -R exp_1_car_rl_a/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --il -R exp_1_car_il/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_raw -R exp_1_car_il_rl_r_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_acc -R exp_1_car_il_rl_a_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r1_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r10_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r100_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r1c.5_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r10c.5_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r100c.5_new/models/model_last --no_viz --hiddens 64 64 64
ls

### test for il_rl_stl
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r1_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r10_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r100_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r1c.5_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r10c.5_new/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r100c.5_new/models/model_last --no_viz --hiddens 64 64 64

python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r1/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r10/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r100/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r1c.5/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r10c.5/models/model_last --no_viz --hiddens 64 64 64
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r100c.5/models/model_last --no_viz --hiddens 64 64 64
ls


#### TESTING (ship)
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r1/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r10/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r100/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r1c.5/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r10c.5/models/model_last --no_viz
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r100c.5/models/model_last --no_viz

python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r1/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r10/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r100/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r1c.5/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r10c.5/models/model_last --no_viz
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r100c.5/models/model_last --no_viz
ls

# ablation study

# test for ablation study


### test for il_rl_stl
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r1_new/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r10_new/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r100_new/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r1c.5_new/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r10c.5_new/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --il --rl_stl -R exp_1_car_il_rl_s_r100c.5_new/models/model_last --no_viz --hiddens 64 64 64 --seed 1008

python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r1/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r10/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r100/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r1c.5/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r10c.5/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
python test_1_car.py --test --rl --rl_stl -R exp_1_car_rl_s_r100c.5/models/model_last --no_viz --hiddens 64 64 64 --seed 1008
ls


#### TESTING (ship)
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r1/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r10/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r100/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r1c.5/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r10c.5/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --rl_stl --num_trials 20 -R exp_2_ship_rl_s_r100c.5/models/model_last --no_viz --seed 1008

python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r1/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r10/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r100/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r1c.5/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r10c.5/models/model_last --no_viz --seed 1008
python test_2_ship.py --test --rl --il --rl_stl --num_trials 20 -R exp_2_ship_il_rl_s_r100c.5/models/model_last --no_viz --seed 1008
ls
