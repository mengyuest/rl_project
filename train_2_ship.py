import argparse
import time
import numpy as np
from stl_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=50000)
    add("--lr", type=float, default=3e-5)
    add("--nt", type=int, default=20)
    add("--dt", type=float, default=0.15)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--sim_freq", type=int, default=1)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)

    add("--hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--stl_sim_steps", type=int, default=2)
    add("--n_obs", type=int, default=1)
    add("--obs_rmin", type=float, default=0.6)
    add("--obs_rmax", type=float, default=1.2)
    add("--river_width", type=float, default=4.0)
    add("--range_x", type=float, default=15.0)
    add("--thrust_max", type=float, default=0.5)
    add("--delta_max", type=float, default=3.0)
    add("--s_phimax", type=float, default=0.5)
    add("--s_umin", type=float, default=3.0)
    add("--s_umax", type=float, default=5.0)
    add("--s_vmax", type=float, default=0.3)
    add("--s_rmax", type=float, default=0.5)

    add("--canvas_h", type=float, default=4.0)
    add("--canvas_w", type=float, default=15.0)

    # CBF's configurations
    add("--train_cbf", action='store_true', default=False)
    add("--net_hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--cbf_hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--num_sim_steps", type=int, default=1)
    add("--cbf_pos_bloat", type=float, default=0.1)
    add("--cbf_neg_bloat", type=float, default=0.1)
    add("--cbf_gamma", type=float, default=0.1)
    add("--cbf_alpha", type=float, default=0.2)
    add("--cbf_cls_w", type=float, default=1)
    add("--cbf_dec_w", type=float, default=1)
    add("--cbf_prior_w", type=float, default=0.0)
    add("--cbf_nn_w", type=float, default=1.0)

    add("--dense_state_cls", action='store_true', default=False)
    add("--dense_state_dec", action='store_true', default=False)
    add("--num_dense_sample", type=int, default=10000)

    add("--alternative", action='store_true', default=False)
    add("--alternative2", action='store_true', default=False)
    add("--alternative_freq", type=int, default=50)

    add("--policy_pretrained_path", type=str, default=None)
    add("--qp", action='store_true', default=False)

    add("--both_state_cls", action='store_true', default=False)
    add("--both_state_dec", action='store_true', default=False)
    add("--dense_ratio", type=float, default=0.5)

    add("--mpc_update_freq", type=int, default=1)

    add("--u_loss", type=float, default=0.0)

    add("--river_w", type=float, default=10.0)
    add("--num_trials", type=int, default=1000)

    add("--track_thres", type=float, default=0.3)
    add("--tmax", type=int, default=25)
    add("--obs_ymin", type=float, default=-0.0)
    add("--obs_ymax", type=float, default=0.0)
    add("--obs_xmin", type=float, default=-1.0)
    add("--obs_xmax", type=float, default=8.0)

    add("--viz_cbf", action='store_true', default=False)
    add("--cbf_pretrained_path", type=str, default=None)
    add("--bloat_d", type=float, default=0.0)
    add("--origin_sampling", action='store_true', default=False)
    add("--origin_sampling2", action='store_true', default=False)
    add("--origin_sampling3", action='store_true', default=False)
    add("--dist_w", type=float, default=0.0)

    add("--test_pid", action='store_true', default=False)
    add("--diff_test", action='store_true', default=False)
    add("--obs_specific", action='store_true', default=False)


    # new-tricks
    # add("--hiddens", type=int, nargs="+", default=[64, 64, 64])
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=1.0)
    add("--update_init_freq", type=int, default=-1)
    add("--add_val", action="store_true", default=False)
    add("--include_first", action="store_true", default=False)

    # new framework specific
    add("--mode", type=str, choices=["car", "maze", "ship1", "ship2", "rover"], default="ship2")
    add("--train_rl", action='store_true', default=False)
    add("--num_workers", type=int, default=None)
    add("--stl_reward", action='store_true', default=False)
    add("--acc_reward", action='store_true', default=False)
    add("--c_val", type=float, default=0.5)

    add("--train_il", action='store_true', default=False)
    add("--il_mode", type=str, default="all", choices=["all", "traj", "first"])
    add("--rl_path", "-R", type=str, default=None)
    
    add("--learn", action="store_true", default=False)
    add("--load_others", action="store_true", default=False)
    add('--il_lr', type=float, default=None)
    add('--stl_ratio', type=float, default=1.0)
    add('--stl_cap', type=float, default=None)

    args = parser.parse_args()
    args.origin_sampling3 = True
    args.obs_specific = True
    t1=time.time()
    main(args)
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))