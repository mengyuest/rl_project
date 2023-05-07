import os
import time
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as ospj

eval_dir = ospj("..", "exps_stl", "eval_result")

EXPS = ["e1_car", "e2_game", "e3_ship_safe", "e4_ship_track", "e5_rover"]
METHODS = ["rl_raw", "rl_stl", "rl_acc", "mpc", "stl_planner", "sgd", "ours", "ours_ft"]
METRICS = ["acc", "t", "safety"] #["acc", "reward", "t", "safety", "battery", "goals"]

OUR_METHOD = "Ours"

def get_data(exp_name):
    res = {}
    for method_str in METHODS:
        data_path = ospj(eval_dir, "result_%s_%s.npz"%(exp_name, method_str))
        print(data_path)
        if os.path.exists(data_path):
            res[method_str]=np.load(data_path, allow_pickle=True)["data_avg"]
    return res

res_list = {}
for exp_name in EXPS:
    res_list[exp_name] = get_data(exp_name)

# plot
color_d = {
    "rl": "#9D9878",
    "rl_raw": "#9D9878",
    "rl_stl": "#8D8868",
    "rl_acc": "#7D7858",
    "mpc": "#C6C4C2",
    "stl_planner":  "#8AAA9A",
    "sgd": "#417A68",
    "ours": "#ed897b",
    "ours_ft": "#E7452E",
}

name_d = {
    "rl": "$RL$",
    "rl_raw": "$RL_{R}$",
    "rl_stl": "$RL_{S}$",
    "rl_acc": "$RL_{A}$",
    "mpc": "$MPC$",
    "stl_planner": "$STL_{M}$",
    "sgd": "$STL_{G}$",
    "ours": "$%s$"%(OUR_METHOD),
    "ours_ft": "$%s_{F}$"%(OUR_METHOD),
}

nt_d = {
    "e1_car": 25,
    "e2_game": 25, 
    "e3_ship_safe": 20, 
    "e4_ship_track": 20, 
    "e5_rover": 10
}

fontsize = 20
label_fontsize = 14
width=0.5

MERGE_RL = False

os.makedirs(ospj(eval_dir, "figs"), exist_ok=True)
for exp_name in EXPS:
    # first plot accuracy
    # next plot reward
    # then plot computation time
    res = res_list[exp_name]
    
    meta_names = [me_i for me_i in res]
    if MERGE_RL:
        meta_names=["rl"]+meta_names[3:]
    names = [name_d[me_i] for me_i in meta_names]
    xs = list(range(len(meta_names)))
    colors = [color_d[me_i] for me_i in meta_names]

    for metric in METRICS:
        nt = nt_d[exp_name]
        if metric in res["ours"].item():
            # ys = np.array([res[me_i].item()[metric] / (nt if metric=="t" and "rl" in me_i else 1) for me_i in res])
            ys = np.array([res[me_i].item()[metric] for me_i in res])
            if MERGE_RL:
                y_max = np.max(ys[:3])
                y_min = np.min(ys[:3])
                ys = np.concatenate((np.array([np.mean(ys[:3])]), ys[3:]))
                yerr_max = y_max - ys[0]
                yerr_min = ys[0] - y_min
                
            ax = plt.gca()
            if metric == "reward":
                ys += 3
            if metric == "t":
                # account for roll out whole seg
                # ys[0] *= 10
                # ys[1] *= 10
                ax.set_yscale("log")
            if metric == "acc":
                ys *= 100
                if MERGE_RL:
                    yerr_max *= 100
                    yerr_min *= 100
            plt.bar(xs, ys, width=width, color=colors)

            if MERGE_RL:
                plt.errorbar(xs[0], ys[0], yerr=[[yerr_min], [yerr_max]], fmt="o", color="black", capsize=4, capthick=1)

            ax.set_xticks(xs)
            # ax.set_xticklabels(names)

            if metric=="acc":
                plt.ylim(0, 100.0)
            elif metric != "t":
                plt.ylim(0, np.max(ys))
            
            # plt.axhline(y=np.mean(ys[:3]), color=color_d["rl"], linestyle="--", label="RL.avg")
            # plt.axhline(y=ys[-2], color=color_d["ours"], linestyle="--", label="Ours")
            
            # plt.xticks(fontsize=fontsize)
            # plt.xticks(fontsize=fontsize, rotation=30, ha='right')
            ax.set_xticklabels(names, fontsize=label_fontsize)
            # ax.set_xticklabels(names, fontsize=fontsize, rotation=30, ha='right')
            plt.yticks(fontsize=label_fontsize)
            plt.xlabel("Methods", fontsize=fontsize)
            if metric == "acc":
                plt.ylabel("STL satisfication rate (%)", fontsize=fontsize)
            elif metric == "t":
                plt.ylabel("Computation time (s)", fontsize=fontsize)
            else:
                plt.ylabel(metric, fontsize=fontsize)
            # plt.legend()
            plt.savefig("%s/%s_%s.png"%(ospj(eval_dir, "figs"), exp_name, metric), bbox_inches='tight', pad_inches=0.1)
            plt.close()
