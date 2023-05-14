import os
import time
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as ospj

eval_dir = ospj("..", "exps_rlp", "eval_result")

EXPS = ["e1_car", "e4_ship_track", "e5_rover"]
# METHODS = ["ours", "rl_raw", "rl_stl", "rl_acc", "il", "il_rl_raw", "il_rl_stl", "il_rl_acc"]
METHODS = ["ours", "rl_raw", "rl_acc", "il", "il_rl_acc"]
# METHODS = ["ours", "rl_raw", "rl_acc", "il"]
METRICS = ["acc"]  # ["acc", "reward", "t", "safety", "battery", "goals"]

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
# color_d = {
#     "ours": "#9D9878",
#     "rl_raw": "#9D9878",
#     "rl_stl": "#8D8868",
#     "rl_acc": "#7D7858",
#     "il": "#C6C4C2",
#     "il_rl_raw":  "#8AAA9A",
#     "il_rl_stl": "#417A68",
#     "il_rl_acc": "#ed897b",
# }

# color_d = {
#     "ours": "#8AAA9A",
#     "rl_raw": "#9D9878",
#     "rl_stl": "#8D8868",
#     "rl_acc": "#ed897b",
#     "il": "#C6C4C2",
#     "il_rl_raw":  "#8AAA9A",
#     "il_rl_stl": "#417A68",
#     "il_rl_acc": "#ed897b",
# }


color_d = {
    "ours": "#f37c80",
    "rl_raw": "#f2c472",
    "rl_stl": "#8D8868",
    "rl_acc": "#a7cd95",
    "il": "#C6C4C2",
    "il_rl_raw":  "#8AAA9A",
    "il_rl_stl": "#417A68",
    "il_rl_acc": "#5f939e",
}

# color_d = {
#     "ours": "#417A68",
#     "rl_raw": "#f37c80",
#     "rl_stl": "#8D8868",
#     "rl_acc": "#a7cd95",
#     "il": "#C6C4C2",
#     "il_rl_raw":  "#8AAA9A",
#     "il_rl_stl": "#ed897b",
#     "il_rl_acc": "#5f939e",
# }


name_d = {
    "ours": "Expert",
    "rl_raw": "$RL_{Heur}$",
    "rl_stl": "$RL_{S}$",
    "rl_acc": "$RL_{STL}$",
    "il": "$IL$",
    "il_rl_raw": "$IL$-$RL_{R}$",
    "il_rl_stl": "$IL$-$RL_{S}$",
    "il_rl_acc": "$IL$-$RL_{STL}$",
}

# fontsize = 16
# label_fontsize = 12
fontsize = 20
label_fontsize = 20
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
        if metric in res["ours"].item():
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

            if metric=="acc":
                plt.ylim(0, 85.0)
            elif metric != "t":
                plt.ylim(0, np.max(ys))

            if metric == "acc" and exp_name == "e4_ship_track":
                print("asd")
                plt.axhline(100*res["ours"].item()["acc"], linestyle="--", linewidth=2.0, color="black")

            ax.set_xticklabels(names, fontsize=label_fontsize)
            plt.yticks(fontsize=label_fontsize)
            plt.xlabel("Methods", fontsize=fontsize)
            if metric == "acc":
                plt.ylabel("STL accuracy (%)", fontsize=fontsize)
            elif metric == "t":
                plt.ylabel("Computation time (s)", fontsize=fontsize)
            else:
                plt.ylabel(metric, fontsize=fontsize)
            plt.savefig("%s/%s_%s.png"%(ospj(eval_dir, "figs"), exp_name, metric), bbox_inches='tight', pad_inches=0.1)
            plt.close()
