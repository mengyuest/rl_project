import csv
import numpy as np
import matplotlib.pyplot as plt
import time

root_dir="../exps_stl/"
t1=time.time()
LOAD=True
LOAD_SMOOTH=True

OUR_NAME="Ours"

COLORS={
    "rl_raw": "orange", #"#9D9878",
    "rl_stl": "green",
    "rl_acc": "blue",
    "ours": "red",
}

std_s=["", "_1008", "_1009"]

paths=[
    ["exp1_ours", "exp1_rl_raw", "exp1_rl_stl", "exp1_rl_acc", ],
    ["exp2_ours", "exp2_rl_raw", "exp2_rl_stl", "exp2_rl_acc", ],
    ["exp3_ours", "exp3_rl_raw", "exp3_rl_stl", "exp3_rl_acc", ],
    ["exp4_ours", "exp4_rl_raw", "exp4_rl_stl", "exp4_rl_acc", ],
    ["exp5_ours_v1", "exp5_rl_raw", "exp5_rl_stl", "exp5_rl_acc", ],
]

curves = [[[None] * len(std_s) for _ in range(len(paths[ii]))] for ii in range(5)]

def smooth(ls, bandwidth=100):
    new_ls = []
    for i in range(len(ls)):
        new_ls.append(np.mean(ls[max(0,i-bandwidth):i+bandwidth]))
    return np.array(new_ls)

name_list=["Exp1-Traffic", "Exp2-Maze game", "Exp3-Ship1", "Exp4-Ship2", "Exp5-Rover"]

data_file_path = "%s/result_std.npz"%(root_dir)
smooth_data_file_path = "%s/result_std_smooth.npz"%(root_dir)
if LOAD_SMOOTH:
    curves = np.load(smooth_data_file_path, allow_pickle=True)["data"]
elif LOAD:
    curves = np.load(data_file_path, allow_pickle=True)["data"]
else:
    for mi in range(len(paths)):
        for li in range(len(paths[mi])):
            for ji, std_str in enumerate(std_s):
                curves[mi][li][ji] = {"steps":[], "r":[], "rs":[], 'acc':[], "times":[], }
                with open("%s/%s%s/monitor_full.csv"%(root_dir, paths[mi][li], std_str)) as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for row in spamreader:
                        s, r, rs, r_acc, t = row
                        curves[mi][li][ji]["steps"].append(float(s))
                        curves[mi][li][ji]["r"].append(float(r))
                        curves[mi][li][ji]["rs"].append(float(rs))
                        curves[mi][li][ji]["acc"].append(float(r_acc))
                        curves[mi][li][ji]["times"].append(float(t))
                curves[mi][li][ji]["r"] = np.array(curves[mi][li][ji]["r"])
                curves[mi][li][ji]["rs"] = np.array(curves[mi][li][ji]["rs"])
                curves[mi][li][ji]["acc"] = np.array(curves[mi][li][ji]["acc"])
    np.savez(data_file_path, data=curves)

def get_mean_std(data_list):
    max_len=data_list[0].shape[0]
    for data in data_list:
        if data.shape[0] < max_len:
            max_len = data.shape[0]
    new_data=[]
    for data in data_list:
        new_data.append(data[:max_len])
    new_data = np.stack(new_data, axis=0)
    means = np.mean(new_data, axis=0)
    stds = np.std(new_data, axis=0)
    return means, stds

def plt_proc(data, color, alpha, label, lw, rate):
    means, stds = get_mean_std([data[i]["acc"] for i in range(len(std_s))])
    plt.plot(list(range(means.shape[0]))[::rate], means[::rate], color=color, alpha=alpha, label=label, linewidth=lw)
    plt.fill_between(
        x=list(range(means.shape[0]))[::rate], 
        y1=np.clip(means[::rate]+stds[::rate] * STD_SCALE, 0, 100), 
        y2=means[::rate]-stds[::rate] * STD_SCALE, 
        color=color, alpha=alpha * 0.3, linewidth=0.1)

for mi in range(len(paths)):
    print(mi, name_list[mi])
    ratio = 1
    plt.figure(figsize=(6, 4))
    ax=plt.gca()
    BANDWIDTH = 100
    _RATE = 500
    if mi==4:
        RATE = _RATE*5
    else:
        RATE = _RATE*1
    FONTSIZE = 18
    LW = 2.0
    ALPHA=1.0
    STD_SCALE=3
    if LOAD_SMOOTH==False:
        for li in range(len(paths[mi])):
            for ji in range(len(std_s)):
                curves[mi][li][ji]["r"] = smooth(curves[mi][li][ji]["r"], bandwidth=BANDWIDTH)
                curves[mi][li][ji]["rs"] = smooth(curves[mi][li][ji]["rs"], bandwidth=BANDWIDTH)
                curves[mi][li][ji]["acc"] = smooth(curves[mi][li][ji]["acc"], bandwidth=BANDWIDTH)
        if mi==len(paths)-1:
            np.savez(smooth_data_file_path, data=curves)
    plt_proc(curves[mi][1], COLORS["rl_raw"], ALPHA, "$RL_{R}$", LW, RATE)
    plt_proc(curves[mi][2], COLORS["rl_stl"], ALPHA, "$RL_{S}$", LW, RATE)
    plt_proc(curves[mi][3], COLORS["rl_acc"], ALPHA, "$RL_{A}$", LW, RATE)
    plt_proc(curves[mi][0], COLORS["ours"], ALPHA, OUR_NAME, LW, RATE)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.xlabel("Training steps", fontsize=FONTSIZE)
    plt.ylabel("STL accuracy (%)", fontsize=FONTSIZE)
    if mi==4:
        plt.xlim(0, 250000)
    else:
        plt.xlim(0, 50000)
    if mi==3:
        plt.ylim(bottom=0)
    else:
        plt.ylim(0, 100)
    
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)

    plt.legend(loc="lower right", fontsize=FONTSIZE)
    plt.tight_layout()
    filename = "rewards"
    plt.savefig("%s/%s_%s.png"%(root_dir, filename, name_list[mi]), bbox_inches='tight', pad_inches=0.1)
    plt.close()

t2=time.time()
print("Finished in %.3f seconds"%(t2-t1))