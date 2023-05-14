import os
import numpy as np
from os.path import join as ospj

eval_dir = ospj("..", "exps_rlp", "eval_result")
EXPS = ["e1_car", "e4_ship_track"]
# METHODS = ["ours", "rl_raw", "rl_stl_r1", "rl_stl_r10", "rl_stl_r100", "rl_stl_r1c.5", "rl_stl_r10c.5", "rl_stl_r100c.5", 
#                 "rl_acc", "il", "il_rl_stl_r1", "il_rl_stl_r10", "il_rl_stl_r100", 
#                 "il_rl_stl_r1c.5", "il_rl_stl_r10c.5", "il_rl_stl_r100c.5", 
#                 "il_rl_acc"]
METHODS = ["rl_stl_r1", "rl_stl_r10", "rl_stl_r100", "rl_stl_r1c.5", "rl_stl_r10c.5", "rl_stl_r100c.5", 
                "il_rl_stl_r1", "il_rl_stl_r10", "il_rl_stl_r100", 
                "il_rl_stl_r1c.5", "il_rl_stl_r10c.5", "il_rl_stl_r100c.5", ]
path_d = {x:x for x in METHODS}
path_d = {x:x+"_1008" for x in path_d}
res={}
for exp_name in EXPS:
    res[exp_name]={}
    for method_str in METHODS:        
        data_path = ospj(eval_dir, "result_%s_%s.npz"%(exp_name, path_d[method_str]))
        # print(data_path)
        # if os.path.exists(data_path):
        acc = np.load(data_path, allow_pickle=True)["data_avg"].item()["acc"]
        res[exp_name][method_str]=acc
        print("%15s %15s %.4f"%(exp_name, method_str, acc))


for k in [1, 10, 100]:
    prev_accs=[]
    for gamma_i in [0, 1]:
        cstr = ["", "c.5"]
        accs=[
            res["e1_car"]["rl_stl_r%s%s"%(k, cstr[gamma_i])], 
            res["e1_car"]["il_rl_stl_r%s%s"%(k, cstr[gamma_i])], 
            res["e4_ship_track"]["rl_stl_r%s%s"%(k, cstr[gamma_i])], 
            res["e4_ship_track"]["il_rl_stl_r%s%s"%(k, cstr[gamma_i])]
        ]
        prelist=["$%s$"%(k), "$\infty$" if gamma_i==0 else "$0.5$"]
        accs_str = []
        symbol = {True: "\\textcolor{OliveGreen}{ (%.2f \\uparrow)}" , False: "\\textcolor{RubineRed}{ (%.2f \\downarrow)}"}
        print(" & ".join( prelist+ ["$%.2f \\,%s$"%(100*accs[i], symbol[accs[i]>accs[i-1]]%(abs(accs[i]-accs[i-1])*100) ) if i%2==1 else "$%.2f$"%(100*accs[i]) for i in range(4)]) + "\\\\")
        prev_accs = accs