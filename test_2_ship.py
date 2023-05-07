from stl_lib import *
from matplotlib.patches import Polygon, Rectangle, Ellipse
import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, soft_step, build_relu_nn, get_exp_dir, eval_proc 

plt.rcParams.update({'font.size': 12})


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        # input (x, y, phi, u, v, r, obs_x, obs_y, obs_r, T)
        input_dim = 10
        output_dim = 2 * args.nt
        self.net = build_relu_nn(input_dim, output_dim, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):     
        num_samples = x.shape[0]
        u = self.net(x).reshape(num_samples, args.nt, -1)
        u0 = torch.tanh(u[..., 0]) * args.thrust_max
        u1 = torch.tanh(u[..., 1]) * args.delta_max
        uu = torch.stack([u0, u1], dim=-1)
        return uu

def dynamics(x0, u):
    t = u.shape[1]
    x = x0.clone()
    segs=[x0]
    for ti in range(t):
        new_x = dynamics_s(x, u[:, ti], num=args.stl_sim_steps)
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)

def dynamics_s(x, uu, num=1):
    for tti in range(num):
        dt = (args.dt/num)
        new_x = torch.zeros_like(x)
        # (x, y, phi, u, v, r)
        new_dx = x[:, 3] * torch.cos(x[:, 2]) - x[:, 4] * torch.sin(x[:, 2])
        new_dy = x[:, 3] * torch.sin(x[:, 2]) + x[:, 4] * torch.cos(x[:, 2])
        new_dphi = x[:, 5]
        new_du = uu[:, 0]
        new_dv = uu[:, 1] * 0.01
        new_dr = uu[:, 1] * 0.5
        new_dT = -soft_step(x[:, 1]**2-args.track_thres**2)

        zeros = 0 * new_dx
        dsdt = torch.stack([new_dx, new_dy, new_dphi, new_du, new_dv, new_dr] + [zeros, zeros, zeros, new_dT], dim=-1)
        new_x = x + dsdt * dt
        new_xx = new_x.clone()
        new_xx[:, 2] = torch.clamp(new_x[:, 2], -args.s_phimax, args.s_phimax)
        new_xx[:, 3] = torch.clamp(new_x[:, 3], args.s_umin, args.s_umax)
        new_xx[:, 4] = torch.clamp(new_x[:, 4], -args.s_vmax, args.s_vmax)
        new_xx[:, 5] = torch.clamp(new_x[:, 5], -args.s_rmax, args.s_rmax)
        
        x = new_xx
    return new_xx

def get_rl_xs_us(x, policy, nt):
    xs = [x]
    us = []
    dt_minus = 0
    for ti in range(nt):
        tt1=time.time()
        u, _ = policy.predict(x.cpu(), deterministic=True)
        u = torch.from_numpy(u).cuda()
        u[..., 0] = torch.clip(u[..., 0] * args.thrust_max, -args.thrust_max, args.thrust_max)
        u[..., 1] = torch.clip(u[..., 1] * args.delta_max, -args.delta_max, args.delta_max)
        new_x = dynamics_s(x, u, num=args.stl_sim_steps)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti > 0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.stack(us, dim=1)  # (N, 2) -> (N, T, 2)
    return xs, us, dt_minus

def initialize_x_cycle(n, is_cbf=False):   
    scene_type = rand_choice_tensor([0, 1, 2, 3], (n, 1))
    # without obs case
    s0_x = uniform_tensor(0, 0, (n, 1))
    if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
        s0_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
        s0_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
    else:
        s0_y = uniform_tensor(-0.5, 0.5, (n, 1))
        s0_phi = uniform_tensor(-args.s_phimax/2, args.s_phimax/2, (n, 1))
    s0_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
    s0_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
    s0_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
    s0_obs_x = uniform_tensor(-5, -5, (n, 1))
    s0_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
    s0_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
    if args.origin_sampling:
        s0_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
    else:
        s0_obs_T = rand_choice_tensor([i * args.dt for i in range(1, 10)], (n, 1))
    
    # far from obs case
    s1_x = uniform_tensor(0, 0, (n, 1))
    if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
        s1_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
        s1_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
    else:
        s1_y = uniform_tensor(-0.5, 0.5, (n, 1))
        s1_phi = uniform_tensor(-args.s_phimax/2, args.s_phimax/2, (n, 1))
    s1_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
    s1_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
    s1_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
    s1_obs_x = uniform_tensor(5, args.obs_xmax, (n, 1))
    s1_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
    s1_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
    if args.origin_sampling:
        s1_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
    elif args.origin_sampling3:
        s1_obs_T = rand_choice_tensor([i * args.dt for i in range(10, args.tmax+1)], (n, 1))
    else:
        s1_obs_T = rand_choice_tensor([i * args.dt for i in range(12, args.tmax+1)], (n, 1))

    ymin = 0.8
    ymax = args.river_width/2
    flip = rand_choice_tensor([-1, 1], (n, 1))
    # closer from obs case (before meet)
    s2_x = uniform_tensor(0, 0, (n, 1))
    if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
        s2_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
    else:
        s2_y = uniform_tensor(ymin, ymax, (n, 1)) * flip
    s2_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
    s2_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
    s2_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
    s2_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
    s2_obs_x = uniform_tensor(0, 5, (n, 1))
    s2_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
    s2_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
    if args.origin_sampling:
        s2_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
    elif args.origin_sampling3:
        s2_obs_T = rand_choice_tensor([i * args.dt for i in range(8, 15)], (n, 1))
    else:
        s2_obs_T = rand_choice_tensor([i * args.dt for i in range(10, 15)], (n, 1))

    # closer from obs case (after meet)
    s3_x = uniform_tensor(0, 0, (n, 1))
    if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
        s3_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
    else:
        s3_y = uniform_tensor(ymin, ymax, (n, 1)) * flip
    s3_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
    s3_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
    s3_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
    s3_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
    s3_obs_x = uniform_tensor(-1, 0, (n, 1))
    s3_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
    s3_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
    if args.origin_sampling:
        s3_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
    elif args.origin_sampling3:
        s3_obs_T = rand_choice_tensor([i * args.dt for i in range(5, 12)], (n, 1))
    else:
        s3_obs_T = rand_choice_tensor([i * args.dt for i in range(8, 12)], (n, 1))

    x = mux(scene_type, s0_x, s1_x, s2_x, s3_x)
    y = mux(scene_type, s0_y, s1_y, s2_y, s3_y)
    phi = mux(scene_type, s0_phi, s1_phi, s2_phi, s3_phi)
    u = mux(scene_type, s0_u, s1_u, s2_u, s3_u)
    v = mux(scene_type, s0_v, s1_v, s2_v, s3_v)
    r = mux(scene_type, s0_r, s1_r, s2_r, s3_r)
    obs_x = mux(scene_type, s0_obs_x, s1_obs_x, s2_obs_x, s3_obs_x)
    obs_y = mux(scene_type, s0_obs_y, s1_obs_y, s2_obs_y, s3_obs_y)
    obs_r = mux(scene_type, s0_obs_r, s1_obs_r, s2_obs_r, s3_obs_r)
    obs_T = mux(scene_type, s0_obs_T, s1_obs_T, s2_obs_T, s3_obs_T)

    rand_zero = rand_choice_tensor([0, 1], (n, 1))
    if is_cbf:
        y = y * 1.2
    res = torch.cat([x, y, phi, u, v, r, obs_x, obs_y, obs_r, obs_T], dim=1)
    return res

def mux(scene_type, x0, x1, x2, x3):
    return (scene_type==0).float() * x0 + (scene_type==1).float() * x1 + (scene_type==2).float() * x2 + (scene_type==3).float() * x3


def initialize_x(n):
    x_list = []
    total_n = 0
    while(total_n<n):
        x_init = initialize_x_cycle(n)
        # safe_bloat = 1.5
        safe_bloat = args.bloat_d
        dd = 5
        n_res = 100
        crit_list = []
        crit1 = torch.norm(x_init[:, :2] - x_init[:, 6:6+2], dim=-1) > x_init[:, 8] + safe_bloat
        crit2 = torch.logical_not(torch.logical_and(x_init[:,1]>1.5, x_init[:,2]>0))  # too close from the river boundary
        crit3 = torch.logical_not(torch.logical_and(x_init[:,1]<-1.5, x_init[:,2]<0))  # too close from the river boundary
        if args.origin_sampling3:
            # cannot be too close to the obstacle
            crit4 = torch.logical_not(torch.logical_and(torch.logical_and(x_init[:,6]-x_init[:,0]<x_init[:,8]+0.5, x_init[:,6]-x_init[:,0]>0), torch.abs(x_init[:,1]-x_init[:,7])<x_init[:,8]))
            # cannot be too close to the obstacle
            crit7 = torch.logical_not(torch.logical_and(torch.logical_and(x_init[:,6]-x_init[:,0]<x_init[:,8]+1.5, x_init[:,6]-x_init[:,0]>0), torch.abs(x_init[:,1]-x_init[:,7])<0.3))
            # should have enough time to escape
            crit5 = torch.logical_not(torch.logical_and(x_init[:, 9] < 5 * args.dt, torch.abs(x_init[:,1]) > 1.5))
            # too large angle
            crit6 = torch.logical_not(torch.logical_or(
                torch.logical_and(x_init[:, 1] > 1.5, x_init[:, 2] > args.s_phimax/2), 
                torch.logical_and(x_init[:, 1] < -1.5, x_init[:, 2] < -args.s_phimax/2), 
             ))
            valids_indices = torch.where(torch.all(torch.stack([crit1, crit2, crit3, crit4, crit5, crit6, crit7], dim=-1),dim=-1)>0)
        else:
            valids_indices = torch.where(torch.all(torch.stack([crit1, crit2, crit3], dim=-1),dim=-1)>0)
        x_val = x_init[valids_indices]
        total_n += x_val.shape[0]
        x_list.append(x_val)
    x_list = torch.cat(x_list, dim=0)[:n]
    return x_list

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        input_dim = 10
        output_dim = 2
        self.net = build_relu_nn(input_dim, output_dim, args.net_hiddens, activation_fn=nn.ReLU)

    def forward(self, x, k1=None, k2=None, k3=None, k4=None, k5=None):
        num_samples = x.shape[0]
        x_enc = x.clone()
        x_enc[:, 0] = 0
        x_enc[:, 1:6] = x[:, 1:6]
        x_enc[:, 6] = x[:, 6] - x[:, 0]
        x_enc[:, 7] = x[:, 7]
        x_enc[:, 8] = x[:, 8]
        x_enc[:, 9] = x[:, 9]
        u = self.net(x_enc).reshape(num_samples, -1)
        
        uref0 = - 5 * (x_enc[:, 3] - 4)
        uref1 = - 3 * (x_enc[:, 1] - 0) - 5 * (x_enc[:, 2] - 0)
        u0 = torch.clip(torch.tanh(u[..., 0]) * args.thrust_max + uref0, -args.thrust_max, args.thrust_max)
        u1 = torch.clip(torch.tanh(u[..., 1]) * args.delta_max + uref1, -args.delta_max, args.delta_max)
        uu = torch.stack([u0, u1], dim=-1)
        return uu

def check_safety_stl(x):
    dist1 = torch.norm(x[..., :2] - x[..., 6:8], dim=-1) - x[..., 8]
    dist3 = args.river_width/2 - torch.abs(x[..., 1]) 
    acc = torch.all(torch.logical_and(torch.logical_and(dist1>=0, dist3>=0), x[..., 9]>=0), dim=-1).float()
    inl = torch.all(dist3>=0, dim=-1).float()
    return acc, inl

def sim_visualization(epi, init_np, seg_np, acc_np, v_np=None):
    plt.figure(figsize=(12, 9))
    col = 5
    row = 5
    bloat = 0.0
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            ax = plt.subplot(row, col, idx+1)
            idx = min(i * col + j, seg_np.shape[0]-1)
            ax.add_patch(Rectangle([0, -args.canvas_h/2], args.canvas_w, args.canvas_h, color="green" if acc_np[idx]>0.5 else "red", alpha=0.1))
            offset = 6
            ax.add_patch(Ellipse([seg_np[idx, 0, offset], seg_np[idx, 0, offset + 1]], seg_np[idx, 0, offset + 2] * 2, seg_np[idx, 0, offset + 2] * 2, 
                                    label="obstacle", color="gray", alpha=0.8))
            ax.add_patch(Ellipse([seg_np[idx, 0, 0], seg_np[idx, 0, 1]], 0.5, 0.5, 
                                        label="ego", color="blue", alpha=0.8))
            plt.plot(seg_np[idx, :, 0], seg_np[idx, :, 1], label="trajectory", color="blue", linewidth=2, alpha=0.5)
            for ti in range(0, args.nt, 2):
                ax.text(seg_np[idx, ti, 0]+0.25, seg_np[idx, ti, 1]+0.25, "%.1f"%(seg_np[idx, ti, -1]), fontsize=6)
            if v_np is not None:
                plt.plot(seg_np[idx, :, 0], v_np[idx, :] * 1.8, label="CBF value (x1.8)", color="red", linewidth=2, alpha=0.3)
            if idx==0:
                plt.legend(fontsize=6, loc="lower right")
            ax.axis("scaled")
            plt.xlim(0-bloat, args.canvas_w+bloat)
            plt.ylim(-args.canvas_h/2-bloat, args.canvas_h/2+bloat)

    figname="%s/iter_%05d.png"%(args.viz_dir, epi)
    plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def test_ship(x_init, net_stl, net_cbf, rl_policy, stl):
    metrics_str=["acc", "reward", "score", "t", "safety"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.ship_env import ShipEnv
    args.mode="ship2"
    the_env = ShipEnv(args)

    # a long map of 2d obstacles
    test_cbf = False
    state_str = "TEST_STL"
    nt = args.test_nt
    n_obs = 100
    n_trials = args.num_trials
    update_freq = args.mpc_update_freq
    reset_T = rand_choice_tensor(list(range(15, 25)), (n_trials, n_obs)) * args.dt
    debug_t1 = time.time()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    obs_list=[]
    obs_x = x_init[0:1, 6].item()
    for i in range(n_obs):
        obs_x = obs_x + uniform_tensor(15, 25, (1, 1))
        obs_y = obs_x * 0 + 0
        obs_r = obs_x * 0 + uniform_tensor(args.obs_rmin, args.obs_rmax, (1, 1))
        if args.obs_specific:
            if obs_r<0.8:
                reset_T[:, i] = 15 * args.dt
            elif obs_r<1.0:
                reset_T[:, i] = 20 * args.dt
            else:
                reset_T[:, i] = 25 * args.dt
        obs = torch.cat([obs_x, obs_y, obs_r], dim=-1)
        obs_list.append(obs)
    obs_map = torch.stack(obs_list, dim=1)[0]  # (M, 3)

    x_init[0:n_trials, 1] = x_init[0:n_trials, 1] * 0.5
    x_init[0:n_trials, 2] = 0
    x_init[0:n_trials, 6] = obs_map[0:1, 0]
    x_init[0:n_trials, 7] = obs_map[0:1, 1] 
    x_init[0:n_trials, 8] = obs_map[0:1, 2]
    x_init[0:n_trials, 9] = 15 * args.dt
    x = x_init[0:n_trials].cpu()
    base_i = [0] * n_trials  # look at the first and the second one
    fs_list = []
    history = []
    seg_list = []

    # statistics
    safety = 0
    move_distance = 0
    cnt = [0] * n_trials
    seg = None
    collide = np.zeros((nt, n_trials))
    real_collide = np.zeros((nt, n_trials))
    cbf_record = [0] * n_trials
    prev_x_input = None
    x_input_list = []
    pwl_list_list = []
    for ti in range(nt):
        if ti % 10 == 0:
            print(ti)
        shall_update = [False] * n_trials
        updated_obs = [False] * n_trials
        x_input = x.cuda()

        for i in range(n_trials):
            if obs_map[base_i[i], 0] - x[i, 0] < -2:
                base_i[i] += 1
                x[i, 6:6+3] = obs_map[base_i[i]]
                x_input[i, 6:6+3] = obs_map[base_i[i]].cuda()
                updated_obs[i] = True
            if torch.norm(x[i, :2]-x[i, 6:8], dim=-1)<x[i, 8] or torch.abs(x[i,1]) > args.river_width/2 or (ti-1>=0 and collide[ti-1, i] == 1) or x[i,9]<0:
                collide[ti, i] = 1
            if torch.norm(x[i, :2]-x[i, 6:8], dim=-1)<x[i, 8] or torch.abs(x[i,1]) > args.river_width/2 or (ti-1>=0 and real_collide[ti-1, i] == 1):
                real_collide[ti, i] = 1
            if cnt[i] % update_freq == 0 or updated_obs[i]:
                shall_update[i] = True
                cnt[i] = 0
                dx = x_input[i, 0]
                x_input[i, 6] = x_input[i, 6] - dx
                if x_input[i, 6] > args.obs_xmax:
                    x_input[i, 6] = -5
                else:  # real obstacle coming
                    if prev_x_input is not None and (prev_x_input[i, 6] == -5 or updated_obs[i]):
                        # reset T
                        x[i, 9] = reset_T[i, base_i[i]]

                x_input[i, 0] = x_input[i, 0] - dx
        x_input_list.append(x_input)
        debug_tt1=time.time()
        dt_minus = 0
        
        if args.rl:
            _, u, dt_minus = get_rl_xs_us(x_input, rl_policy, args.nt)
            seg_out = dynamics(x.cpu(), u.cpu())
        else:
            u = net_stl(x_input)
            seg_out = dynamics(x.cpu(), u.cpu())
        debug_tt2=time.time()
        seg_total = seg_out.clone()
        
        # EVALUATION
        debug_dt = debug_tt2 - debug_tt1
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(the_env.generate_reward_batch(to_np(seg_total[:,0])))
        
        print(ti, acc[:,0])
        safety = 1-np.mean(collide[ti])
        metrics["t"].append(debug_dt - dt_minus)
        metrics["safety"].append(safety)
        metrics["acc"].append(acc_avg)
        metrics["score"].append(score_avg)
        metrics["reward"].append(reward)

        if seg is None:
            seg = seg_out
        else:
            seg = seg_list[-1].clone()
            for i in range(n_trials):
                if shall_update[i]:
                    seg[i] = seg_out[i].detach().cpu()
        seg_list.append(seg.detach().cpu())
        history.append(x.clone())
        for i in range(n_trials):
            x[i] = seg[i, cnt[i]+1].detach().cpu()
        for i in range(n_trials):
            cnt[i] += 1
        
        prev_x_input = x_input.clone()

    print(metrics["acc"])
    print("Real  safe:%.3f"%(np.mean(1-real_collide[-1])))
    print("Total safe:%.3f"%(np.mean(1-collide[-1])))
    print("Total acc: %.3f"%(np.mean(np.array(metrics["acc"]))))
    eval_proc(metrics, "e4_ship_track", args)
    if args.no_viz:
        return

    history = torch.stack(history, dim=1)
    seg_list = torch.stack(seg_list, dim=1)
    
    # visualization
    bloat = 1.0
    ratio = 1.0  # make space for ship
    ship_ratio = 1 - ratio
    ship_ratio = 0.2
    ratio = 0.8
    bloat = 0.5
    extend_x = 10
    r = np.sqrt(2)/2
    bk = np.sqrt(1 - r**2) 
    poly_ship = np.array([
        [1, 0],
        [0, r],
        [-bk, r],
        [-bk, -r],
        [0, -r]
    ])
    poly_ship = poly_ship * ship_ratio
    for ti in range(nt):
        if ti % args.sim_freq == 0 or ti == nt - 1:
            print(state_str, "Viz", ti, 1-np.mean(collide[ti]), 1-np.mean(real_collide[ti]))
            fig = plt.figure(figsize=(8.5, 2.5))
            ax = plt.gca()
            for obs_i in range(n_obs):
                ax.add_patch(Ellipse([obs_map[obs_i, 0], obs_map[obs_i, 1]], obs_map[obs_i, 2] * 2 * ratio, obs_map[obs_i, 2] * 2 * ratio, 
                    label="obstacle" if obs_i==0 else None, color="gray", alpha=0.8))
            i_cnt = 0
            for i in range(n_trials):
                i_cnt+=1
                s = to_np(history[i, ti])
                poly_ship_t = np.array(poly_ship)
                poly_ship_t[:, 0] = poly_ship[:, 0] * np.cos(s[2]) - poly_ship[:, 1] * np.sin(s[2])
                poly_ship_t[:, 1] = poly_ship[:, 0] * np.sin(s[2]) + poly_ship[:, 1] * np.cos(s[2])
                poly_ship_t[:, 0] += s[0]
                poly_ship_t[:, 1] += s[1]
                ax.add_patch(Polygon(poly_ship_t, label="ego ship" if i_cnt==1 else None, color="brown", alpha=1.0, zorder=100))
                plt.plot(seg_list[i, ti, :, 0], seg_list[i, ti, :, 1], label="trajectory" if i_cnt==1 else None, color="green", linewidth=1.5, alpha=0.5)
            idx = 0 # camera idx
            s = history[idx, ti]
            RY = args.river_width/2
            ax.add_patch(Rectangle([s[0]-bloat, -RY], args.canvas_w+extend_x+bloat, 2*RY, label="river", color="blue", alpha=0.3))
            plt.legend(fontsize=12, loc="lower right")
            ax.axis("scaled")

            ppd = 72./fig.dpi  # points per dot
            mybbox = ax.get_position()
            mybbox.x0, mybbox.y0 = fig.transFigure.inverted().transform(((20)/ppd, 20/ppd))
            mybbox.x1, mybbox.y1 = fig.transFigure.inverted().transform(((700-120)/ppd, 150/ppd))
            ax.set_position(mybbox)

            plt.xlim(s[0]-bloat, s[0]+args.canvas_w+extend_x+bloat)
            plt.ylim(-RY-bloat, RY+bloat)
            debug_input = to_np(x_input_list[ti])
            plt.title("Simulation (%04d/%04d) Safe:%.1f%%"%(ti, nt, 100*(1-np.mean(collide[ti]))))
            figname="%s/t_%03d.png"%(args.viz_dir, ti)
            plt.savefig(figname)
            fs_list.append(figname)
            plt.close()
    
    print("Real  safe:%.3f"%(np.mean(1-real_collide[-1])))
    print("Total safe:%.3f"%(np.mean(1-collide[-1])))
    print("Total acc: %.3f"%(np.mean(np.array(metrics["acc"]))))

    os.makedirs("%s/animation"%(args.viz_dir), exist_ok=True)
    generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.1, fs_list)
    debug_t2 = time.time()
    print("Finished in %.2f seconds"%(debug_t2 - debug_t1))


def main():
    # Experiment setup
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)    
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
    
    # STL definition
    avoid = Always(0, args.nt, AP(lambda x: torch.norm(x[..., :2] - x[..., 6:6+2], dim=-1)**2 - x[..., 6+2]**2 - args.bloat_d**2, comment="Avoid Obs"))
    in_river = Always(0, args.nt, AP(lambda x: (args.river_width/2)**2-x[..., 1]**2, comment="In River"))
    diverge = AP(lambda x: - args.track_thres**2 + x[..., 1]**2, comment="Diverge")
    in_T = AP(lambda x: x[..., 9], comment="In Time")
    track = Always(0, args.nt, Not(diverge))
    finally_track = Until(0, args.nt, in_T, track)
    stl = ListAnd([avoid, in_river, finally_track])

    # Train NN stl policy
    print(stl)
    stl.update_format("word")
    print(stl)

    # initialize x
    x_init = initialize_x(args.num_samples).float().cuda()

    # testing case
    net_stl = Policy(args).cuda()
    if args.rl:
        from stable_baselines3 import SAC, PPO, A2C
        rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
        test_ship(x_init, net_stl, None, rl_policy, stl)
    if args.net_pretrained_path is not None:
        net_stl.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
        test_ship(x_init, net_stl, None, None, stl)
    return


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
    add("--n_obs", type=int, default=3)
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

    add("--canvas_h", type=float, default=10.0)
    add("--canvas_w", type=float, default=10.0)

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

    add("--mpc", action='store_true', default=False)
    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)

    add("--plan", action="store_true", default=False)
    add("--rl", action="store_true", default=False)
    add("--rl_path", "-R", type=str, default=None)
    add("--rl_stl", action="store_true", default=False)
    add("--rl_acc", action="store_true", default=False)
    add("--eval_path", type=str, default="eval_result")
    add("--no_viz", action="store_true", default=False)

    add("--test_nt", type=int, default=200)

    add("--finetune", action="store_true", default=False)
    add("--solve2", action="store_true", default=False)

    add("--backup", action='store_true', default=False)
    add("--not_use_backup", action='store_true', default=False)
    add("--video", action='store_true', default=False)

    add("--color", action='store_true', default=False)
    args = parser.parse_args()

    args.origin_sampling3 = True
    args.obs_specific = True
    args.diff_test = True

    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))