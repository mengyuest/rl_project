from stl_lib import *
from matplotlib.patches import Polygon, Rectangle, Circle
import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, \
            check_pts_collision, check_seg_collision, soft_step, to_torch, \
            pts_in_poly, seg_int_poly, build_relu_nn, soft_step_hard, get_exp_dir, \
            eval_proc, xxyy_2_Ab

plt.rcParams.update({'font.size': 12})


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        # input  (rover xy; dest xy; charger xy; battery t; hold_t)
        # output (rover v theta; astro v theta)
        self.net = build_relu_nn( 2 + 2 + 2 + 2, 2 * args.nt, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):     
        num_samples = x.shape[0]
        u = self.net(x).reshape(num_samples, args.nt, -1)
        if self.args.no_tanh:
            u0 = torch.clip(u[..., 0], 0, 1)
            u1 = torch.clip(u[..., 1], -np.pi, np.pi)
        else:
            u0 = torch.tanh(u[..., 0]) * 0.5 + 0.5
            u1 = torch.tanh(u[..., 1]) * np.pi
        uu = torch.stack([u0, u1], dim=-1)
        return uu

def dynamics(x0, u, include_first=False):
    # input:  x0, y0, x1, y1, x2, y2, T, hold_t
    # input:  u, (n, T)
    # return: s, (n, T, 9)
    
    t = u.shape[1]
    x = x0.clone()
    if include_first:
        segs=[x0]
    else:
        segs = []
    for ti in range(t):
        new_x = dynamics_per_step(x, u[:, ti])
        segs.append(new_x)
        x = new_x

    return torch.stack(segs, dim=1)

def dynamics_per_step(x, u):
    new_x = torch.zeros_like(x)
    close_enough_dist = args.close_thres
    if args.hard_soft_step:
        if args.norm_ap:
            near_charger = soft_step_hard(args.tanh_ratio*(close_enough_dist - torch.norm(x[:, 0:2] - x[:, 4:6], dim=-1)))
        else:
            near_charger = soft_step_hard(args.tanh_ratio*(close_enough_dist**2 - (x[:, 0] - x[:, 4])**2 - (x[:, 1] - x[:, 5])**2))
    else:
        if args.norm_ap:
            near_charger = soft_step(args.tanh_ratio*(close_enough_dist - torch.norm(x[:, 0:2] - x[:, 4:6], dim=-1)))
        else:
            near_charger = soft_step(args.tanh_ratio*(close_enough_dist**2 - (x[:, 0] - x[:, 4])**2 - (x[:, 1] - x[:, 5])**2))
    v_rover = u[:, 0] * (args.rover_vmax - args.rover_vmin) + args.rover_vmin
    th_rover = u[:, 1]
    vx0 = v_rover * torch.cos(th_rover)
    vy0 = v_rover * torch.sin(th_rover)

    new_x[:, 0] = x[:, 0] + vx0 * args.dt
    new_x[:, 1] = x[:, 1] + vy0 * args.dt
    new_x[:, 2:6] = x[:, 2:6]
    new_x[:, 6] = (x[:, 6] - args.dt) * (1-near_charger) + args.battery_charge * near_charger
    new_x[:, 7] = x[:, 7] - args.dt * near_charger
    return new_x


def get_rl_xs_us(x, policy, nt, include_first=False):
    xs = []
    us = []
    if include_first:
        xs.append(x)
    dt_minus = 0
    for ti in range(nt):
        tt1=time.time()
        u, _ = policy.predict(x.cpu(), deterministic=True)
        u = torch.from_numpy(u).cuda()
        u[..., 0] = torch.clip((u[..., 0] + 1)/2, 0, 1)
        u[..., 1] = torch.clip(u[..., 1] * np.pi, -np.pi, np.pi)
        new_x = dynamics_per_step(x, u)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti>0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.stack(us, dim=1)  # (N, 2) -> (N, T, 2)
    return xs, us, dt_minus


def initialize_x_cycle(n):
    charger_x = uniform_tensor(0, 10, (n, 1))
    charger_y = uniform_tensor(0, 10, (n, 1))    

    MAX_BATTERY_N = 25
    battery_t = rand_choice_tensor([args.dt * nn for nn in range(MAX_BATTERY_N+1)], (n, 1))
    rover_theta = uniform_tensor(-np.pi, np.pi, (n, 1))
    rover_rho = uniform_tensor(0, 1, (n, 1)) * (battery_t * args.rover_vmax)
    rover_rho = torch.clamp(rover_rho, args.close_thres, 14.14)

    rover_x = charger_x + rover_rho * torch.cos(rover_theta)
    rover_y = charger_y + rover_rho * torch.sin(rover_theta)

    dest_x = uniform_tensor(0, 10, (n, 1))
    dest_y = uniform_tensor(0, 10, (n, 1))

    # place hold case
    ratio = 0.25
    rand_mask = uniform_tensor(0, 1, (n, 1))
    rand = rand_mask>1-ratio
    ego_rho = uniform_tensor(0, args.close_thres, (n, 1))
    rover_x[rand] = (charger_x + ego_rho * torch.cos(rover_theta))[rand]
    rover_y[rand] = (charger_y + ego_rho * torch.sin(rover_theta))[rand]
    battery_t[rand] = args.dt * MAX_BATTERY_N

    hold_t = 0 * dest_x + args.dt * args.hold_t
    hold_t[rand] = rand_choice_tensor([args.dt * nn for nn in range(args.hold_t+1)], (n, 1))[rand]

    return torch.cat([rover_x, rover_y, dest_x, dest_y, charger_x, charger_y, battery_t, hold_t], dim=1)

def initialize_x(n, objs, test=False):
    x_list = []
    total_n = 0
    while(total_n<n):
        x_init = initialize_x_cycle(n)
        valids = []
        for obj_i, obj in enumerate(objs):
            obs_cpu = obj.detach().cpu()
            xmin, xmax, ymin, ymax = \
                torch.min(obs_cpu[:,0]), torch.max(obs_cpu[:,0]), torch.min(obs_cpu[:,1]), torch.max(obs_cpu[:,1]), 

            for x,y in [(x_init[:,0], x_init[:,1]), (x_init[:,2], x_init[:,3]),(x_init[:,4], x_init[:,5])]:
                if obj_i ==0:  # in map
                    val = torch.logical_and(
                        (x - xmin) * (xmax - x)>=0, 
                        (y - ymin) * (ymax - y)>=0, 
                        )
                else:  # avoid obstacles
                    val = torch.logical_not(torch.logical_and(
                        (x - xmin) * (xmax - x)>=0, 
                        (y - ymin) * (ymax - y)>=0, 
                        ))
                valids.append(val)
        
        valids = torch.stack(valids, dim=-1)
        valids_indices = torch.where(torch.all(valids, dim=-1)==True)[0]
        x_val = x_init[valids_indices]
        total_n += x_val.shape[0]
        x_list.append(x_val)
    
    x_list = torch.cat(x_list, dim=0)[:n]
    return x_list


def in_poly(xy0, xy1, poly):
    n_pts = 1000
    ts = torch.linspace(0, 1, n_pts)
    xys = xy0.unsqueeze(0) + (xy1-xy0).unsqueeze(0) * ts.unsqueeze(1)
    xmin, xmax, ymin, ymax = torch.min(poly[:,0]), torch.max(poly[:,0]),torch.min(poly[:,1]), torch.max(poly[:,1])
    
    inside = torch.logical_and(
        (xys[:,0]-xmin) * (xmax -xys[:,0])>=0,
        (xys[:,1]-ymin) * (ymax -xys[:,1])>=0,
    )
    
    res = torch.any(inside)
    return res


def max_dist_init(xys, n):
    n_trials = 1000
    N = xys.shape[0]
    max_dist=-1
    for i in range(n_trials):
        choice = np.random.choice(N, n)
        _xy = xys[choice]
        dist = torch.mean(torch.norm(_xy[:, None] - _xy[None, :], dim=-1))
        if dist > max_dist:
            max_dist = dist
            max_choice = choice
    return xys[max_choice]


def find_station(x_init, segs):
    max_i = None
    for i in range(args.num_stations):
        if torch.norm(x_init[i,0:2]-x_init[i, 4:6])<=args.close_thres:
            return i
    if segs is not None:
        for ti in range(args.nt):
            for i in range(args.num_stations):
                if (segs[ti][0]-x_init[i, 4])**2 + (segs[ti][1]-x_init[i,5])**2 <= args.close_thres**2:
                    return i
    return max_i


def test_mars(net, rl_policy, stl, objs_np, objs):
    metrics_str=["acc", "reward", "score", "t", "safety", "battery", "distance", "goals"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.rover_env import RoverEnv
    the_env = RoverEnv(args)

    nt = 200
    n_astros = 100
    update_freq = args.mpc_update_freq
    debug_t1 = time.time()

    x_init = initialize_x(n_astros, objs, test=True)
    # pick the first rover, charge station, battery time, and
    # rolling for all sampled astro_xy and dest_xy
    # (rover_xy, astro_xy, dest_xy, charge_xy, battery_T)
    num_stations = args.num_stations
    x = x_init[0:1]
    if args.multi_test:
        tmp_map = max_dist_init(x_init[:, 4:6], num_stations)
        x[:, 4:6] = tmp_map[0:1]

    fs_list = []
    history = []
    seg_list = []
    astro_i = 0
    close_enough_dist = args.close_thres

    # statistics
    stl_success = 0
    safety = 0
    battery_ok = 0
    move_distance = 0
    arrived_charger = False
    prev_arrived = False
    cnt = 0
    if args.multi_test:
        map_list = []
        marker_list = []
        stl_max_i = None

    if net is not None:
        u3 = net(x.cuda())
    
    for ti in range(nt):
        # if update
        if cnt % update_freq == 0 or arrived_dest or (arrived_charger and not prev_arrived):
            cnt = 0
            x_input = x.cuda()
            x_input2 = x_input.clone()
            x_input2[0, 6] = torch.clamp(x_input[0, 6], 0, 25 * args.dt)
            x_input2[0, 7] = torch.clamp(x_input[0, 7], -0.2, args.hold_t * args.dt)
            print("xe:%.2f ye:%.2f xd:%.2f yd:%.2f xc:%.2f yc:%.2f TB:%.2f hold:%.2f"%(
                x_input2[0, 0], x_input2[0, 1], x_input2[0, 2], x_input2[0, 3], x_input2[0, 4], 
                x_input2[0, 5], x_input2[0, 6], x_input2[0, 7], 
            ))
            dt_minus = 0
            debug_tt1=time.time()
            if args.multi_test and not arrived_charger:
                # only when not in charger stations
                x_input3 = x_input2.repeat([num_stations, 1])
                x_input3[:, 4:6] = tmp_map[:, 0:2]                
                if args.rl:
                    _, u3, dt_minus = get_rl_xs_us(x_input3, rl_policy, args.nt, include_first=True)
                    seg3 = dynamics(x_input3, u3, include_first=True)
                    stl_score = stl(seg3, args.smoothing_factor, d={"hard":False})[:, :1]
                    stl_max_i = torch.argmax(stl_score, dim=0)
                    u = u3[stl_max_i:stl_max_i+1]
                    seg = seg3[stl_max_i:stl_max_i+1]
                else:
                    u3 = net(x_input3)
                    seg3 = dynamics(x_input3, u3, include_first=True)
                    stl_score = stl(seg3, args.smoothing_factor, d={"hard":False})[:, :1]
                    stl_max_i = torch.argmax(stl_score, dim=0)
                    u = u3[stl_max_i:stl_max_i+1]
                    seg = seg3[stl_max_i:stl_max_i+1]

            else:
                if args.rl:
                    _, u, dt_minus = get_rl_xs_us(x_input2, rl_policy, args.nt, include_first=True)
                else:
                    u = net(x_input2)
                    seg = dynamics(x_input2, u, include_first=True)
                seg = dynamics(x_input, u, include_first=True)
            debug_tt2=time.time()
        if args.multi_test:
            map_list.append(tmp_map)
            marker_list.append(stl_max_i)
        
        seg_list.append(seg.detach().cpu())
        history.append(x.clone())

        move_distance += torch.norm(seg[:, cnt+1, :2].detach().cpu()-x[:, :2], dim=-1)
        stl_success += (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        safety += 1-int(any([in_poly(x[0, :2], seg[0,cnt+1,:2].detach().cpu(), obs.detach().cpu()) for obs in objs[1:]]))
        battery_ok += int(x[0, 6]>=0)
        x = seg[:, cnt+1].detach().cpu()
        cnt+=1

        seg_total = seg.clone()
        # EVALUATION
        debug_dt = debug_tt2 - debug_tt1 - dt_minus
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(the_env.generate_reward_batch(to_np(seg_total[:,0])))
        
        metrics["t"].append(debug_dt)
        metrics["safety"].append(safety/(ti+1))
        metrics["acc"].append(acc_avg)
        metrics["score"].append(score_avg)
        metrics["reward"].append(reward)
        metrics["battery"].append(battery_ok/(ti+1))
        metrics["distance"].append(move_distance.item())
        metrics["goals"].append(astro_i)

        # any update and corresponding handling
        arrived_dest = torch.norm(x[0, 0:2] - x[0, 2:4], dim=-1) < close_enough_dist
        if arrived_dest:
            astro_i += 1
            x[0, 2:6] = x_init[astro_i, 2:6]

        arrived_charger = torch.norm(x[0, 0:2] - x[0, 4:6], dim=-1) < close_enough_dist
        if arrived_charger==False:
            x[0, 7] = args.hold_t * args.dt

        prev_arrived = arrived_charger

        NT = ti+1
        print("t:%03d| MPC:%d, stl-acc:%.2f safety:%.2f battery_ok:%.2f distance:%.2f goals:%d" %(
            ti, args.mpc_update_freq, stl_success/NT, safety/NT, battery_ok/NT, move_distance, astro_i
        ))
    stat_str="MPC:%d, stl-acc:%.2f safety:%.2f battery_ok:%.2f distance:%.2f goals:%d" %(
            args.mpc_update_freq, stl_success/nt, safety/nt, battery_ok/nt, move_distance, astro_i
        )
    print(stat_str)
    
    metrics_avg = {xx:np.mean(np.array(metrics[xx])) for xx in metrics}
    metrics_avg["safety"] = metrics["safety"][-1]
    metrics_avg["battery"] = metrics["battery"][-1]
    metrics_avg["distance"] = metrics["distance"][-1]
    metrics_avg["goals"] = metrics["goals"][-1]
    if args.no_eval==False:
        eval_proc(metrics, "e5_rover", args, metrics_avg)

    if args.no_viz==False:
        history = torch.stack(history, dim=1)
        seg_list = torch.stack(seg_list, dim=1)
        # visualization
        for ti in range(nt):
            if ti % args.sim_freq == 0 or ti == nt - 1:
                if ti % 5 == 0:
                    print("Viz", ti)
                ax = plt.gca()
                plot_env(ax, objs_np)
                s = history[0, ti]
                ax.add_patch(Circle([s[0], s[1]], args.close_thres/2, color="blue", label="rover"))
                ax.add_patch(Circle([s[2], s[3]], args.close_thres/2, color="green", label="destination"))
                if args.multi_test:
                    if map_list[ti] is not None:
                        for j in range(num_stations):
                            tmp_map = map_list[ti]
                            ax.add_patch(Circle([tmp_map[j,0], tmp_map[j,1]], args.close_thres/2, color="orange", label="charger" if j==0 else None))
                        tmp_max_i = to_np(marker_list[ti])
                        ax.add_patch(Circle([to_np(tmp_map[tmp_max_i,0]), to_np(tmp_map[tmp_max_i,1])], args.close_thres/2, color="chocolate"))
                    else:
                        ax.add_patch(Circle([s[4], s[5]], args.close_thres/2, color="orange", label="charger"))
                else:
                    ax.add_patch(Circle([s[4], s[5]], args.close_thres/2, color="orange", label="charger"))
                ax.plot(seg_list[0, ti,:,0], seg_list[0, ti,:,1], color="blue", linewidth=2, alpha=0.5, zorder=10)
                ax.text(s[0]+0.25, s[1]+0.25, "%.1f"%(s[6]), fontsize=12)
                plt.xlim(0, 10)
                plt.ylim(0, 10)
                plt.legend(fontsize=14, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3))
                ax.axis("scaled")
                figname="%s/t_%03d.png"%(args.viz_dir, ti)
                plt.title("Simulation (%04d/%04d) Runtime:%5.1fFPS\n STL:%.2f%% Safe:%.2f%% battery:%.2f%% goal:%2d"%(
                    ti, nt, 1/np.mean(np.array(metrics["t"][ti])),
                    100*np.mean(np.array(metrics["acc"][:ti+1])), 100*metrics["safety"][ti], 
                    100*metrics["battery"][ti], metrics["goals"][ti]), fontsize=14)
                plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                fs_list.append(figname)
        
        os.makedirs("%s/animation"%(args.viz_dir), exist_ok=True)
        generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.08, fs_list)
    debug_t2 = time.time()
    print("Finished in %.2f seconds"%(debug_t2 - debug_t1))


def plot_env(ax, objs_np):
    for ii, obj in enumerate(objs_np[1:]):
        rect = Polygon(obj, color="gray", alpha=0.25, label="obstacle" if ii==0 else None)
        ax.add_patch(rect)


def generate_objs():
    objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
    objs_np.append(np.array([[0.0, 0.0], [args.obs_w, 0], [args.obs_w, args.obs_w], [0, args.obs_w]]))  # first obstacle
    objs_np.append(objs_np[1] + np.array([[5-args.obs_w/2, 10-args.obs_w]]))  # second obstacle (top-center)
    objs_np.append(objs_np[1] + np.array([[10-args.obs_w, 0]]))  # third obstacle (bottom-right)
    objs_np.append(objs_np[1] / 2 + np.array([[5-args.obs_w/4, 5-args.obs_w/4]]))  # forth obstacle (center-center, shrinking)

    objs = [to_torch(ele) for ele in objs_np]
    objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
    objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

    return objs_np, objs, objs_t1, objs_t2


def main():
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    objs_np, objs, objs_t1, objs_t2 = generate_objs()

    # STL definition
    in_map = Always(0, args.nt, 
        AP(lambda x: pts_in_poly(x[..., :2], objs[0], args, obses_1=objs_t1[0], obses_2=objs_t2[0]))
    )

    avoid_func = lambda y, y1, y2: Always(0, args.nt, And(
        AP(lambda x: -pts_in_poly(x[..., :2], y, args, obses_1=y1, obses_2=y2)), 
        AP(lambda x: args.seg_gain * -seg_int_poly(x[..., :2], y, args, obses_1=y1, obses_2=y2))
    ))

    avoids = []
    for obs, obs1, obs2 in zip(objs[1:], objs_t1[1:], objs_t2[1:]):
        avoids.append(avoid_func(obs, obs1, obs2))
    if args.norm_ap:
        at_dest = AP(lambda x: args.close_thres - torch.norm(x[...,0:2]-x[...,2:4], dim=-1))
        at_charger = AP(lambda x: args.close_thres - torch.norm(x[...,0:2]-x[...,4:6], dim=-1))
    else:
        at_dest = AP(lambda x: -(x[...,0]-x[...,2])**2-(x[...,1]-x[...,3])**2+args.close_thres**2)
        at_charger = AP(lambda x: -(x[...,0]-x[...,4])**2-(x[...,1]-x[...,5])**2+args.close_thres**2)
    
    battery_limit = args.dt*args.nt
    reaches = [Imply(AP(lambda x: x[..., 6] - battery_limit), Eventually(0, args.nt+1, at_dest))]
    battery = Always(0, args.nt, AP(lambda x:x[..., 6]))
    emergency = Imply(AP(lambda x: battery_limit - x[..., 6]), Eventually(0, args.nt, at_charger))
    if args.norm_ap:
        stand_by = AP(lambda x: 0.1 - torch.norm(x[..., 0:2] - x[..., 0:1, 0:2], dim=-1), comment="Stand by")
    else:
        stand_by = AP(lambda x: 0.1 **2 - (x[..., 0] - x[..., 0:1, 0])**2 -  (x[..., 1] - x[..., 0:1, 1])**2, comment="Stand by")
    enough_stay = AP(lambda x: -x[..., 7], comment="Stay>%d"%(args.hold_t))
    hold_cond = [Imply(at_charger, Always(0, args.hold_t, Or(stand_by, enough_stay)))]
    stl = ListAnd([in_map] + avoids + reaches + hold_cond + [battery, emergency])

    print(stl)
    stl.update_format("word")
    print(stl)

    from stable_baselines3 import SAC, PPO, A2C
    rl_policy = None
    if args.rl:
        rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
    test_mars(net, rl_policy, stl, objs_np, objs)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=250000)
    add("--lr", type=float, default=3e-5)
    add("--nt", type=int, default=10)
    add("--dt", type=float, default=0.2)
    add("--print_freq", type=int, default=500)
    add("--viz_freq", type=int, default=5000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)   

    add("--sim_freq", type=int, default=1)
    add("--rover_vmax", type=float, default=10.0)
    add("--astro_vmax", type=float, default=0.0)
    add("--rover_vmin", type=float, default=0.0)
    add("--astro_vmin", type=float, default=0.0)
    add("--close_thres", type=float, default=0.8)
    add("--battery_decay", type=float, default=1.0)
    add("--battery_charge", type=float, default=5.0)
    add("--obs_w", type=float, default=3.0)
    add("--ego_turn", action="store_true", default=False)
    add("--hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--no_obs", action="store_true", default=False)
    add("--one_obs", action="store_true", default=False)
    add("--limited", action="store_true", default=False)
    add("--if_cond", action='store_true', default=False)
    add("--nominal", action='store_true', default=False)
    add("--dist_w", type=float, default=0.01)
    add("--no_acc_mask", action='store_true', default=False)
    add("--seq_reach", action='store_true', default=False)
    add("--together_ratio", type=float, default=0.2)
    add("--until_emergency", action='store_true', default=False)
    add("--list_and", action='store_true', default=False)
    add("--mpc_update_freq", type=int, default=1)
    add("--seg_gain", type=float, default=1.0)
    add("--hold_t", type=int, default=3)
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=0.05)
    add("--update_init_freq", type=int, default=500)

    add("--multi_test", action='store_true', default=False)
    add("--mpc", action='store_true', default=False)
    add("--num_stations", type=int, default=5)

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
    add("--no_eval", action="store_true", default=False)

    add("--finetune", action="store_true", default=False)

    args = parser.parse_args()

    args.no_acc_mask = True
    args.no_tanh = True
    args.norm_ap = True
    args.hard_soft_step = True
    args.multi_test = True

    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))