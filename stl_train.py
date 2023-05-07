from stl_lib import *
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from matplotlib.collections import PatchCollection
import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, to_torch, build_relu_nn, build_relu_nn1
from envs.base_env import BaseEnv
from envs.car_env import CarEnv
from envs.ship_env import ShipEnv
from envs.rover_env import RoverEnv


plt.rcParams.update({'font.size': 12})

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

import csv

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0, args=None, eta=None):
        super(CustomCallback, self).__init__(verbose)
        self.args = args
        self.eta = eta
        self.csvfile = open('%s/monitor_full.csv'%(args.exp_dir_full), 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
   
    def _on_step(self):
        args = self.args
        epi = (self.n_calls-1) // args.nt
        eta = self.eta
        triggered = (self.n_calls-1) % args.nt == 0
        if triggered:
            eta.update()
            r_rs = self.model.env.env_method("get_rewards")
            r_rs = np.array(r_rs, dtype=np.float32)
            r_avg = np.mean(r_rs[:, 0])
            rs_avg = np.mean(r_rs[:, 1])
            racc_avg = np.mean(r_rs[:, 2])
            self.csvwriter.writerow([epi, r_avg, rs_avg, racc_avg, eta.elapsed()])

        if triggered and epi % args.print_freq == 0:
            x, y = ts2xy(load_results(args.exp_dir_full), "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
            else:
                mean_reward = 0.0
            print("%s RL epi:%07d reward:%.2f dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1],
                epi, mean_reward, eta.interval_str(), eta.elapsed_str(), eta.eta_str()
                ))
        if triggered:
            if epi % 100 == 0:
                self.model.save("%s/model_last"%(args.model_dir))
            if epi % ((args.epochs // args.num_workers)//5) == 0:
                self.model.save("%s/model_%05d"%(args.model_dir, epi))
        return True


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        T = args.nt
        IO_DIMS = {"car":[7, 1*T], "maze":[9, 1*T], "ship1":[12, 2*T], "ship2":[10, 2*T], "rover":[8, 2*T]}
        self.net = build_relu_nn1(IO_DIMS[args.mode], args.hiddens, activation_fn=nn.ReLU)
    
    def clip_u(self, x, amin, amax):
        if self.args.no_tanh:
            return torch.clip(x, amin, amax)
        else:
            return torch.tanh(x) * (amax - amin) / 2 + (amax  + amin) / 2

    def forward(self, x):  
        args = self.args   
        N = x.shape[0]
        T = args.nt
        u = self.net(x).reshape(N, T, -1)
        if self.args.mode == "car":
            u = u[..., 0]
            uu = self.clip_u(u, -10.0, 10.0)
        elif self.args.mode == "maze":
            u = u[..., 0]
            uu = self.clip_u(u, -40.0, 40.0)
        elif self.args.mode in ["ship1", "ship2"]: 
            u0 = self.clip_u(u[..., 0], -args.thrust_max, args.thrust_max)
            u1 = self.clip_u(u[..., 1], -args.delta_max, args.delta_max)
            uu = torch.stack([u0, u1], dim=-1)
        elif self.args.mode == "rover":
            u0 = self.clip_u(u[..., 0], 0, 1)
            u1 = self.clip_u(u[..., 1], -np.pi, np.pi)
            uu = torch.stack([u0, u1], dim=-1)
        else:
            raise NotImplementError
        return uu

def run_test(net, env):
    return


def make_env(env_name, args, seed_i, seed, logdir):
    def _f():
        env = env_name(args)
        env.seed(seed)
        env.pid=seed_i
        if seed_i==0:
            return Monitor(env, logdir)
        else:
            return env
    return _f


def soft_step(x):
    return (torch.tanh(500 * x) + 1)/2

def dynamics_per_step_car(x, u, args):
    new_x = torch.zeros_like(x)
    new_x[:, 0] = x[:, 0] + x[:, 1] * args.dt
    # mask = (torch.logical_and(x[:, 0]<args.stop_x, x[:, 2]==0)).float() # stop sign, before the stop region
    mask = (torch.logical_and(x[:, 0]<args.stop_x, torch.logical_and(x[:, 2]==0, x[:, 4]<0))).float() # stop sign, before the stop region
    if args.test:
        new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10) * (1-mask) + torch.clip(x[:, 1] + (u[:, 0]) * args.dt, 0.1, 10) * mask
    else:
        new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10)
    new_x[:, 2] = x[:, 2]
    stop_timer = (x[:, 3] + args.dt * soft_step(x[:,0]-args.stop_x)) * soft_step(-x[:,0])
    light_timer = (x[:, 3] + args.dt) % args.phase_t
    new_x[:, 3] = (1-x[:, 2]) * stop_timer + x[:, 2] * light_timer
    new_x[:, 4] = x[:, 4] + (x[:, 5] - x[:, 1]) * args.dt * (x[:, 4]>=0).float()
    new_x[:, 5] = x[:, 5]
    new_x[:, 6] = x[:, 6]
    return new_x

def dynamics_per_step_ship2(x, uu, args, num=1):
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

def dynamics_per_step_rover(x, u, args):
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


def get_rl_xs_us(x, policy, nt, args, include_first=False):
    xs = []
    us = []
    dt_minus=0
    if include_first:
        xs.append(x)
    for ti in range(nt):
        tt1 = time.time()
        # u, _ = policy.predict(x.cpu(), deterministic=True)
        u, _, _ = policy.actor.get_action_dist_params(x)
        if args.mode == "car":
            u = torch.clip(torch.from_numpy(u * args.amax), -args.amax, args.amax).cuda()
            new_x = dynamics_per_step_car(x, u, args)
        elif args.mode == "ship2":
            u[..., 0] = torch.clip(u[..., 0] * args.thrust_max, -args.thrust_max, args.thrust_max)
            u[..., 1] = torch.clip(u[..., 1] * args.delta_max, -args.delta_max, args.delta_max)
            new_x = dynamics_per_step_ship2(x, u, args, num=args.stl_sim_steps)
        elif args.mode == "rover":
            u[..., 0] = torch.clip((u[..., 0] + 1)/2, 0, 1)
            u[..., 1] = torch.clip(u[..., 1] * np.pi, -np.pi, np.pi)
            new_x = dynamics_per_step_rover(x, u, args)
        else:
            raise NotImplementError
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti>0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.stack(us, dim=1)  # because u [N,1] => [N,T]
    return xs, us, dt_minus


def main(args):
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq, args.num_workers)
    
    env_dict = {"car": CarEnv, "ship2": ShipEnv, "rover": RoverEnv}
    env_func = env_dict[args.mode]

    # RL case
    if args.train_rl or args.train_il:
        if args.num_workers != None:
            seeds = [args.seed + seed_i for seed_i in range(args.num_workers)]
            envs = [make_env(env_func, args, seed_i, seed, args.exp_dir_full) for seed_i, seed in enumerate(seeds)] 
            env = SubprocVecEnv(envs)
        else:
            env = env_func(args)
            env.seed(args.seed)
            env.pid = 0
            env = Monitor(env, args.exp_dir_full)
        callback = CustomCallback(args=args, eta=eta)

        print("Now train the policy ...")
        model = SAC("MlpPolicy", env, verbose=0, seed=args.seed, policy_kwargs={"net_arch":{"pi":args.hiddens, "qf":[256, 256]}})
        if args.train_rl:
            model.learn(total_timesteps=args.epochs*args.nt, callback=callback) #()

            print("Now evaluate ...")
            vec_env = model.get_env()
            obs = vec_env.reset()
            for i in range(1000):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                vec_env.env_method(method_name='my_render')
            return


    # model setup
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
    
    env = env_func(args)
    stl = env.generate_stl()
    env.print_stl()

    csvfile = open('%s/monitor_full.csv'%(args.exp_dir_full), 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    csvfile_val = open('%s/monitor_full_val.csv'%(args.exp_dir_full), 'w', newline='')
    csvwriter_val = csv.writer(csvfile_val, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    x_init = env.init_x(args.num_samples).float().cuda()
    x_init_val = env.init_x(5000).float().cuda()

    if args.train_il:  # train BC to let the RL-controller mimic pretrained STL-controller
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr)       
        for epi in range(args.epochs):
            eta.update()
            if args.update_init_freq >0 and epi % args.update_init_freq == 0 and epi!=0:
                x_init = env.init_x(args.num_samples).float().cuda()
            x0 = x_init.detach()
            u_ref = net(x0)
            seg_nn = env.dynamics(x0, u_ref, include_first=True)

            _n, _t, _k = seg_nn.shape
            #TODO there are many ways to do this? (off-policy, on-policy?)
            #TODO is include_first always True across different envs
            if args.il_mode=="all":
                u_list=[]
                for iii in range(args.nt):
                    u, _, _ = model.actor.get_action_dist_params(seg_nn[:,iii])
                    if args.mode == "car":
                        u = torch.clip(torch.from_numpy(u * args.amax), -args.amax, args.amax).cuda()
                    elif args.mode == "ship2":
                        u[..., 0] = torch.clip(u[..., 0] * args.thrust_max, -args.thrust_max, args.thrust_max)
                        u[..., 1] = torch.clip(u[..., 1] * args.delta_max, -args.delta_max, args.delta_max)
                    elif args.mode == "rover":
                        u[..., 0] = torch.clip((u[..., 0] + 1)/2, 0, 1)
                        u[..., 1] = torch.clip(u[..., 1] * np.pi, -np.pi, np.pi)
                    else:
                        raise NotImplementError
                    u_list.append(u)
                u_est = torch.stack(u_list, dim=1)
            elif args.il_mode=="traj":
                xs, u_est, _ = get_rl_xs_us(x0, model, args.nt, args, include_first=True)

                seg_est = xs
                score = stl(seg_est, args.smoothing_factor)[:, :1]
                score_avg = torch.mean(score)

                acc = (stl(seg_est, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_avg = torch.mean(acc)

                acc_reward = (acc_avg * 100).item()
                stl_reward = (score_avg).item()
            elif args.il_mode=="first":
                u, _, _ = model.actor.get_action_dist_params(seg_nn[:,iii])
                if args.mode == "car":
                    u = torch.clip(torch.from_numpy(u * args.amax), -args.amax, args.amax).cuda()
                elif args.mode == "ship2":
                    u[..., 0] = torch.clip(u[..., 0] * args.thrust_max, -args.thrust_max, args.thrust_max)
                    u[..., 1] = torch.clip(u[..., 1] * args.delta_max, -args.delta_max, args.delta_max)
                elif args.mode == "rover":
                    u[..., 0] = torch.clip((u[..., 0] + 1)/2, 0, 1)
                    u[..., 1] = torch.clip(u[..., 1] * np.pi, -np.pi, np.pi)
                else:
                    raise NotImplementError
                u_est = u
            if args.il_mode=="first":
                loss = torch.mean(torch.square(u_est-u_ref))
            else:
                loss = torch.mean(torch.square(u_est-u_ref[:,0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.il_mode=="all":
                csvwriter.writerow([epi, loss.item(), eta.elapsed()])
            else:
                csvwriter.writerow([epi, loss.item(), stl_reward, acc_reward, eta.elapsed()])
            csvfile.flush()

            if epi % args.print_freq == 0:
                # u_val = net(x_init_val.detach())
                xs_val, u_est_val, _ = get_rl_xs_us(x_init_val.detach(), model, args.nt, args, include_first=True)
                # seg_val = env.dynamics(x_init_val, u_val, include_first=True)
                seg_val = xs_val
                score_val = stl(seg_val, args.smoothing_factor)[:, :1]
                score_avg_val = torch.mean(score_val)

                acc_val = (stl(seg_val, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_avg_val = torch.mean(acc_val)

                all_states_val = to_np(seg_val.reshape(seg_val.shape[0] *_t, -1))
                reward_val = np.mean(env.generate_reward_batch(all_states_val)) * _t
                acc_reward_val = (acc_avg_val * 100).item()
                stl_reward_val = (score_avg_val).item()

                csvwriter_val.writerow([epi, reward_val, stl_reward_val, acc_reward_val, eta.elapsed()])
                csvfile_val.flush()
                if args.il_mode=="all":
                    print("%s|%06d  loss:%.3f acc_val:%.3f R:%.2f R':%.2f dT:%s T:%s ETA:%s" % (
                        args.exp_dir_full.split("/")[-1], epi, loss.item(), acc_avg_val.item(),
                        stl_reward_val, acc_reward_val, eta.interval_str(), eta.elapsed_str(), eta.eta_str()))
                else:
                    print("%s|%06d  loss:%.3f acc:%.3f acc_val:%.3f R:%.2f R':%.2f dT:%s T:%s ETA:%s" % (
                        args.exp_dir_full.split("/")[-1], epi, loss.item(), acc_avg.item(), acc_avg_val.item(),
                        stl_reward, acc_reward, eta.interval_str(), eta.elapsed_str(), eta.eta_str()))

            # Save models
            if epi % args.save_freq == 0:
                torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))

            if epi == args.epochs-1 or epi % 100 == 0:
                torch.save(net.state_dict(), "%s/model_last.ckpt"%(args.model_dir))


    else:  # STL-controller learning by maximizing robustness score
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)      
        for epi in range(args.epochs):
            eta.update()
            if args.update_init_freq >0 and epi % args.update_init_freq == 0 and epi!=0:
                x_init = env.init_x(args.num_samples).float().cuda()
            x0 = x_init.detach()
            u = net(x0)
            seg = env.dynamics(x0, u, include_first=True)

            score = stl(seg, args.smoothing_factor)[:, :1]
            score_avg = torch.mean(score)
            acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
            acc_avg = torch.mean(acc)
            _n, _t, _k = seg.shape
            all_states = to_np(seg.reshape(_n*_t, -1))
            reward = np.mean(env.generate_reward_batch(all_states)) * _t
            acc_reward = (acc_avg * 100).item()
            stl_reward = (score_avg).item()
            dist_loss = env.generate_heur_loss(acc, seg)

            loss = torch.mean(nn.ReLU()(args.c_val-score)) + dist_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            csvwriter.writerow([epi, reward, stl_reward, acc_reward, eta.elapsed()])
            csvfile.flush()
            
            if epi % args.print_freq == 0:
                u_val = net(x_init_val.detach())        
                seg_val = env.dynamics(x_init_val, u_val, include_first=True)
                score_val = stl(seg_val, args.smoothing_factor)[:, :1]
                score_avg_val = torch.mean(score_val)

                acc_val = (stl(seg_val, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_avg_val = torch.mean(acc_val)

                all_states_val = to_np(seg_val.reshape(seg_val.shape[0] *_t, -1))
                reward_val = np.mean(env.generate_reward_batch(all_states_val)) * _t
                acc_reward_val = (acc_avg_val * 100).item()
                stl_reward_val = (score_avg_val).item()

                csvwriter_val.writerow([epi, reward_val, stl_reward_val, acc_reward_val, eta.elapsed()])
                csvfile_val.flush()

                print("%s|%03d  loss:%.3f acc:%.3f dist:%.3f acc_val:%.3f R:%.2f R':%.2f R'':%.2f dT:%s T:%s ETA:%s" % (
                    args.exp_dir_full.split("/")[-1], epi, loss.item(), acc_avg.item(),
                    dist_loss.item(), acc_avg_val.item(), reward, stl_reward, acc_reward, eta.interval_str(), eta.elapsed_str(), eta.eta_str()))

            # Save models
            if epi % args.save_freq == 0:
                torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))

            if epi == args.epochs-1 or epi % 100 == 0:
                torch.save(net.state_dict(), "%s/model_last.ckpt"%(args.model_dir))

            if epi % args.viz_freq == 0 or epi == args.epochs - 1:
                env.visualize(x_init, seg, acc, epi)