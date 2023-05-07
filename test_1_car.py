from stl_lib import *
from matplotlib.patches import Rectangle, Ellipse
import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, build_relu_nn, soft_step_hard, get_exp_dir, eval_proc

plt.rcParams.update({'font.size': 20})


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.net = build_relu_nn(7, args.nt, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):
        u = self.net(x)
        if args.no_tanh:
            u = torch.clip(u, -10.0, 10.0)
        else:
            u = torch.tanh(u) * 10.0
        return u

def soft_step(x):
    if args.hard_soft_step:
        return soft_step_hard(args.tanh_ratio * x)
    else:
        return (torch.tanh(500 * x) + 1)/2

def dynamics(x0, u, include_first=False):
    # input:  x, (n, 6)  # xe, ve, i, t, x_dist, vo, trigger
    # input:  u, (n, T, 1)
    # return: s, (n, T, 6)
    t = u.shape[1]
    x = x0.clone()
    segs = []
    if include_first:
        segs.append(x)
    for ti in range(t):
        new_x = dynamics_per_step(x, u[:, ti:ti+1])
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)


def dynamics_per_step(x, u):
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

def get_rl_xs_us(x, policy, nt, include_first=False):
    xs = []
    us = []
    dt_minus=0
    if include_first:
        xs.append(x)
    for ti in range(nt):
        tt1=time.time()
        u, _ = policy.predict(x.cpu(), deterministic=True)
        u = torch.clip(torch.from_numpy(u * args.amax), -args.amax, args.amax).cuda()
        new_x = dynamics_per_step(x, u)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti>0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.cat(us, dim=1)  # because u [N,1] => [N,T]
    return xs, us, dt_minus

class Lane():
    def __init__(self, id, from_xy, to_xy, lane_width, lane_length, from_id, to_id, viz_xy, viz_w, viz_h, lane_type):
        self.lane_id = id
        self.from_xy = from_xy
        self.to_xy = to_xy
        self.lane_width = lane_width
        self.lane_length = lane_length
        self.from_id = from_id
        self.to_id = to_id
        self.veh_ids = []
        self.viz_xy = viz_xy
        self.viz_w = viz_w
        self.viz_h = viz_h
        self.lane_type = lane_type
    
    def dist(self):
        # > 0, remaining
        # < 0, passed
        return self.lane_length
    
    def get_dx_dy(self, ds):
        if self.lane_type == 0:
            return 0.0, ds
        elif self.lane_type == 1:
            return ds, 0.0
        elif self.lane_type == 2:
            return 0.0, -ds
        else:
            return -ds, 0.0


class Intersection():
    def __init__(self, id, xy, width, is_light, timer, viz_xy, viz_w, viz_h):
        self.inter_id = id
        self.xy = xy
        self.width = width
        self.length = width
        self.is_light = is_light
        self.timer = timer
        self.veh_ids = []
        self.viz_xy = viz_xy
        self.viz_w = viz_w
        self.viz_h = viz_h

    def update(self):
        if self.is_light:
            self.timer = (self.timer + args.dt) % args.phase_t

    def dist(self, direction):
        # dir = -1, left; 0, straight; +1, right
        if direction==0:
            return self.width
        elif direction == -1:
            return 0.75 * self.width * np.pi / 2
        elif direction == 1:
            return 0.25 * self.width * np.pi / 2
    
    def get_dx_dy(self, remain_s, ds, lane_type, direction):
        if direction == 0:
            dx = 0
            dy = ds
        elif direction == -1:
            r = 0.75 * self.width
            arc = r * np.pi / 2
            th0 = (1 - remain_s / arc) * np.pi / 2
            th1 = (1 - (remain_s - ds) / arc) * np.pi / 2
            dx = r * (np.cos(th1) - np.cos(th0))
            dy = r * (np.sin(th1) - np.sin(th0))
        elif direction == 1:
            r = 0.25 * self.width
            arc = r * np.pi / 2
            th0 = (1 - remain_s / arc) * np.pi / 2
            th1 = (1 - (remain_s - ds) / arc) * np.pi / 2
            dx = - r * (np.cos(th1) - np.cos(th0))
            dy = r * (np.sin(th1) - np.sin(th0))

        if lane_type == 1:
            dx, dy = dy, -dx
        
        if lane_type == 2:
            dx, dy = -dx, -dy

        if lane_type == 3:
            dx, dy = -dy, dx

        return dx, dy

class Vehicle():
    def __init__(self, id, v, in_lane, in_id, dist, xy, timer, base_i, is_light, hist, tri, out_a, is_hybrid):
        self.id = id
        self.v = v
        self.in_lane = in_lane
        self.in_id = in_id
        self.dist = dist
        self.xy = xy
        self.timer = timer
        self.base_i = base_i
        self.is_light = is_light
        self.hist = hist
        self.tri = tri
        self.out_a = out_a
        self.is_hybrid = is_hybrid


def get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j):
    if prev_i is None:
        return False
    di0 = curr_i - prev_i
    dj0 = curr_j - prev_j
    di1 = next_i - curr_i
    dj1 = next_j - curr_j
    
    lefts = [(1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0)]
    rights = [(1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0)]
    if (di0, dj0, di1, dj1) in lefts:
        return -1
    if (di0, dj0, di1, dj1) in rights:
        return 1
    return 0


def compute_lane_dist(ego_xy, other_xy, lane_type):
    # print(ego_xy, other_xy, lane_type)
    if lane_type == 0:
        return other_xy[1] - ego_xy[1]
    if lane_type == 1:
        return other_xy[0] - ego_xy[0]
    if lane_type == 2:
        return -other_xy[1] + ego_xy[1]
    if lane_type == 3:
        return -other_xy[0] + ego_xy[0]

def sim_multi(net, rl_policy, stl):
    metrics_str=["acc", "reward", "score", "t", "safety", "avg_x", "avg_v"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.car_env import CarEnv
    car_env = CarEnv(args)

    nt = 150 # 350
    N = 1
    n_vehs = 15 # 10
    nx = 4
    ny = 5
    IW = 6
    n_roads = 16
    dx_list = np.random.choice([10., 12, 15, 18], nx) + IW / 2
    dy_list = np.random.choice([12., 13, 15, 20], ny) + IW / 2
    map_xy = np.zeros((ny, nx, 2))

    # randomly assign traffic light (phases) and stop signs
    timers = np.random.rand(ny, nx) * args.phase_t
    is_light = np.random.choice(2, (ny, nx)) 
    is_light[0, 0] = 0
    is_light[0, nx-1] = 0
    is_light[ny-1, 0] = 0
    is_light[ny-1, nx-1] = 0
    
    # generate intersections
    inters = dict()
    inters_2d = [[None for i in range(nx)] for j in range(ny)]
    inter_id = 0
    for i in range(ny):
        for j in range(nx):
            map_xy[i, j, 0] = np.sum(dx_list[:j]) if j>0 else 0
            map_xy[i, j, 1] = np.sum(dy_list) - (np.sum(dy_list[:i]) if i>0 else 0)
            viz_xy = map_xy[i, j] + np.array([-IW/2, -IW/2])
            viz_w = IW
            viz_h = IW
            inters[inter_id] = Intersection(inter_id, map_xy[i, j], IW, is_light[i, j], timers[i, j], viz_xy, viz_w, viz_h)
            inters_2d[i][j] = inters[inter_id]
            inter_id += 1

    # build the roads
    lanes = dict()
    lanes_2d = dict()
    lane_id = 0
    viz_from_dxy = np.array([[IW/4, IW/2], [IW/2, -IW/4], [-IW/4, -IW/2], [-IW/2, IW/4]])
    viz_to_dxy = np.array([[IW/4, -IW/2], [-IW/2, -IW/4], [-IW/4, IW/2], [IW/2, IW/4]])
    for i in range(ny):
        for j in range(nx):
            for dii, (di, dj) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                if 0<=i+di<ny and 0<=j+dj<nx:
                    from_id = inters_2d[i][j].inter_id
                    to_id = inters_2d[i+di][j+dj].inter_id
                    from_xy = map_xy[i, j] + viz_from_dxy[dii]
                    to_xy = map_xy[i+di, j+dj] + viz_to_dxy[dii]
                    lane_width = IW / 2
                    lane_length = np.linalg.norm(from_xy-to_xy)
                    if dii == 0:
                        viz_xy = map_xy[i, j] + np.array([0, IW/2]) 
                    elif dii == 1:
                        viz_xy = map_xy[i, j] + np.array([IW/2, -IW/2])
                    elif dii == 2:
                        viz_xy = map_xy[i+di, j+dj] + np.array([-IW/2, IW/2])
                    elif dii == 3:
                        viz_xy = map_xy[i+di, j+dj] + np.array([IW/2, 0])
                    viz_w = lane_width if dii in [0, 2] else lane_length
                    viz_h = lane_length if dii in [0, 2] else lane_width
                    lanes[lane_id] = Lane(lane_id, from_xy, to_xy, lane_width, lane_length, from_id, to_id, viz_xy, viz_w, viz_h, dii)
                    lanes_2d[from_id, to_id] = lanes[lane_id]
                    lane_id += 1
    
    poss_starts = []
    poss_starts_idx = list(range(ny*nx))
    for i in range(ny):
        for j in range(nx):
            poss_starts.append([i, j])
    
    # random routes:
    moves = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    routes = {}
    is_used = []
    vehicles = {}
    for veh_id in range(n_vehs):
        _road_i = np.random.choice(poss_starts_idx)
        _roads = [poss_starts[_road_i]]
        poss_starts_idx.remove(_road_i)
        is_used.append(_roads[0])
        prev_i, prev_j = None, None
        for k in range(n_roads):
            curr_i, curr_j = _roads[-1]
            done = False
            while not done:
                done = True 
                rand_dir = np.random.choice(4)
                next_i, next_j = curr_i + moves[rand_dir][0], curr_j + moves[rand_dir][1]
                if next_i<0 or next_i>=ny or next_j<0 or next_j>=nx:
                    done = False
                    continue
                if k>0 and next_i == _roads[-2][0] and next_j == _roads[-2][1]:
                    done = False 
                    continue
                if is_light[curr_i, curr_j] and get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j)==-1:
                    done = False
                    continue
            prev_i, prev_j = curr_i, curr_j
            _roads.append((next_i, next_j))

        routes[veh_id] = _roads

        # TODO vectorized the cars info
        from_id = inters_2d[_roads[0][0]][_roads[0][1]].inter_id
        to_id = inters_2d[_roads[1][0]][_roads[1][1]].inter_id
        to_inter = inters_2d[_roads[1][0]][_roads[1][1]]
        lane_id = lanes_2d[from_id, to_id].lane_id
        lane = lanes[lane_id]
        lane.veh_ids.append(veh_id)
        
        if to_inter.is_light:
            if lane.lane_type in [0, 2]:
                car_timer = to_inter.timer
            else:
                car_timer = (to_inter.timer + args.phase_red) % args.phase_t
                # red-3, green-5
                # 1 2 3 4 5 6 7 0
                # R R R G G G G R
                # G G G R R R R G
                # 5 6 7 0 1 2 3 4
        else:
            car_timer = 0
        vehicles[veh_id] = Vehicle(
            id=veh_id, v=5.0, in_lane=True, in_id=lane_id, dist=lane.dist(), xy=np.array(lane.from_xy),
            timer=car_timer, base_i=0, is_light=to_inter.is_light, hist=[], tri=0, out_a=0, is_hybrid=veh_id>=n_vehs,
        )
    
    CAR_L = 2
    CAR_W = 1.5
    fs_list = []
    for ti in range(nt):
        # compute cars_tri LOGIC
        cars_out_a = {}
        for int_i in range(ny):
            for int_j in range(nx):
                veh_queue = []
                inter = inters_2d[int_i][int_j]
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if 0<=di+int_i<ny and 0<=dj+int_j<nx:
                        from_inter = inters_2d[di+int_i][dj+int_j]
                        link = lanes_2d[from_inter.inter_id, inter.inter_id]
                        for veh_id in link.veh_ids:
                            veh = vehicles[veh_id]
                            lead = 1 if veh_id == link.veh_ids[0] else 0
                            weight = lead * 1000 - veh.dist + (veh.v)**2/2/args.amax
                            veh_queue.append((veh_id, weight))
                
                if len(veh_queue)>0:
                    veh_queue = sorted(veh_queue, key=lambda x:x[1], reverse=True)
                    veh = vehicles[veh_queue[0][0]]
                    veh.tri = 0.0
                    for veh_pair in veh_queue[1:]:
                        veh_id, weight = veh_pair
                        vehicles[veh_id].tri = 1.0
        # get the info for cars on lanes
        x_input = []
        vid=[]
        for lane_id in lanes:
            lane = lanes[lane_id]
            for veh_id in lane.veh_ids:
                veh = vehicles[veh_id]
                x_ego = -veh.dist
                v_ego = veh.v
                i_light = veh.is_light
                timer = veh.timer
                if veh_id != lane.veh_ids[0]:
                    other_id = lane.veh_ids[0]
                    other_veh = vehicles[other_id]
                    x_o = compute_lane_dist(veh.xy, other_veh.xy, lane.lane_type)
                    v_o = other_veh.v
                    x_o = np.clip(x_o-args.bloat_dist, 0, 10)
                else:
                    x_o = -1
                    v_o = 0
                i_tri = veh.tri

                # clip
                x_ego = np.clip(x_ego, -10, 0)

                print("LANE  t=%d|id=%d x:%.2f v:%.2f L:%.1f T:%.1f xo:%.2f vo:%.2f Trigger:%d"%(ti, veh_id, x_ego, v_ego, i_light, timer, x_o, v_o, i_tri))
                x_input.append([x_ego, v_ego, i_light, timer, x_o, v_o, i_tri])
                vid.append(veh_id)

        x_input = torch.tensor(x_input).float().cuda()
        dt_minus1 = 0
        if x_input.shape[0]>0:
            debug_t1=time.time()       
            
            if args.rl:
                tmp_xs, u_output, dt_minus1 = get_rl_xs_us(x_input, rl_policy, args.nt, include_first=True)
            else:
                u_output = net(x_input)
            debug_t2=time.time()
            
            nn_idx=0
            for lane_id in lanes:
                lane = lanes[lane_id]
                for veh_id in lane.veh_ids:
                    veh = vehicles[veh_id]
                    nn_idx += 1
            seg = dynamics(x_input, u_output)
            
            nn_idx = 0
            for lane_id in lanes:
                lane = lanes[lane_id]
                for veh_id in lane.veh_ids:
                    veh = vehicles[veh_id]
                    veh.out_a = u_output[nn_idx, 0].item()
                    veh.timer = seg[nn_idx, 0, 3].item()
                    veh.hist = seg[nn_idx].detach().cpu().numpy()
                    veh.v = seg[nn_idx, 0, 1].item()
                    nn_idx += 1
        else:
            debug_t2 = debug_t1 = 0

        # get the info for cars in the intersections
        x_input2 = []
        vid=[]
        for inter_id in inters:
            inter = inters[inter_id]
            for veh_id in inter.veh_ids:
                veh = vehicles[veh_id]
                curr_i, curr_j = routes[veh_id][veh.base_i]
                next_i, next_j = routes[veh_id][veh.base_i+1]
                next_inter = inters_2d[next_i][next_j]
                to_id = next_inter.inter_id
                next_lane = lanes_2d[inter_id, to_id]
                x_ego = -(veh.dist + next_lane.lane_length)
                v_ego = veh.v
                i_light = next_inter.is_light
                timer = veh.timer
                x_o = -1
                v_o = 0
                i_tri = 0
                x_ego = np.clip(x_ego, -10, 0)
                print("INTER t=%d|id=%d x:%.2f v:%.2f L:%.1f T:%.1f xo:%.2f vo:%.2f Trigger:%d"%(ti, veh_id, x_ego, v_ego, i_light, timer, x_o, v_o, i_tri))
                x_input2.append([x_ego, v_ego, i_light, timer, x_o, v_o, i_tri])
                vid.append(veh_id)
        
        x_input2 = torch.tensor(x_input2).float().cuda()
        dt_minus2 = 0
        if x_input2.shape[0]>0:
            debug_t3=time.time()
            if args.rl:
                tmp_xs, u_output2, dt_minus2 = get_rl_xs_us(x_input2, rl_policy, args.nt, include_first=True)
            else:
                u_output2 = net(x_input2)
            
            debug_t4=time.time()

            nn_idx = 0
            for inter_id in inters:
                inter = inters[inter_id]
                for veh_id in inter.veh_ids:
                    veh = vehicles[veh_id]
                    nn_idx += 1
            seg2 = dynamics(x_input2, u_output2)
            nn_idx = 0
            for inter_id in inters:
                inter = inters[inter_id]
                for veh_id in inter.veh_ids:
                    veh = vehicles[veh_id]
                    veh.out_a = u_output2[nn_idx, 0].item()
                    veh.timer = seg2[nn_idx, 0, 3].item()
                    veh.hist = seg2[nn_idx].detach().cpu().numpy()
                    veh.v = seg2[nn_idx, 0, 1].item()
                    nn_idx += 1
        
        else:
            debug_t4 = debug_t3 = 0
            seg2 = None

        ### UDPATE INFO
        is_handled = [False for _ in routes]
        for lane_id in lanes:
            lane = lanes[lane_id]
            for veh_id in lane.veh_ids:
                veh = vehicles[veh_id]
                is_handled[veh_id] = True
                ds = veh.v * args.dt
                dx, dy = lane.get_dx_dy(ds)

                veh.xy[0] += dx
                veh.xy[1] += dy 
                veh.dist -= ds

                # NEW REGISTRATION
                if veh.dist < 0:
                    veh.in_id = lane.to_id
                    inter = inters[lane.to_id]
                    new_ds = -veh.dist
                    lane.veh_ids.remove(veh_id)
                    inter.veh_ids.append(veh_id)

                    veh.base_i += 1
                    prev_i, prev_j = routes[veh_id][veh.base_i - 1]
                    curr_i, curr_j = routes[veh_id][veh.base_i]
                    next_i, next_j = routes[veh_id][veh.base_i + 1]
                    lane_type = lane.lane_type
                    direction = get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j)
                    
                    # clean the distance
                    veh.xy = np.array(lane.to_xy)
                    veh.dist = inter.dist(direction)
                    dx, dy = inter.get_dx_dy(veh.dist, new_ds, lane_type, direction)
                    veh.xy[0] += dx
                    veh.xy[1] += dy 
                    veh.dist -= new_ds

                    next_inter = inters_2d[next_i][next_j]
                    veh.is_light = next_inter.is_light
                    if next_inter.is_light:
                        next_lane = lanes_2d[inter.inter_id, next_inter.inter_id]
                        if next_lane.lane_type in [0, 2]:
                            veh.timer = next_inter.timer
                        else:
                            veh.timer = (next_inter.timer + args.phase_red) % args.phase_t
                    else:
                        veh.timer = 0

        for inter_id in inters:
            inter = inters[inter_id]
            for veh_id in inter.veh_ids:
                veh = vehicles[veh_id]
                if is_handled[veh_id]:
                    continue
                is_handled[veh_id] = True
                ds = veh.v * args.dt
                prev_i, prev_j = routes[veh_id][veh.base_i - 1] if veh.base_i!=0 else (None, None)
                curr_i, curr_j = routes[veh_id][veh.base_i]
                next_i, next_j = routes[veh_id][veh.base_i + 1]
                from_id = inters_2d[prev_i][prev_j].inter_id
                to_id = inters_2d[curr_i][curr_j].inter_id
                lane_id = lanes_2d[from_id, to_id].lane_id
                lane_type = lanes[lane_id].lane_type
                direction = get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j)
                dx, dy = inter.get_dx_dy(veh.dist, ds, lane_type, direction)
                veh.xy[0] += dx
                veh.xy[1] += dy 
                veh.dist -= ds

                # NEW REGISTRATION
                if veh.dist < 0:
                    next_i, next_j = routes[veh_id][veh.base_i + 1]
                    from_id = veh.in_id
                    to_id = inters_2d[next_i][next_j].inter_id
                    lane = lanes_2d[from_id, to_id]
                    veh.in_id = lane.lane_id
                    inter.veh_ids.remove(veh_id)
                    lane.veh_ids.append(veh_id)

                    # clean the distance
                    ds = -veh.dist
                    veh.xy = np.array(lane.from_xy)
                    veh.dist = lane.dist()
                    dx, dy = lane.get_dx_dy(ds)
                    veh.xy[0] += dx
                    veh.xy[1] += dy 
                    veh.dist -= ds
        
        # EVALUATION
        debug_dt = debug_t4-debug_t3 - dt_minus2 + debug_t2-debug_t1 - dt_minus1
        if seg is None:
            seg_total = seg2
        elif seg2 is None:
            seg_total = seg
        else:
            seg_total = torch.cat([seg, seg2], dim=0)
        
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(car_env.generate_reward_batch(to_np(seg_total[:,0])))
        safety = np.mean(np.logical_or(to_np(seg_total[:, 0, 4]) > 0, to_np(seg_total[:, 0, 4]) ==-1))
        avg_x = np.mean(to_np(seg_total[:,-1, 0]) - to_np(seg_total[:,0, 0]))
        avg_v = np.mean(to_np(seg_total[:,0, 1]))

        metrics["t"].append(debug_dt)
        metrics["safety"].append(safety)
        metrics["avg_x"].append(avg_x)
        metrics["avg_v"].append(avg_v)
        metrics["acc"].append(acc_avg)
        metrics["score"].append(score_avg)
        metrics["reward"].append(reward)

        # car-map visualization
        if args.no_viz==False:
            plt.figure(figsize=(12, 12))
            ax = plt.gca()
            light_r = 0.8
            stop_r = 0.8
            has_seen_red=False
            has_seen_green=False
            has_seen_stop=False
            for lane_id in lanes:
                alpha=1
                lane = lanes[lane_id]
                rect = Rectangle(lane.viz_xy, lane.viz_w, lane.viz_h, color="gray", alpha=0.5, zorder=10)
                ax.add_patch(rect)
                to_inter = inters[lane.to_id]
                _w = stop_r*5 if lane.lane_type in [0,2] else 0.5*stop_r
                _h = stop_r*5 if lane.lane_type in [1,3] else 0.5*stop_r
                if to_inter.is_light:
                    if lane.lane_type in [0, 2]:
                        if to_inter.timer % args.phase_t < args.phase_red:
                            light_color = "red"
                        else:
                            light_color = "green"
                    else:
                        if to_inter.timer % args.phase_t < args.phase_red:
                            light_color = "green"
                        else:
                            light_color = "red"
                    label=None
                    if not has_seen_green:
                        label="Traffic light"
                        has_seen_green=True
                    obj = Ellipse(alpha*lane.to_xy + (1-alpha) * to_inter.xy, _w, _h, color=light_color, zorder=25, label=label)
                else:
                    label = None
                    if not has_seen_stop:
                        has_seen_stop=True
                        label="stop sign"
                    obj = Ellipse(lane.to_xy, _w, _h, color="black", zorder=25, label=label)
                ax.add_patch(obj)

            for inter_id in inters:
                inter = inters[inter_id]
                rect = Rectangle(inter.viz_xy, inter.viz_w, inter.viz_h, color="lightgray", alpha=0.5, zorder=10)
                ax.add_patch(rect)

            for veh_id in vehicles:
                veh = vehicles[veh_id]
                color = "brown" if veh_id>=n_vehs else "royalblue"
                label = None
                if veh_id == 0:
                    label = "Cars"
                rect = Rectangle(veh.xy-np.array([CAR_W/2,CAR_W/2]), CAR_W, CAR_W, color=color, alpha=0.9, zorder=50, label=label)
                ax.add_patch(rect)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=16)
            plt.xlabel("x (m)", fontsize=16)
            plt.ylabel("y (m)", fontsize=16)
            
            plt.axis("scaled")
            plt.xlim(-4, 61)
            plt.ylim(11, 97)
            plt.legend(loc="upper center", fontsize=16, ncol=3, bbox_to_anchor=(0.5, 1.1))
            plt.title("Simulation (%04d/%04d)" % (ti, nt), fontsize=16)
            filename = "%s/t_%03d.png"%(args.viz_dir, ti)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            fs_list.append(filename)
            plt.close()
            
        print(ti)

        # update the intersection timers
        for inter_i in inters:
            inters[inter_i].update()
        
        # save the trajectories
    print("Acc:%.3f"%(np.mean(np.array(metrics["acc"]))))
    eval_proc(metrics, "e1_car", args)

    if args.no_viz==False:
        os.makedirs('%s/animation'%(args.viz_dir), exist_ok=True)
        generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.1, fs_list)
    return


def heading_base_sampling(set0_ve):
    relu = nn.ReLU()
    n = set0_ve.shape[0]
    set00_xo = uniform_tensor(-1, -1, (n//2, 1))
    set00_vo = uniform_tensor(0, 0, (n//2, 1))
    set01_xo = uniform_tensor(args.safe_thres, args.xo_max, (n//2, 1))
    lower = torch.sqrt(relu((args.safe_thres - set01_xo)*args.amax*2 + set0_ve[n//2:]**2))
    set01_vo = uniform_tensor(0, 1, (n//2, 1)) * (args.vmax-lower) + lower

    return torch.cat([set00_xo, set01_xo], dim=0), torch.cat([set00_vo, set01_vo], dim=0)


def initialize_x(N):
    # generate initial points
    # set-0
    # x ~ [-10, 0]
    # v ~ [] make sure v^2/2a < |x|
    # t ~ 0
    #####################################################################
    n = N // 4
    set0_xe = uniform_tensor(-10, args.stop_x, (n, 1))
    bound = torch.clip(torch.sqrt(2*args.amax*(-set0_xe+args.stop_x)), 0, args.vmax)
    set0_ve = uniform_tensor(0, 1, (n, 1)) * bound  # v<\sqrt{2a|x|}
    set0_id = uniform_tensor(0, 0, (n, 1))
    set0_t = uniform_tensor(0, 0, (n, 1))
    set0_xo, set0_vo = heading_base_sampling(set0_ve)
    set0_tri = rand_choice_tensor([0, 1], (n, 1))

    set1_xe = uniform_tensor(args.stop_x, 0, (n, 1))
    set1_ve = uniform_tensor(0, 1.0, (n, 1))
    set1_id = uniform_tensor(0, 0, (n, 1))
    set1_t = uniform_tensor(0, args.stop_t+0.1, (n, 1))
    set1_xo = uniform_tensor(-1, -1, (n, 1))
    set1_vo = uniform_tensor(0, 0, (n, 1))
    set1_tri = uniform_tensor(0, 0, (n, 1))

    n2 = 2*n
    set2_xe = uniform_tensor(-10, args.traffic_x, (n2, 1))
    bound = torch.clip(torch.sqrt(2*args.amax*(-set2_xe + args.traffic_x)), 0, args.vmax)
    set2_ve = uniform_tensor(0, 1, (n2, 1)) * bound  # v<\sqrt{2a|x|}
    set2_id = uniform_tensor(1.0, 1.0, (n2, 1))
    set2_t = uniform_tensor(0, args.phase_t, (n2, 1))
    set2_xo, set2_vo = heading_base_sampling(set2_ve)
    set2_tri = uniform_tensor(0, 0, (n2, 1))

    set0 = torch.cat([set0_xe, set0_ve, set0_id, set0_t, set0_xo, set0_vo, set0_tri], dim=-1)
    set1 = torch.cat([set1_xe, set1_ve, set1_id, set1_t, set1_xo, set1_vo, set1_tri], dim=-1)
    set2 = torch.cat([set2_xe, set2_ve, set2_id, set2_t, set2_xo, set2_vo, set2_tri], dim=-1)
    
    x_init = torch.cat([set0, set1, set2], dim=0).float().cuda()

    return x_init


def main():
    utils.setup_exp_and_logger(args, test=args.test, ford=False)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        state_dict = torch.load(utils.find_path(args.net_pretrained_path))
        net.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # stl definition
    cond_wait = Eventually(0, args.nt, AP(lambda x: x[..., 3] - args.stop_t, comment="t_stop>=%.1fs"%(args.stop_t)))   
    cond_sig = Imply(AP(lambda x: x[..., 6]-0.5, "Triggered"), Always(0,args.nt, AP(lambda x:args.stop_x*0.5 - x[..., 0], "stop")))
    cond1 = And(cond_wait, cond_sig)
    cond2 = Always(0, args.nt, 
                Not(And(AP(lambda x: args.phase_red - x[...,3], comment="t=red"),
                        AP(lambda x: -x[..., 0] * (x[..., 0]-args.traffic_x), comment="inside intersection")
                )))
    cond3 = Always(0, args.nt, AP(lambda x:x[..., 4]-args.safe_thres,comment="heading>0"))

    stl = ListAnd([
        Imply(AP(lambda x: 0.5-x[..., 2], comment="I=stop"), cond1),  # stop signal condition
        Imply(AP(lambda x: x[...,2]-0.5, comment="I=light"), cond2),  # light signal condition
        Imply(AP(lambda x: x[..., 4]+0.5, comment="heading"), cond3)  # heading condition
        ])

    print(stl)
    stl.update_format("word")
    print(stl)

    # test for rl policy/nn policy
    from stable_baselines3 import SAC, PPO, A2C
    rl_policy = None
    if args.rl:
        rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
    sim_multi(net, rl_policy, stl)
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
    add("--nt", type=int, default=25)
    add("--dt", type=float, default=0.1)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=100.0)
    add("--sim", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)
    add("--amax", type=float, default=10)
    add("--stop_x", type=float, default=-1.0)
    add("--v_loss", type=float, default=0.1)
    add("--phase_t", type=float, default=8.0)
    add("--phase_red", type=float, default=4.0)
    add("--traffic_x", type=float, default=-1.0)
    add("--sim_freq", type=int, default=5)
    add("--stop_t", type=float, default=1.0)
    add("--vmax", type=float, default=10.0)
    add("--s_loss", type=float, default=0.1)
    add("--inter_x", type=float, default=0.0)

    add("--test", action='store_true', default=False)
    add("--triggered", action='store_true', default=False)
    add('--heading', action='store_true', default=False)

    add("--safe_thres", type=float, default=1.0)
    add("--xo_max", type=float, default=10.0)

    add('--mock', action='store_true', default=False)
    add('--no_tri_mock', action='store_true', default=False)
    add('--hybrid', action='store_true', default=False)
    add('--bloat_dist', type=float, default=1.0)
    add('--no_viz', action='store_true', default=False)
    add('--ford', action='store_true', default=False)

    # new-tricks
    add("--hiddens", type=int, nargs="+", default=[64, 64, 64])
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=1.0)
    add("--update_init_freq", type=int, default=-1)
    add("--add_val", action="store_true", default=False)
    add("--include_first", action="store_true", default=False)

    add("--mpc", action="store_true", default=False)
    add("--plan", action="store_true", default=False)
    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)
    add("--rl", action="store_true", default=False)
    add("--rl_stl", action="store_true", default=False)
    add("--rl_acc", action="store_true", default=False)
    add("--rl_path", "-R", type=str, default=None)
    add("--eval_path", type=str, default="eval_result")

    add("--finetune", action="store_true", default=False)
    add("--backup", action='store_true', default=False)
    add("--video", action='store_true', default=False)
    args = parser.parse_args()
    args.triggered=True
    args.heading=True

    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))

