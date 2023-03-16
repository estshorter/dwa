import numpy as np
from scipy.integrate import solve_ivp

from animation import Animation_robot

# def min_max_normalize(data):
#     data = np.array(data)
#     max_val = data.max()
#     min_val = data.min()

#     diff = max_val - min_val
#     eps = 1e-7
#     if abs(diff) < eps:
#         data = np.zeros(len(data))
#     else:
#         data = (data - min_val) / diff

#     return data


# 角度補正用
def angle_range_corrector(angle):
    if angle > np.pi:
        while angle > np.pi:
            angle -= 2 * np.pi
    elif angle < -np.pi:
        while angle < -np.pi:
            angle += 2 * np.pi

    return angle


# ルール
# x, y, thは基本的に今のstate
# g_ はgoal
# traj_ は過去の軌跡
# 単位は，角度はrad，位置はm
# 二輪モデルなので入力は速度と角速度

# 速度、角速度一定の時の経路
class Path:
    def __init__(self, x, y, th, u_th, u_v) -> None:
        self.xs = x
        self.ys = y
        self.ths = th
        self.u_v = u_v
        self.u_th = u_th


class Obstacle:
    def __init__(self, x, y, size) -> None:
        self.x = x
        self.y = y
        self.size = size


class TwoWheeledRobot:
    def __init__(self, init_x, init_y, init_th) -> None:
        self.x = init_x
        self.y = init_y
        self.th = init_th
        self.u_v = 0.0
        self.u_th = 0.0

        self.traj_x = [init_x]
        self.traj_y = [init_y]
        self.traj_th = [init_th]
        self.traj_u_v = [0.0]
        self.traj_u_th = [0.0]

    # xi: [x, y, theta]
    # u:[u_th, u_v]
    @staticmethod
    def state_equation(xi, u):
        dxi = np.empty(3)
        dxi[0] = u[1] * np.cos(xi[2])
        dxi[1] = u[1] * np.sin(xi[2])
        dxi[2] = u[0]
        return dxi

    def update_state(self, u_th, u_v, dt):
        self.u_th = u_th
        self.u_v = u_v

        # rk45等で数値積分する
        xi_init = np.array([self.x, self.y, self.th])
        u = np.array([u_th, u_v])
        sol = solve_ivp(
            lambda t, xi: TwoWheeledRobot.state_equation(xi, u), [0, dt], xi_init
        )
        integrated = sol.y[:, -1]
        next_x = integrated[0]
        next_y = integrated[1]
        next_th = integrated[2]

        self.traj_x.append(next_x)
        self.traj_y.append(next_y)
        self.traj_th.append(next_th)

        self.x = next_x
        self.y = next_y
        self.th = next_th

        return self.x, self.y, self.th


class CoarseSimulator:
    def __init__(self) -> None:
        self.max_acc = 1.0  # m/s^2
        self.max_ang_acc = np.deg2rad(100)  # rad/s^2

        self.lim_max_vel = 1.6  # m/s
        self.lim_min_vel = 0.0
        self.lim_max_ang_vel = np.pi  # deg/s
        self.lim_min_ang_vel = -self.lim_max_ang_vel

    def predict_state(self, ang_vel, vel, x, y, th, dt, pre_step):
        next_xs = []
        next_ys = []
        next_ths = []

        for _ in range(pre_step):
            x = vel * np.cos(th) * dt + x
            y = vel * np.sin(th) * dt + y
            th = ang_vel * dt + th

            next_xs.append(x)
            next_ys.append(y)
            next_ths.append(th)

        return next_xs, next_ys, next_ths


class ConstGoal:
    def __init__(self) -> None:
        self.traj_g_x = []
        self.traj_g_y = []

    def calc_goal(self, time_step):
        # g_x = g_y = 10.0
        if time_step <= 100:
            g_x = 10.0
            g_y = 10.0
        else:
            g_x = -10.0
            g_y = -10.0

        self.traj_g_x.append(g_x)
        self.traj_g_y.append(g_y)

        return g_x, g_y


class DWA:
    def __init__(self, samplingtime) -> None:
        self.simu_robot = CoarseSimulator()

        self.pre_time = 3
        self.pre_step = 30

        self.delta_vel = 0.02
        self.delta_ang_vel = 0.02

        self.samplingtime = samplingtime

        self.weight_angle = 0.04
        self.weight_vel = 0.2
        self.weight_obs = 0.1

        # 近傍とみなす距離
        area_dis_to_obs = 5
        self.area_dis_to_obs_sqrd = area_dis_to_obs**2

        # スコアの最大値
        score_obstacle = 2
        self.score_obstacle_sqrd = score_obstacle**2

        self.traj_paths = []
        self.traj_opt = []

    def calc_input(self, g_x, g_y, state, obstacles):
        paths = self._make_path(state)
        opt_path = self._eval_path(paths, g_x, g_y, state, obstacles)
        self.traj_opt.append(opt_path)
        return paths, opt_path

    def _make_path(self, state):
        min_ang_vel, max_ang_vel, min_vel, max_vel = self._calc_range_vels(state)

        paths = []

        for ang_vel in np.arange(min_ang_vel, max_ang_vel, self.delta_ang_vel):
            for vel in np.arange(min_vel, max_vel, self.delta_vel):
                next_x, next_y, next_th = self.simu_robot.predict_state(
                    ang_vel,
                    vel,
                    state.x,
                    state.y,
                    state.th,
                    self.samplingtime,
                    self.pre_step,
                )
                paths.append(Path(next_x, next_y, next_th, ang_vel, vel))

        self.traj_paths.append(paths)

        return paths

    def _calc_range_vels(self, state):
        range_ang_vel = self.samplingtime * self.simu_robot.max_ang_acc

        min_ang_vel = max(state.u_th - range_ang_vel, self.simu_robot.lim_min_ang_vel)
        max_ang_vel = min(state.u_th + range_ang_vel, self.simu_robot.lim_max_ang_vel)

        range_vel = self.samplingtime * self.simu_robot.max_acc
        min_vel = max(state.u_v - range_vel, self.simu_robot.lim_min_vel)
        max_vel = min(state.u_v + range_vel, self.simu_robot.lim_max_vel)

        return min_ang_vel, max_ang_vel, min_vel, max_vel

    def _eval_path(self, paths, g_x, g_y, state, obstacles):
        neighbor_obs = self._calc_neighbor_obs(state, obstacles)

        score_heading_angles = []
        score_heading_vels = []
        score_obstacles = []

        for path in paths:
            score_obs = self._calc_obstacles_score(path, neighbor_obs)
            if score_obs == -float("inf"):
                continue
            score_heading_angles.append(self._calc_heading_angle_score(path, g_x, g_y))
            score_heading_vels.append(self._calc_heading_vel_score(path))
            score_obstacles.append(score_obs)

        if len(score_heading_angles) == 0:
            raise RuntimeError("All paths cannot avoid obstacles")

        # パラメータチューニングがうまくいかなかったため、正規化していない
        score_heading_angles_np = np.array(score_heading_angles)
        score_heading_vels_np = np.array(score_heading_vels)
        score_obstacles_np = np.array(score_obstacles)

        scores = (
            self.weight_angle * score_heading_angles_np
            + self.weight_vel * score_heading_vels_np
            + self.weight_obs * score_obstacles_np
        )
        return paths[scores.argmax()]

    def _calc_heading_angle_score(self, path, g_x, g_y):
        last_x = path.xs[-1]
        last_y = path.ys[-1]
        last_th = path.ths[-1]

        angle_to_goal = np.arctan2(g_y - last_y, g_x - last_x)
        score_angle = angle_to_goal - last_th
        # ぐるぐる防止
        score_angle = abs(angle_range_corrector(score_angle))

        # 最大と最小をひっくり返す
        score_angle = np.pi - score_angle

        return score_angle

    def _calc_heading_vel_score(self, path):
        return path.u_v

    def _calc_neighbor_obs(self, state, obstacles):
        neighbor_obs = []

        for obs in obstacles:
            temp_dis_to_obs = (state.x - obs.x) ** 2 + (state.y - obs.y) ** 2
            if temp_dis_to_obs < self.area_dis_to_obs_sqrd:
                neighbor_obs.append(obs)
        return neighbor_obs

    def _calc_obstacles_score(self, path, neighbor_obs):
        score_obstacle_sqrd = self.score_obstacle_sqrd
        for (path_x, path_y) in zip(path.xs, path.ys):
            for obs in neighbor_obs:
                temp_dis_to_obs = (path_x - obs.x) ** 2 + (path_y - obs.y) ** 2
                if temp_dis_to_obs < score_obstacle_sqrd:
                    score_obstacle_sqrd = temp_dis_to_obs
                if temp_dis_to_obs < obs.size + 0.75:  # マージン
                    return -float("inf")

        return np.sqrt(score_obstacle_sqrd)


class MainController:
    def __init__(self) -> None:
        self.samplingtime = 0.1

        self.robot = TwoWheeledRobot(0.0, 0.0, 0)
        self.goal_maker = ConstGoal()
        self.planner = DWA(self.samplingtime)

        self.obstacles = [
            Obstacle(4, 1, 0.25),
            Obstacle(0, 4.5, 0.25),
            # Obstacle(3, 4.5, 0.25),
            # Obstacle(5, 3.5, 0.25),
            Obstacle(7.5, 9.0, 0.25),
        ]
        # self.obstacles = []
        # for _ in range(10):
        #     x = np.random.randint(-5, 5)
        #     y = np.random.randint(-5, 5)
        #     size = 0.25
        #     self.obstacles.append(Obstacle(x, y, size))

    def run(self):
        time_step = 0
        goal_th = 0.5
        goal_th_sqrd = goal_th**2
        max_timestep = 500

        while True:
            g_x, g_y = self.goal_maker.calc_goal(time_step)

            _, opt_path = self.planner.calc_input(g_x, g_y, self.robot, self.obstacles)

            u_th = opt_path.u_th
            u_v = opt_path.u_v

            self.robot.update_state(u_th, u_v, self.samplingtime)

            dist_to_goal = (g_x - self.robot.x) ** 2 + (g_y - self.robot.y) ** 2

            if dist_to_goal < goal_th_sqrd:
                break
            time_step += 1
            if time_step >= max_timestep:
                break

        return (
            self.robot.traj_x,
            self.robot.traj_y,
            self.robot.traj_th,
            self.goal_maker.traj_g_x,
            self.goal_maker.traj_g_y,
            self.planner.traj_paths,
            self.planner.traj_opt,
            self.obstacles,
        )


def main():
    animation = Animation_robot()
    animation.fig_set()

    controller = MainController()
    (
        traj_x,
        traj_y,
        traj_th,
        traj_g_x,
        traj_g_y,
        traj_paths,
        traj_opt,
        obstacles,
    ) = controller.run()

    animation.func_anim_plot(
        traj_x, traj_y, traj_th, traj_paths, traj_g_x, traj_g_y, traj_opt, obstacles
    )


if __name__ == "__main__":
    main()
