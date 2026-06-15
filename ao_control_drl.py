"""
Adaptive Optics Simulation with Deep Reinforcement Learning Control
Using HCIPY for Dynamic Atmosphere, DM, and Sensor Image Generation.
Supports multiple DRL algorithms: PPO, SAC, TD3, DDPG with parallel training and observation modes.
"""
import os
import subprocess
import numpy as np
import hcipy as hc
import torch
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO, QRDQN, TQC
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================================
# 1. 动态自适应光学环境类(Dynamic AO Environment)
# ========================================================
class DynamicAOEnvironment:
    def __init__(self, pupil_grid_size=128, focal_grid_q=4, focal_num_airy=16, num_zernike=15):
        """
        初始化动态自适应光学仿真环境
        :param pupil_grid_size: 瞳孔网格大小（分辨率）
        :param focal_grid_q: 焦面采样率（每个Airy盘的像素数）
        :param focal_num_airy: 焦面网格大小（Airy盘半径）
        :param num_zernike: 考虑的泽尼克模式数量（排除Piston）
        """
        self.wavelength = 1e-6
        self.pupil_diameter = 1.0
        self.num_zernike = num_zernike

        # 创建瞳面网格与孔径
        self.pupil_grid = hc.make_pupil_grid(pupil_grid_size, self.pupil_diameter)
        self.aperture = hc.make_circular_aperture(self.pupil_diameter)(self.pupil_grid)

        # 创建焦面网格与传播器
        self.focal_grid = hc.make_focal_grid(focal_grid_q, focal_num_airy,
                                             spatial_resolution=self.wavelength/self.pupil_diameter)
        self.propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid)

        # 配置大气扰动
        fried_parameter = 0.06
        outer_scale = 20.0
        velocity = 10.0
        Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
        self.atmosphere = hc.InfiniteAtmosphericLayer(self.pupil_grid, Cn_squared, outer_scale, velocity)

        self.t = 0.0
        self.dt = 0.004

        # 创建泽尼克基底
        self.zernike_basis = hc.make_zernike_basis(num_zernike, self.pupil_diameter, self.pupil_grid, starting_mode=2)

        # 配置变形镜
        num_actuators_across = 11
        actuator_spacing = self.pupil_diameter / num_actuators_across
        influence_functions = hc.make_gaussian_influence_functions(
            pupil_grid=self.pupil_grid,
            num_actuators_across_pupil=num_actuators_across,
            actuator_spacing=actuator_spacing
        )
        self.dm = hc.DeformableMirror(influence_functions)

        # 预计算泽尼克相位系数到DM致动器高度的映射
        mask = self.aperture > 0
        zernike_matrix = np.asarray(self.zernike_basis.transformation_matrix[mask, :])
        influence_matrix = self.dm.influence_functions.transformation_matrix[mask, :]
        if hasattr(influence_matrix, "toarray"):
            influence_matrix = influence_matrix.toarray()
        else:
            influence_matrix = np.asarray(influence_matrix)
        pinv_influence = hc.inverse_tikhonov(influence_matrix, rcond=1e-3)
        self.zernike_to_dm_matrix = pinv_influence.dot(
            zernike_matrix * (self.wavelength / (4.0 * np.pi))
        )

        # 配置离焦相差
        defocus_basis = hc.make_zernike_basis(3, self.pupil_diameter, self.pupil_grid, starting_mode=4)
        self.defocus_phase = defocus_basis[0] * 2.0

        # 初始化扩展目标
        self.extended_object = self._generate_extended_target()
        self.current_dm_zernike_commands = np.zeros(self.num_zernike)

    def _generate_extended_target(self):
        """生成扩展目标图像"""
        shape = self.focal_grid.shape
        obj = np.zeros(shape)
        r, c = shape[0] // 2, shape[1] // 2
        obj[r-12:r+12, c-2:c+2] = 1.0
        obj[r-2:r+2, c-12:c+12] = 1.0
        return obj / np.sum(obj)

    def step(self, dm_commands=None):
        """
        环境向前推进一个时间步长
        """
        self.t += self.dt
        self.atmosphere.evolve_until(self.t)

        if dm_commands is not None:
            self.current_dm_zernike_commands = np.asarray(dm_commands, dtype=float)
        physical_actuators = self.zernike_to_dm_matrix.dot(self.current_dm_zernike_commands)
        self.dm.actuators = physical_actuators

        wf_in = hc.Wavefront(self.aperture, self.wavelength)
        wf_perturbed = self.atmosphere(wf_in)
        wf_corrected = self.dm(wf_perturbed)

        correction_phase = self.dm.phase_for(self.wavelength)
        atmo_phase = self.atmosphere.phase_for(self.wavelength)
        residual_phase = atmo_phase + correction_phase

        mask_2d = self.aperture.shaped > 0
        atmosphere_phase_2d = atmo_phase.shaped * mask_2d
        correction_phase_2d = correction_phase.shaped * mask_2d
        residual_phase_2d = residual_phase.shaped * mask_2d

        mask = self.aperture > 0
        basis_matrix = self.zernike_basis.transformation_matrix[mask, :]

        residual_zernike, _, _, _ = np.linalg.lstsq(basis_matrix, residual_phase[mask], rcond=None)
        open_loop_zernike, _, _, _ = np.linalg.lstsq(basis_matrix, atmo_phase[mask], rcond=None)

        # 开环状态
        open_psf_infocus = self.propagator(wf_perturbed).intensity.shaped
        wf_open_defocus = wf_perturbed.copy()
        wf_open_defocus.electric_field *= np.exp(1j * self.defocus_phase)
        open_psf_defocus = self.propagator(wf_open_defocus).intensity.shaped

        if open_psf_infocus.sum() > 0: open_psf_infocus /= open_psf_infocus.sum()
        if open_psf_defocus.sum() > 0: open_psf_defocus /= open_psf_defocus.sum()

        open_img_infocus = fftconvolve(self.extended_object, open_psf_infocus, mode='same')
        open_img_defocus = fftconvolve(self.extended_object, open_psf_defocus, mode='same')

        # 闭环状态
        psf_infocus = self.propagator(wf_corrected).intensity.shaped
        wf_defocus = wf_corrected.copy()
        wf_defocus.electric_field *= np.exp(1j * self.defocus_phase)
        psf_defocus = self.propagator(wf_defocus).intensity.shaped

        if psf_infocus.sum() > 0: psf_infocus /= psf_infocus.sum()
        if psf_defocus.sum() > 0: psf_defocus /= psf_defocus.sum()

        img_infocus = fftconvolve(self.extended_object, psf_infocus, mode='same')
        img_defocus = fftconvolve(self.extended_object, psf_defocus, mode='same')

        observation = {
            'img_infocus': img_infocus,
            'img_defocus': img_defocus,
            'psf_infocus': psf_infocus,
            'psf_defocus': psf_defocus,
            'open_img_infocus': open_img_infocus,
            'open_img_defocus': open_img_defocus,
            'open_psf_infocus': open_psf_infocus,
            'open_psf_defocus': open_psf_defocus,
            'atmosphere_phase': atmosphere_phase_2d,
            'dm_phase': correction_phase_2d,
            'residual_phase': residual_phase_2d
        }

        truth = {
            'residual_zernike': residual_zernike,
            'open_loop_zernike': open_loop_zernike
        }
        return observation, truth

    def reset(self):
        self.t = 0.0
        self.atmosphere.reset()
        self.current_dm_zernike_commands = np.zeros(self.num_zernike)
        self.dm.flatten()


# ========================================================
# 2. 深度强化学习Gymnasium环境
# ========================================================
class AOControlDRLEnv(gym.Env):
    """基于Gymnasium的自适应光学DRL环境"""
    metadata = {'render_modes': [None, 'rgb_array']}

    def __init__(self, num_zernike=15, pupil_grid_size=128, render_mode=None,
                 reward_type='strehl', max_steps=300):
        """
        :param num_zernike: 泽尼克模式数量
        :param pupil_grid_size: 瞳孔网格大小
        :param render_mode: 渲染模式
        :param reward_type: 奖励类型 ('strehl', 'mse_residual', 'hybrid')
        :param max_steps: 最大步数
        """
        super().__init__()
        self.num_zernike = num_zernike
        self.pupil_grid_size = pupil_grid_size
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.current_step = 0

        # 环境
        self.ao_env = DynamicAOEnvironment(pupil_grid_size=pupil_grid_size, num_zernike=num_zernike)

        # 动作空间：泽尼克系数的增量控制
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_zernike,), dtype=np.float32
        )

        # 观测空间：在焦和离焦PSF图像
        img_shape = self.ao_env.focal_grid.shape
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(2, img_shape[0], img_shape[1]), dtype=np.float32
        )

        # 历史记录
        self.dm_commands = np.zeros(num_zernike)
        self.prev_residual_rms = None
        self.prev_strehl = None
        self.frame_buffer = []

    def _normalize_image(self, img):
        """归一化图像到[0,1]"""
        img = np.clip(img, 0, None).astype(np.float32)
        vmax = img.max()
        if vmax > 0:
            img = img / vmax
        return img

    def _compute_strehl_ratio(self, psf):
        """计算斯特列尔比"""
        psf_norm = self._normalize_image(psf)
        peak = np.max(psf_norm)
        encircled_energy = np.sum(psf_norm)
        if encircled_energy > 0:
            strehl = peak / encircled_energy
        else:
            strehl = 0.0
        return np.clip(strehl, 0, 1)

    def _compute_reward(self, obs, truth):
        """计算奖励"""
        psf_infocus = obs['psf_infocus']
        residual_zernike = truth['residual_zernike']

        if self.reward_type == 'strehl':
            # 基于斯特列尔比的奖励
            strehl = self._compute_strehl_ratio(psf_infocus)
            reward = strehl
            # 增加对改进的激励
            if self.prev_strehl is not None:
                improvement = strehl - self.prev_strehl
                reward += 0.5 * np.tanh(improvement / 0.01)
            self.prev_strehl = strehl

        elif self.reward_type == 'mse_residual':
            # 基于残差波前MSE的奖励
            residual_rms = np.sqrt(np.mean(residual_zernike ** 2))
            # 反向MSE：RMS越小，奖励越大
            reward = np.exp(-residual_rms)
            if self.prev_residual_rms is not None:
                improvement = self.prev_residual_rms - residual_rms
                reward += 0.5 * np.tanh(improvement / 0.01)
            self.prev_residual_rms = residual_rms

        elif self.reward_type == 'hybrid':
            # 混合奖励：结合斯特列尔比和残差波前
            strehl = self._compute_strehl_ratio(psf_infocus)
            residual_rms = np.sqrt(np.mean(residual_zernike ** 2))
            norm_rms = np.exp(-residual_rms)
            reward = 0.6 * strehl + 0.4 * norm_rms
            if self.prev_strehl is not None:
                improvement_s = strehl - self.prev_strehl
                reward += 0.3 * np.tanh(improvement_s / 0.01)
            self.prev_strehl = strehl
            self.prev_residual_rms = residual_rms
        else:
            reward = 0.0

        return float(reward)

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.ao_env.reset()
        self.dm_commands = np.zeros(self.num_zernike)
        self.current_step = 0
        self.prev_residual_rms = None
        self.prev_strehl = None
        self.frame_buffer = []

        obs, _ = self.ao_env.step(dm_commands=self.dm_commands)
        psf_in = self._normalize_image(obs['psf_infocus'])
        psf_de = self._normalize_image(obs['psf_defocus'])
        state = np.stack([psf_in, psf_de], axis=0)

        return state.astype(np.float32), {}

    def step(self, action):
        """执行一步"""
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # 应用动作：泽尼克系数的增量控制（乘以缩放因子）
        action_scale = 0.3
        self.dm_commands = self.dm_commands + action * action_scale

        # 可选：限制命令范围以防止发散
        self.dm_commands = np.clip(self.dm_commands, -3.0, 3.0)

        # 环境步进
        obs, truth = self.ao_env.step(dm_commands=self.dm_commands)

        # 计算奖励
        reward = self._compute_reward(obs, truth)

        # 构建观测
        psf_in = self._normalize_image(obs['psf_infocus'])
        psf_de = self._normalize_image(obs['psf_defocus'])
        state = np.stack([psf_in, psf_de], axis=0)

        # 保存帧用于渲染
        if self.render_mode == 'rgb_array':
            self.frame_buffer.append({
                'obs': obs,
                'truth': truth,
                'reward': reward,
                'dm_commands': self.dm_commands.copy()
            })

        truncated = False
        return state.astype(np.float32), float(reward), terminated, truncated, {}

    def render(self):
        """渲染（在本环境中通过后期处理实现）"""
        pass


# ========================================================
# 3. 自定义回调用于监控训练
# ========================================================
class AOControlCallback(BaseCallback):
    """训练回调，记录关键指标"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        pass


# ========================================================
# 4. 深度强化学习训练系统
# ========================================================
class DRLAOControlSystem:
    """深度强化学习自适应光学控制系统"""

    def __init__(self, num_zernike=15, pupil_grid_size=128, algorithm='PPO',
                 reward_type='strehl', model_dir='./drl_models', log_dir='./drl_logs'):
        """
        初始化DRL系统
        :param num_zernike: 泽尼克模式数
        :param pupil_grid_size: 瞳孔网格大小
        :param algorithm: DRL算法 ('PPO', 'SAC', 'TD3', 'DDPG', 'RecurrentPPO', 'TQC')
        :param reward_type: 奖励类型
        :param model_dir: 模型保存目录
        :param log_dir: 日志目录
        """
        self.num_zernike = num_zernike
        self.pupil_grid_size = pupil_grid_size
        self.algorithm = algorithm
        self.reward_type = reward_type
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.env = None
        self.model = None
        self.best_mean_reward = -np.inf
        self.best_model_path = os.path.join(self.model_dir, f"best_model_{algorithm}.zip")

    def _create_model(self, env):
        """创建指定算法的模型"""
        logger.info(f"Creating model with algorithm: {self.algorithm}")

        policy_kwargs = dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU
        )

        if self.algorithm == 'PPO':
            model = PPO(
                'CnnPolicy',
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                device=self.device,
                verbose=1,
                tensorboard_log=self.log_dir
            )
        elif self.algorithm == 'SAC':
            model = SAC(
                'CnnPolicy',
                env,
                learning_rate=3e-4,
                buffer_size=10000,
                batch_size=64,
                ent_coef='auto',
                target_entropy='auto',
                device=self.device,
                verbose=1,
                tensorboard_log=self.log_dir
            )
        elif self.algorithm == 'TD3':
            model = TD3(
                'CnnPolicy',
                env,
                learning_rate=3e-4,
                buffer_size=10000,
                batch_size=64,
                policy_delay=2,
                target_policy_noise=0.2,
                device=self.device,
                verbose=1,
                tensorboard_log=self.log_dir
            )
        elif self.algorithm == 'DDPG':
            model = DDPG(
                'CnnPolicy',
                env,
                learning_rate=3e-4,
                buffer_size=10000,
                batch_size=64,
                device=self.device,
                verbose=1,
                tensorboard_log=self.log_dir
            )
        elif self.algorithm == 'RecurrentPPO':
            model = RecurrentPPO(
                'CnnLstmPolicy',
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                device=self.device,
                verbose=1,
                tensorboard_log=self.log_dir
            )
        elif self.algorithm == 'TQC':
            model = TQC(
                'CnnPolicy',
                env,
                learning_rate=3e-4,
                buffer_size=10000,
                batch_size=64,
                device=self.device,
                verbose=1,
                tensorboard_log=self.log_dir
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return model

    def train_parallel(self, num_envs=4, total_timesteps=100000, save_interval=10000):
        """
        模式一：并行训练模式
        高效的并行环境训练，不生成视频
        :param num_envs: 并行环境数
        :param total_timesteps: 总训练步数
        :param save_interval: 保存模型间隔
        """
        from stable_baselines3.common.vec_env import SubprocVecEnv

        logger.info(f"Starting parallel training with {num_envs} environments")
        logger.info(f"Algorithm: {self.algorithm}, Reward type: {self.reward_type}")
        logger.info(f"Total timesteps: {total_timesteps}")

        def make_env(rank):
            def _init():
                env = AOControlDRLEnv(
                    num_zernike=self.num_zernike,
                    pupil_grid_size=self.pupil_grid_size,
                    reward_type=self.reward_type,
                    max_steps=300
                )
                return env
            return _init()

        # 创建向量化环境
        vec_env = SubprocVecEnv([lambda: make_env(i) for i in range(num_envs)])

        # 创建模型
        self.model = self._create_model(vec_env)

        # 训练循环
        checkpoint_timesteps = 0
        best_mean_reward = -np.inf

        for step in range(0, total_timesteps, save_interval):
            remaining = min(save_interval, total_timesteps - step)
            logger.info(f"Training... ({step}/{total_timesteps})")
            self.model.learn(total_timesteps=remaining, reset_num_timesteps=False)

            # 评估
            mean_reward, std_reward = self._evaluate_policy(vec_env, num_episodes=5)
            logger.info(f"Timestep {step + remaining}: Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")

            # 保存最佳模型
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
                logger.info(f"Best model saved to {self.best_model_path}")

            checkpoint_timesteps += remaining

        vec_env.close()
        logger.info(f"Parallel training completed. Best model: {self.best_model_path}")
        return self.best_model_path

    def _evaluate_policy(self, env, num_episodes=5):
        """评估策略"""
        episode_rewards = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    def observe_trained_model(self, model_path, num_steps=300, video_name="ao_drl_observation.mp4",
                              fps=30):
        """
        模式二：观测模式
        加载训练好的模型，生成观测视频
        :param model_path: 训练好的模型路径
        :param num_steps: 观测步数
        :param video_name: 视频输出名
        :param fps: 视频帧率
        """
        logger.info(f"Loading model from {model_path}")
        
        # 创建环境（启用rgb_array渲染）
        env = AOControlDRLEnv(
            num_zernike=self.num_zernike,
            pupil_grid_size=self.pupil_grid_size,
            reward_type=self.reward_type,
            render_mode='rgb_array',
            max_steps=num_steps
        )

        # 加载模型
        algo_class = self._get_algorithm_class()
        model = algo_class.load(model_path, env=env, device=self.device)

        logger.info("Starting observation mode...")
        obs, _ = env.reset()

        # 构建画布
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=100)
        fig.suptitle(f"AO Control DRL Observation - {self.algorithm} ({self.reward_type})",
                     fontsize=16)

        frames_bytes = []
        rewards_history = []
        residual_rms_history = []

        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            if env.frame_buffer:
                frame_data = env.frame_buffer[-1]
                frame_obs = frame_data['obs']
                frame_truth = frame_data['truth']
                frame_reward = frame_data['reward']
                frame_dm = frame_data['dm_commands']

                rewards_history.append(frame_reward)
                residual_rms = np.sqrt(np.mean(frame_truth['residual_zernike'] ** 2))
                residual_rms_history.append(residual_rms)

                # 清空子图
                for ax in axes.flat:
                    ax.clear()

                # 绘制
                im1 = axes[0, 0].imshow(frame_obs['atmosphere_phase'], cmap='RdBu')
                axes[0, 0].set_title("Atmosphere Phase")
                plt.colorbar(im1, ax=axes[0, 0])

                im2 = axes[0, 1].imshow(frame_obs['dm_phase'], cmap='RdBu')
                axes[0, 1].set_title("DM Correction Phase")
                plt.colorbar(im2, ax=axes[0, 1])

                im3 = axes[0, 2].imshow(frame_obs['residual_phase'], cmap='RdBu')
                axes[0, 2].set_title("Residual Phase")
                plt.colorbar(im3, ax=axes[0, 2])

                im4 = axes[1, 0].imshow(frame_obs['psf_infocus'], cmap='inferno')
                axes[1, 0].set_title(f"PSF In-focus (Reward: {frame_reward:.4f})")
                plt.colorbar(im4, ax=axes[1, 0])

                im5 = axes[1, 1].imshow(frame_obs['psf_defocus'], cmap='inferno')
                axes[1, 1].set_title("PSF Defocus")
                plt.colorbar(im5, ax=axes[1, 1])

                # 性能曲线
                axes[1, 2].plot(rewards_history, 'b-', label='Reward', alpha=0.7)
                axes[1, 2].plot(residual_rms_history, 'r--', label='Residual RMS', alpha=0.7)
                axes[1, 2].set_title("Performance Metrics")
                axes[1, 2].set_xlabel("Step")
                axes[1, 2].legend()
                axes[1, 2].grid(True)

                plt.tight_layout()
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                frames_bytes.append(bytes(fig.canvas.buffer_rgba()))

                logger.info(f"Step {step + 1}/{num_steps}: Reward={frame_reward:.4f}, "
                           f"Residual RMS={residual_rms:.4f}")

            if terminated or truncated:
                break

        plt.close(fig)
        logger.info(f"Captured {len(frames_bytes)} frames")

        # 编码视频
        if frames_bytes:
            self._encode_video(frames_bytes, video_name, width, height, fps)

        return video_name

    def _get_algorithm_class(self):
        """获取算法类"""
        algorithm_map = {
            'PPO': PPO,
            'SAC': SAC,
            'TD3': TD3,
            'DDPG': DDPG,
            'RecurrentPPO': RecurrentPPO,
            'TQC': TQC,
        }
        return algorithm_map.get(self.algorithm)

    def _encode_video(self, frames_bytes, video_name, width, height, fps):
        """使用FFmpeg编码视频"""
        logger.info(f"Encoding video: {video_name}")
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-vcodec', 'rawvideo', '-pix_fmt', 'rgba',
            '-s', f'{width}x{height}', '-r', f'{fps}',
            '-i', '-', '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', '-b:v', '3000k',
            video_name
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            all_video_data = b"".join(frames_bytes)
            proc.communicate(input=all_video_data)
            logger.info(f"Video saved to: {video_name}")
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")


# ========================================================
# 5. 主程序接口
# ========================================================
def main_parallel_training():
    """并行训练模式示例"""
    logger.info("=== Starting Parallel Training Mode ===")

    system = DRLAOControlSystem(
        num_zernike=15,
        pupil_grid_size=128,
        algorithm='PPO',  # 可选: 'PPO', 'SAC', 'TD3', 'DDPG', 'RecurrentPPO', 'TQC'
        reward_type='hybrid',  # 可选: 'strehl', 'mse_residual', 'hybrid'
        model_dir='./drl_models',
        log_dir='./drl_logs'
    )

    best_model = system.train_parallel(
        num_envs=4,
        total_timesteps=500000,
        save_interval=50000
    )

    logger.info(f"Training complete. Best model: {best_model}")
    return best_model


def main_observation_mode(model_path):
    """观测模式示例"""
    logger.info("=== Starting Observation Mode ===")

    system = DRLAOControlSystem(
        num_zernike=15,
        pupil_grid_size=128,
        algorithm='PPO',
        reward_type='hybrid',
        model_dir='./drl_models'
    )

    video_path = system.observe_trained_model(
        model_path=model_path,
        num_steps=500,
        video_name='ao_drl_observation.mp4',
        fps=30
    )

    logger.info(f"Observation complete. Video: {video_path}")
    return video_path


if __name__ == "__main__":
    import sys

    # 模式一：并行训练
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        best_model_path = main_parallel_training()
        print(f"\nBest model saved at: {best_model_path}")
        print("To run observation mode, use: python ao_control_drl.py observe <model_path>")

    # 模式二：观测
    elif len(sys.argv) > 2 and sys.argv[1] == 'observe':
        model_path = sys.argv[2]
        video_path = main_observation_mode(model_path)
        print(f"\nObservation video saved at: {video_path}")

    else:
        print("Usage:")
        print("  Parallel Training Mode (high-efficiency, no video):")
        print("    python ao_control_drl.py train")
        print("\n  Observation Mode (with video generation):")
        print("    python ao_control_drl.py observe <model_path>")
        print("\nExample:")
        print("  python ao_control_drl.py train")
        print("  python ao_control_drl.py observe ./drl_models/best_model_PPO.zip")
