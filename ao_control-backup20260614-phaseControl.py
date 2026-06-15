"""
Adaptive Optics Simulation and Closed-Loop Verification Project
Using HCIPY for Dynamic Atmosphere, DM, and Sensor Image Generation.
"""
import os
import subprocess
import numpy as np
import hcipy as hc
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use('Agg')   # 强制使用无GUI的Agg后端
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter


# add
from torch_npu.contrib import transfer_to_npu


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
        self.wavelength = 1e-6    # 仿真波长： 1微米
        self.pupil_diameter = 1.0 # 望远镜瞳径： 1米
        self.num_zernike = num_zernike

        # 创建瞳面网格与孔径
        self.pupil_grid = hc.make_pupil_grid(pupil_grid_size, self.pupil_diameter)
        self.aperture = hc.make_circular_aperture(self.pupil_diameter)(self.pupil_grid)

        # 创建焦面网格与传播器
        self.focal_grid = hc.make_focal_grid(focal_grid_q, focal_num_airy,
                                             spatial_resolution=self.wavelength/self.pupil_diameter)
        self.propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid)

        # 配置大气扰动（泰勒冻结流模型，连续流动）
        fried_parameter = 0.06      # 弗里德参数 r0 = 6 cm
        outer_scale = 20.0          # 大气外尺度 L0 = 20 m
        velocity = 10.0             # 风速： 10 m/s
        Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
        self.atmosphere = hc.InfiniteAtmosphericLayer(self.pupil_grid, Cn_squared, outer_scale, velocity)

        self.t = 0.0
        self.dt = 0.004             # 时间步长： 4ms

        # 创建泽尼克基底（从Noll索引2开始，排除1-Piston）
        self.zernike_basis = hc.make_zernike_basis(num_zernike, self.pupil_diameter, self.pupil_grid, starting_mode=2)

        # # 配置真实连续面形变形镜(高斯影响函数)
        # num_actuators_across = 11   # 沿口径方向11x11致动器
        # actuator_spacing = self.pupil_diameter / num_actuators_across
        # # 生成高斯影响函数（表面高度基，单位：米）
        # influence_functions = hc.make_gaussian_influence_functions(
        #     pupil_grid=self.pupil_grid,
        #     num_actuators_across_pupil=num_actuators_across,
        #     actuator_spacing=actuator_spacing
        # )
        # self.dm = hc.DeformableMirror(influence_functions)
        # # 获取泽尼克矩阵
        # zernike_matrix = self.zernike_basis.transformation_matrix
        # influence_matrix = self.dm.influence_functions.transformation_matrix
        # pinv_influence = hc.inverse_tikhonov(influence_matrix, rcond=1e-3)
        # self.zernike_to_dm_matrix = pinv_influence.dot(zernike_matrix)

        # 配置离焦相差（用于生成相差多样性所需的离焦图像，采用Noll 4 Defocus）
        defocus_basis = hc.make_zernike_basis(3, self.pupil_diameter, self.pupil_grid, starting_mode=4)
        self.defocus_phase = defocus_basis[0] * 2.0   # 施加2弧度幅度的固定离焦

        # 初始化扩展目标
        self.extended_object = self._generate_extended_target()
        # 初始化校正相位屏相位
        self.current_phase_commands = np.zeros(self.num_zernike)

    def _generate_extended_target(self):
        """生成一个用于仿真的扩展目标图像（十字靶标）"""
        shape = self.focal_grid.shape
        obj = np.zeros(shape)
        r, c = shape[0] // 2, shape[1] // 2
        obj[r-12:r+12, c-2:c+2] = 1.0
        obj[r-2:r+2, c-12:c+12] = 1.0
        return obj / np.sum(obj)

    def step(self, phase_commands=None):
        """
        环境向前推进一个时间步长
        :param phase_commands: 相位屏校正控制量（泽尼克模式系数向量）
        :return: observation（图像及相位网格），truth（波前真实泽尼克系数）
        """
        # 大气演化
        self.t += self.dt
        self.atmosphere.evolve_until(self.t)

        # # 更新变形镜
        # if dm_commands is not None:
        #     self.current_zernike_commands = dm_commands
        # physical_actuators = self.zernike_to_dm_matrix.dot(self.current_zernike_commands)
        # self.dm.actuators = physical_actuators

        # 更新相位屏指令
        if phase_commands is not None:
            self.current_phase_commands = phase_commands

        # 校正相位
        correction_phase = self.zernike_basis.linear_combination(self.current_phase_commands)
        phase_screen = hc.Apodizer(np.exp(1j * correction_phase))

        # 构建入射波前并物理传播
        wf_in = hc.Wavefront(self.aperture, self.wavelength)
        wf_perturbed = self.atmosphere(wf_in)
        wf_corrected = phase_screen(wf_perturbed)

        # 提取真实的、未被2pi截断的连续物理相位（Unwrapped Phase)
        atmo_phase = self.atmosphere.phase_for(self.wavelength)
        # 光路中相位是线性叠加的
        residual_phase = atmo_phase + correction_phase


        # 转换为二维网格用于图像可视化
        mask_2d = self.aperture.shaped > 0
        atmosphere_phase_2d = atmo_phase.shaped * mask_2d
        correction_phase_2d = correction_phase.shaped * mask_2d
        residual_phase_2d = residual_phase.shaped * mask_2d

        # 最小二乘投影计算泽尼克模式系数（无额外归一化）
        mask = self.aperture > 0
        basis_matrix = self.zernike_basis.transformation_matrix[mask, :]

        residual_zernike, _, _, _ = np.linalg.lstsq(basis_matrix, residual_phase[mask], rcond=None)
        open_loop_zernike, _, _, _ = np.linalg.lstsq(basis_matrix, atmo_phase[mask], rcond=None)

        # -----A. 开环/未校正状态  ---
        # 生成PSF图像
        open_psf_infocus = self.propagator(wf_perturbed).intensity.shaped
        wf_open_defocus = wf_perturbed.copy()
        wf_open_defocus.electric_field *= np.exp(1j * self.defocus_phase)
        open_psf_defocus = self.propagator(wf_open_defocus).intensity.shaped

        if open_psf_infocus.sum() > 0: open_psf_infocus /= open_psf_infocus.sum()
        if open_psf_defocus.sum() > 0: open_psf_defocus /= open_psf_defocus.sum()

        # 卷积得到扩展目标
        open_img_infocus = fftconvolve(self.extended_object, open_psf_infocus, mode='same')
        open_img_defocus = fftconvolve(self.extended_object, open_psf_defocus, mode='same')


        # -----B. 闭环/校正状态  ---
        # 生成PSF图像
        psf_infocus = self.propagator(wf_corrected).intensity.shaped
        wf_defocus = wf_corrected.copy()
        wf_defocus.electric_field *= np.exp(1j * self.defocus_phase)
        psf_defocus = self.propagator(wf_defocus).intensity.shaped

        if psf_infocus.sum() > 0: psf_infocus /= psf_infocus.sum()
        if psf_defocus.sum() > 0: psf_defocus /= psf_defocus.sum()

        # 卷积得到扩展目标
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
            'screen_phase': correction_phase_2d,
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
        self.current_phase_commands = np.zeros(self.num_zernike)
        # self.dm.actuators = np.zeros(self.dm.num_actuators)

# ========================================================
# 2. 神经网络交互接口（Neural Network Predictor）
# ========================================================
class WavefrontSensorNN:
    def __init__(self, model_path=None, model_class_name="ZernikeNet", num_modes=15, in_channels=2, device=None):
        self.model_path = model_path
        self.has_model = False
        self.num_modes = num_modes
        self.in_channels = in_channels
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if model_path and os.path.exists(model_path):
            from model import ZernikeNet
            model_map = {
                "ZernikeNet": lambda: ZernikeNet(num_outputs=num_modes,in_channels=in_channels, weight_path=None)
            }
            if model_class_name not in model_map:
                raise ValueError(f"未知模型类型:{model_class_name}, 可选:{list(model_map.keys())}")
            self.model = model_map[model_class_name]().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.has_model = True
            print(f"成功加载训练好的神经网络模型：{model_class_name} @ {model_path}")
        else:
            print("未加载物理模型，交互时将使用高保真模拟器预测（用于流程验证）。")

    def _preprocess(self, infocus, defocus):
        # 还原数据收集时的变换逻辑
        def normalize_frame(img):
            img = np.clip(img, 0, None)
            vmax = img.max()
            if vmax > 0:
                img = img / vmax
            return img.astype(np.float32)

        in_norm = normalize_frame(infocus)
        de_norm = normalize_frame(defocus)

        stacked = np.stack([in_norm, de_norm], axis=-1)
        x = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).to(self.device)
        x = torch.log1p(x)
        return x


    def predict(self, infocus, defocus):
        """输入在焦和离焦图像，输出预测的残差泽尼克系数向量"""
        if not self.has_model:
            return None

        x = self._preprocess(infocus, defocus)
        with torch.no_grad():
            pred = self.model(x).detach().cpu().numpy()[0]
        return pred


# ========================================================
# 3. 开环仿真数据收集（Data Collection）
# ========================================================
def _to_uint8_image(img):
    img = np.asarray(img, dtype=np.float32)
    img = np.clip(img, 0, None)
    vmax = float(img.max())
    if vmax > 0:
        img = img / vmax
    return (img * 255.0).astype(np.uint8)

def do_data_collection(num_frames=500, save_path="./dataset/ao_simulated"):
    """
    生成丰富化的仿真数据：包含开环大气扰动、闭环微小残差、以及发散边缘的随机畸变。
    以此解决因“分布偏移”导致的闭环发散问题。
    """
    env = DynamicAOEnvironment(pupil_grid_size=128, num_zernike=15)
    print("\n>>> 开始收集抗发散的丰富化仿真数据...")
    out_dir = save_path
    os.makedirs(out_dir, exist_ok=True)

    # 维持一个模拟的相位屏当前指令
    screen_commands = np.zeros(env.num_zernike)

    for frame in range(num_frames):
        # 推进环境：此时相机拍到的，是叠加了screen_commands后的残差光场
        obs, truth = env.step(phase_commands=screen_commands)
        idx = frame + 1

        Image.fromarray(_to_uint8_image(obs['psf_infocus']), mode='L').save(
            os.path.join(out_dir, f"imgIF{idx}.jpg")
        )
        Image.fromarray(_to_uint8_image(obs['psf_defocus']), mode='L').save(
            os.path.join(out_dir, f"imgPoDF{idx}.jpg")
        )
        np.savetxt(
            os.path.join(out_dir, f"Zernike{idx}.csv"),
            truth['residual_zernike'],
            delimiter=","
        )

        # 为下一帧设计“教学场景”（制造多样化的训练分布）
        rand_scenario = np.random.rand()

        if rand_scenario < 0.2:
            # 场景A（20%概率）：纯开环状态
            # 强制指令归零，让模型学习应对原始的大尺度大气湍流
            screen_commands = np.zeros(env.num_zernike)
        elif rand_scenario < 0.6:
            # 场景B（40%概率）：模拟健康的闭环收敛中间态
            # 利用当前真值做一次带随机增益的控制，制造各种微小残差的图像
            gain = np.random.uniform(0.3,0.9)
            # 理想控制律：新指令 = 旧指令 - 增益*测量的残差误差
            screen_commands = screen_commands - gain * truth['residual_zernike']
        elif rand_scenario < 0.8:
            # 场景C（20%概率）：模拟预测错误/发散边缘
            # 故意注入随机的泽尼克噪声，强迫模型见识“越校越歪”的情况并学习纠偏
            noise = np.random.normal(0,0.4,env.num_zernike)
            gain = np.random.uniform(0.1,0.5)
            screen_commands = screen_commands - gain * truth['residual_zernike'] + noise
        else:
            # 场景D（20%概率）：彻底发散与灾难恢复（Loos of Lock）
            # 无视当前收敛状态，直接向相位屏注入幅值极大的随机畸变（标准差1.5）
            # 此时焦面上的图像大概率已经完全碎裂，强迫模型学习如何从混沌中找回梯度
            massive_chaos = np.random.normal(0,1.5,env.num_zernike)
            screen_commands = massive_chaos

        if (frame + 1) % 100 == 0:
            print(f"已生成{frame + 1} / {num_frames}帧数据...")

    print(f">>> 数据集生成完毕，成功存盘至：{out_dir}\n")

# ========================================================
# 4. 实时闭环交互验证与监控视频生成（Closed-Loop & Video）
# ========================================================
def do_closed_loop_verification(nn_sensor, num_steps=120, loop_gain=0.3, video_name="ao_interaction_verification.mp4"):
    """运行闭环系统，与神经网络模型进行实时交互，并保存整个物理状态视频"""
    env = DynamicAOEnvironment(pupil_grid_size=128, num_zernike=15)
    # 连续面形变形镜的致动器命令
    phase_commands = np.zeros(env.num_zernike)


    # 构建2x3画布展示各个物理环节
    fig, axes = plt.subplots(3,4,figsize=(12,8), dpi=150)
    fig.suptitle("Adaptive Optics Closed-Loop Real-time Interaction System", fontsize=15)

    open_rms_history = []
    res_rms_history = []

    # 预加载首帧设定图像参数
    obs, truth = env.step(phase_commands=phase_commands)

    # 第一行：相位环路与控制指标
    im_atmo = axes[0,0].imshow(obs['atmosphere_phase'], cmap='RdBu', vmin=-50, vmax=50)
    axes[0,0].set_title("1. Input Atmosphere Phase\n(Continues Disturbance)")
    fig.colorbar(im_atmo, ax=axes[0,0])

    im_screen = axes[0,1].imshow(obs['screen_phase'], cmap='RdBu', vmin=-20, vmax=20)
    axes[0,1].set_title("2. Phase Screen Phase\n(Real-time Apodizer)")
    fig.colorbar(im_screen, ax=axes[0,1])

    im_res = axes[0,2].imshow(obs['residual_phase'], cmap='RdBu', vmin=-50, vmax=50)
    axes[0,2].set_title("3. Residual Phase\n(Corrected Wavefront)")
    fig.colorbar(im_res, ax=axes[0,2])

    line_open, = axes[0, 3].plot([], [], color='red', linestyle='-', label='Uncorrected RMS', alpha=0.6)
    line_res, = axes[0, 3].plot([], [], color='blue', linestyle='--', label='Corrected RMS', alpha=0.6)
    axes[0, 3].set_title("4. Convergence Performance")
    axes[0, 3].set_xlim(0, num_steps)
    axes[0, 3].set_ylim(0, 3.0)
    axes[0, 3].legend(loc='upper right')
    axes[0, 3].grid(True)

    # 第二行：没有施加DM时的成像（开环）
    im_psf1_open = axes[1,0].imshow(obs['open_psf_infocus'], cmap='inferno')
    axes[1,0].set_title("5. Uncorrected PSF: In-focus")
    fig.colorbar(im_psf1_open, ax=axes[1,0])

    im_psf2_open = axes[1,1].imshow(obs['open_psf_defocus'], cmap='inferno')
    axes[1,1].set_title("6. Uncorrected PSF: Defocus")
    fig.colorbar(im_psf2_open, ax=axes[1,1])

    im_cam1_open = axes[1,2].imshow(obs['open_img_infocus'], cmap='inferno')
    axes[1,2].set_title("7. Uncorrected Sensor: In-focus Image")

    im_cam2_open = axes[1,3].imshow(obs['open_img_defocus'], cmap='inferno')
    axes[1,3].set_title("8. Uncorrected Sensor: Defocus Image")

    # 第三行：施加了DM后的实时成像（闭环）
    im_psf1 = axes[2,0].imshow(obs['psf_infocus'], cmap='inferno')
    axes[2,0].set_title("9. PSF: In-focus")
    fig.colorbar(im_psf1, ax=axes[2,0])

    im_psf2 = axes[2,1].imshow(obs['psf_defocus'], cmap='inferno')
    axes[2,1].set_title("10. PSF: Defocus")
    fig.colorbar(im_psf2, ax=axes[2,1])

    im_cam1 = axes[2,2].imshow(obs['img_infocus'], cmap='inferno')
    axes[2,2].set_title("11. Sensor: In-focus Image")

    im_cam2 = axes[2,3].imshow(obs['img_defocus'], cmap='inferno')
    axes[2,3].set_title("12. Sensor: Defocus Image")

    plt.tight_layout()
    frames_bytes = []
    print(">>> [1/2] 开始闭环仿真并实时抓取视频...")

    for step in range(num_steps):
        # 1. 环境更新：推进大气的状态，并将上一帧计算的指令发送给变形镜
        obs, truth = env.step(phase_commands=phase_commands)

        img_in, img_de = obs['img_infocus'], obs['img_defocus']
        psf_in, psf_de = obs['psf_infocus'], obs['psf_defocus']
        img_in_open, img_de_open = obs['open_img_infocus'], obs['open_img_defocus']
        psf_in_open, psf_de_open = obs['open_psf_infocus'], obs['open_psf_defocus']

        # 2. 交互环节：将环境新生成的观测图像喂给神经网络组件
        pred_residual = nn_sensor.predict(psf_in, psf_de)

        # 流程保护占位
        if pred_residual is None:
            # 模拟一个带有些许测量迟滞和微弱高斯噪声的闭环收敛过程
            pred_residual = truth['residual_zernike'] * 0.70 + np.random.normal(0, 0.04, env.num_zernike)

        # 3. 实时控制律计算
        phase_commands = phase_commands - loop_gain * pred_residual

        # 计算开闭环波前统计标准差（RMS）评估性能
        open_rms = np.std(truth['open_loop_zernike'])
        res_rms = np.std(truth['residual_zernike'])
        open_rms_history.append(open_rms)
        res_rms_history.append(res_rms)

        # 4. 刷新视频当前帧的图像数据
        im_atmo.set_data(obs['atmosphere_phase'])
        im_screen.set_data(obs['screen_phase'])
        im_res.set_data(obs['residual_phase'])
        line_open.set_data(range(len(open_rms_history)), open_rms_history)
        line_res.set_data(range(len(res_rms_history)), res_rms_history)

        # 行2（无DM开环数据刷新）
        im_psf1_open.set_data(psf_in_open)
        im_psf2_open.set_data(psf_de_open)
        im_cam1_open.set_data(img_in_open)
        im_cam2_open.set_data(img_de_open)

        # 行3（有DM闭环数据刷新）
        im_psf1.set_data(psf_in)
        im_psf2.set_data(psf_de)
        im_cam1.set_data(img_in)
        im_cam2.set_data(img_de)

        # 动态调整对比度上限（保证图像不会因为绝对亮度过亮或过暗而丧失对比度细节）
        im_psf1_open.set_clim(vmin=0, vmax=psf_in_open.max())
        im_psf2_open.set_clim(vmin=0, vmax=psf_de_open.max())
        im_cam1_open.set_clim(vmin=0, vmax=img_in_open.max())
        im_cam2_open.set_clim(vmin=0, vmax=img_de_open.max())

        im_psf1.set_clim(vmin=0, vmax=psf_in.max())
        im_psf2.set_clim(vmin=0, vmax=psf_de.max())
        im_cam1.set_clim(vmin=0, vmax=img_in.max())
        im_cam2.set_clim(vmin=0, vmax=img_de.max())

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frames_bytes.append(bytes(fig.canvas.buffer_rgba()))

        if (step + 1) % 20 == 0:
            print(f"  视频录制进度:{step+1}/{num_steps} | 原始扰动:{open_rms:.3f} | 校正残差{res_rms:.3f}")

    plt.close(fig)
    print(f">>> 仿真结束！已在内存中缓存{len(frames_bytes)}帧画面（分辨率:{width}x{height}）")
    print(f">>> [2/2] 正在安全调用系统FFmpeg核心进行高性能 H.264视频编码...")

    fps = 30
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgba',
        '-s', f'{width}x{height}', '-r', f'{fps}',
        '-i', '-', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-b:v', '3000k',
        video_name
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        all_video_data = b"".join(frames_bytes)
        proc.communicate(input=all_video_data)
        print(f">>> 实时验证完毕！交互控制监控视频已写入至:{video_name}")
    except Exception as e:
        print(f">>> [错误] 调用系统FFmpeg失败，错误信息{e}")



    print("do_closed_loop_verification finished")




if __name__ == "__main__":
    nn_sensor = WavefrontSensorNN(model_path="./weights/model_best.pth",model_class_name="ZernikeNet")
    do_closed_loop_verification(nn_sensor=nn_sensor,
                                num_steps=2000,
                                loop_gain=0.3
                                )
    # do_data_collection(num_frames=10000)
