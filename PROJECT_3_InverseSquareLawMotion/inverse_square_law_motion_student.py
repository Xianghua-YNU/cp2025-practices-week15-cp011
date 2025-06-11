"""
Module: InverseSquareLawMotion Solution
File: inverse_square_law_motion_solution.py
Author: Trae AI
Date: 2025-06-04

本模块用于模拟一个粒子在反平方中心力场（如引力或库仑力）中所受的运动，特别是轨道运动。
适用于椭圆轨道（行星轨道）、抛物线轨道（逃逸临界速度）和双曲线轨道（超越逃逸速度）。
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 常量设定（可根据需要调整）---
GM = 1.0  # 中心天体质量与万有引力常数的乘积 G*M，例如太阳为中心时 GM = G*M_sun
# 假设轨道粒子质量为1，便于简化计算为单位质量下的“比能量”“比角动量”

# --- 定义状态方程 ---
def derivatives(t, state_vector, gm_val):
    """
    计算状态变量的一阶导数：[x, y, vx, vy] 的导数 [vx, vy, ax, ay]

    参数：
        t (float): 当前时间。对本问题而言为自治系统，t不直接出现在表达式中。
        state_vector (np.ndarray): 当前状态 [x, y, vx, vy]
        gm_val (float): 万有引力常数乘以中心质量 GM

    返回：
        np.ndarray: 导数数组 [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state_vector
    r_cubed = (x**2 + y**2) ** 1.5  # 计算 r^3

    # 如果粒子接近中心（r趋近于0），避免除以0
    if r_cubed < 1e-12:
        ax = -gm_val * x / (1e-12) if x != 0 else 0
        ay = -gm_val * y / (1e-12) if y != 0 else 0
        return [vx, vy, ax, ay]

    # 由牛顿万有引力公式计算加速度分量
    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    return [vx, vy, ax, ay]

# --- 数值求解轨道函数 ---
def solve_orbit(initial_conditions, t_span, t_eval, gm_val=GM):
    """
    用SciPy的solve_ivp函数求解轨道微分方程组

    参数：
        initial_conditions: 初始状态 [x0, y0, vx0, vy0]
        t_span: 时间区间 (t_start, t_end)
        t_eval: 需要记录解的时间点数组
        gm_val: 万有引力常数乘以中心质量 GM

    返回：
        OdeSolution: 包含时间序列、位置速度等解的对象（sol.t，sol.y）
    """
    sol = solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='RK45',  # 使用Runge-Kutta 4(5)阶方法
        rtol=1e-7,      # 相对误差控制
        atol=1e-9       # 绝对误差控制
    )
    return sol

# --- 能量计算 ---
def calculate_energy(state_vector, gm_val=GM, m=1.0):
    """
    计算粒子的机械能（单位质量），用于判断轨道类型（E<0椭圆，E=0抛物线，E>0双曲线）

    参数：
        state_vector: 状态数组 [x, y, vx, vy]（支持多组）
        gm_val: GM参数
        m: 粒子质量，默认为1（计算“比能量”）

    返回：
        能量数组或标量：E = 0.5*v^2 - GM/r
    """
    is_single_state = (state_vector.ndim == 1)
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)

    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]

    r = np.sqrt(x**2 + y**2)
    v_squared = vx**2 + vy**2

    # 处理 r=0 时的能量发散
    potential_energy_per_m = np.zeros_like(r)
    non_zero_r_mask = (r > 1e-12)
    potential_energy_per_m[non_zero_r_mask] = -gm_val / r[non_zero_r_mask]
    if np.any(~non_zero_r_mask):
        print("Warning: r=0 encountered in energy calculation. Potential energy is singular.")
        potential_energy_per_m[~non_zero_r_mask] = -np.inf

    kinetic_energy_per_m = 0.5 * v_squared
    specific_energy = kinetic_energy_per_m + potential_energy_per_m
    total_energy = m * specific_energy

    return total_energy[0] if is_single_state else total_energy

# --- 角动量计算 ---
def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算z方向上的角动量 Lz（单位质量），Lz = x*vy - y*vx

    参数：
        state_vector: 状态数组 [x, y, vx, vy]
        m: 粒子质量，默认为1，计算单位质量角动量

    返回：
        角动量标量或数组
    """
    is_single_state = (state_vector.ndim == 1)
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)

    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]

    specific_Lz = x * vy - y * vx
    total_Lz = m * specific_Lz

    return total_Lz[0] if is_single_state else total_Lz

# --- 主程序：演示三种轨道 ---
if __name__ == "__main__":
    print("Demonstrating orbital simulations...")

    # 基本设置
    t_start = 0
    t_end_ellipse = 20
    t_end_hyperbola = 5
    t_end_parabola = 10
    n_points = 1000
    mass_particle = 1.0

    # --- 情况1：椭圆轨道 ---
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]
    t_eval_ellipse = np.linspace(t_start, t_end_ellipse, n_points)
    sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end_ellipse), t_eval_ellipse)
    x_ellipse, y_ellipse = sol_ellipse.y[0], sol_ellipse.y[1]
    energy_ellipse = calculate_energy(sol_ellipse.y.T, GM, mass_particle)
    Lz_ellipse = calculate_angular_momentum(sol_ellipse.y.T, mass_particle)
    print(f"Ellipse: Initial E = {energy_ellipse[0]:.3f}, Initial Lz = {Lz_ellipse[0]:.3f}")

    # --- 情况2：抛物线轨道（逃逸边界）---
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2 * GM)]
    t_eval_parabola = np.linspace(t_start, t_end_parabola, n_points)
    sol_parabola = solve_orbit(ic_parabola, (t_start, t_end_parabola), t_eval_parabola)
    x_parabola, y_parabola = sol_parabola.y[0], sol_parabola.y[1]
    energy_parabola = calculate_energy(sol_parabola.y.T, GM, mass_particle)
    print(f"Parabola: Initial E = {energy_parabola[0]:.3f}")

    # --- 情况3：双曲线轨道（超越逃逸速度）---
    ic_hyperbola = [1.0, 0.0, 0.0, 1.2 * np.sqrt(2 * GM)]
    t_eval_hyperbola = np.linspace(t_start, t_end_hyperbola, n_points)
    sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end_hyperbola), t_eval_hyperbola)
    x_hyperbola, y_hyperbola = sol_hyperbola.y[0], sol_hyperbola.y[1]
    energy_hyperbola = calculate_energy(sol_hyperbola.y.T, GM, mass_particle)
    print(f"Hyperbola: Initial E = {energy_hyperbola[0]:.3f}")

    # --- 画图 ---
    plt.figure(figsize=(10, 8))
    plt.plot(x_ellipse, y_ellipse, label=f'Elliptical (E={energy_ellipse[0]:.2f})')
    plt.plot(x_parabola, y_parabola, label=f'Parabolic (E={energy_parabola[0]:.2f})')
    plt.plot(x_hyperbola, y_hyperbola, label=f'Hyperbolic (E={energy_hyperbola[0]:.2f})')
    plt.plot(0, 0, 'ko', markersize=10, label='Central Body')
    plt.title('Orbits in an Inverse-Square Law Gravitational Field')
    plt.xlabel('x (arbitrary units)')
    plt.ylabel('y (arbitrary units)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # --- 额外演示：固定能量，改变发射角以改变角动量 ---
    print("\nDemonstrating varying angular momentum for E < 0...")
    E_target = -0.2
    r0 = 1.5

    if E_target + GM * mass_particle / r0 < 0:
        print("Error: 不可能以该能量达到初始位置 r0")
    else:
        v0 = np.sqrt(2 / mass_particle * (E_target + GM * mass_particle / r0))
        print(f"给定能量 {E_target} 对应速度 v0 = {v0:.3f}")

        plt.figure(figsize=(10, 8))
        plt.plot(0, 0, 'ko', markersize=10)

        # 不同角度对应不同角动量
        for angle_deg in [90, 60, 45]:
            angle_rad = np.deg2rad(angle_deg)
            vx0 = v0 * np.cos(angle_rad)
            vy0 = v0 * np.sin(angle_rad)
            ic = [r0, 0, vx0, vy0]
            current_E = calculate_energy(np.array(ic), GM, mass_particle)
            current_Lz = calculate_angular_momentum(np.array(ic), mass_particle)
            print(f"  Angle {angle_deg}°: E={current_E:.3f}, Lz={current_Lz:.3f}")
            sol = solve_orbit(ic, (t_start, t_end_ellipse * 1.5),
                              np.linspace(t_start, t_end_ellipse * 1.5, n_points), gm_val=GM)
            plt.plot(sol.y[0], sol.y[1], label=f'Lz={current_Lz:.2f} (Angle {angle_deg}°)')

        plt.title(f'Orbits with Fixed Energy E = {E_target}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

