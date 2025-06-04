#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[马海尔吉]
学号：[20231050169]
完成日期：[20250604]
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve a boundary value problem using the shooting method.
    
    Parameters:
        x_span (tuple): Interval boundaries (x0, x1)
        boundary_conditions (tuple): Boundary values (u(x0), u(x1))
        n_points (int): Number of points in the solution grid
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        tuple: (x array, y array) containing the solution
        
    Raises:
        ValueError: If x_span or boundary_conditions are invalid
        TypeError: If input types are incorrect
    """
    # Input validation
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span must be a tuple or list of length 2")
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple or list of length 2")
    if x_span[0] >= x_span[1]:
        raise ValueError("x_span must be in increasing order")
    if not isinstance(n_points, int) or n_points <= 1:
        raise ValueError("n_points must be an integer greater than 1")
    
    x0, x1 = x_span
    u0, u1 = boundary_conditions
    
    # Define the initial slope guesses
    m_guess1 = 0.0
    m_guess2 = -1.0
    
    def objective(m):
        """Objective function to find root of (u(x1) - u1)"""
        sol = solve_ivp(ode_system_shooting, x_span, [u0, m], 
                       t_eval=np.linspace(x0, x1, n_points))
        return sol.y[0, -1] - u1
    
    # Use secant method to find the correct initial slope
    for _ in range(max_iterations):
        f1 = objective(m_guess1)
        f2 = objective(m_guess2)
        
        if abs(f1) < tolerance:
            m_final = m_guess1
            break
        if abs(f2) < tolerance:
            m_final = m_guess2
            break
            
        # Secant method update
        m_new = m_guess2 - f2 * (m_guess2 - m_guess1) / (f2 - f1)
        m_guess1, m_guess2 = m_guess2, m_new
    else:
        m_final = m_guess2  # Use the last guess if not converged
    
    # Solve with the final slope
    sol = solve_ivp(ode_system_shooting, x_span, [u0, m_final], 
                   t_eval=np.linspace(x0, x1, n_points))
    
    return sol.t, sol.y[0]


def ode_system_shooting(t, y):
    """
    Convert the second-order ODE to a system of first-order ODEs for shooting method.
    
    Parameters:
        t (float): Independent variable (x)
        y (array): Dependent variables [u, u']
        
    Returns:
        array: Derivatives [u', u'']
    """
    # Ensure y is always treated as an array
    y = np.asarray(y)
    u, up = y
    upp = -np.pi * (u + 1) / 4
    return np.array([up, upp])


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve the boundary value problem using scipy.integrate.solve_bvp.
    
    Parameters:
        x_span (tuple): Interval boundaries (x0, x1)
        boundary_conditions (tuple): Boundary values (u(x0), u(x1))
        n_points (int): Number of points in the solution grid
        
    Returns:
        tuple: (x array, y array) containing the solution
    """
    # Input validation
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span must be a tuple or list of length 2")
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple or list of length 2")
    if x_span[0] >= x_span[1]:
        raise ValueError("x_span must be in increasing order")
    if not isinstance(n_points, int) or n_points <= 1:
        raise ValueError("n_points must be an integer greater than 1")
    
    x0, x1 = x_span
    u0, u1 = boundary_conditions
    
    # Initial mesh
    x = np.linspace(x0, x1, n_points)
    
    # Initial guess (linear interpolation between boundary conditions)
    y_guess = np.zeros((2, x.size))
    y_guess[0] = np.linspace(u0, u1, n_points)
    y_guess[1] = (u1 - u0) / (x1 - x0)
    
    # Solve the BVP
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess, tol=1e-6)
    
    # Interpolate solution to a finer grid for better plotting
    x_fine = np.linspace(x0, x1, 100)
    y_fine = sol.sol(x_fine)
    
    return x_fine, y_fine[0]


def ode_system_scipy(x, y):
    """
    Convert the second-order ODE to a system of first-order ODEs for scipy.solve_bvp.
    
    Parameters:
        x (float): Independent variable
        y (array): Dependent variables [u, u']
        
    Returns:
        array: Derivatives [u', u'']
    """
    u, up = y
    upp = -np.pi * (u + 1) / 4
    return np.vstack((up, upp))


def boundary_conditions_scipy(ya, yb):
    """
    Define the boundary conditions for scipy.solve_bvp.
    
    Parameters:
        ya (array): Solution at left boundary [u(x0), u'(x0)]
        yb (array): Solution at right boundary [u(x1), u'(x1)]
        
    Returns:
        array: Boundary condition residuals
    """
    # u(x0) = 1, u(x1) = 1
    return np.array([ya[0] - 1, yb[0] - 1])


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1)):
    """
    Compare the shooting method and scipy.solve_bvp solutions and plot results.
    
    Parameters:
        x_span (tuple): Interval boundaries (x0, x1), default (0, 1)
        boundary_conditions (tuple): Boundary values (u(x0), u(x1)), default (1, 1)
        
    Returns:
        dict: Dictionary containing solution data and comparison metrics
    """
    # Solve with both methods
    x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions)
    
    # Calculate difference
    y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
    difference = y_shoot - y_scipy_interp
    max_diff = np.max(np.abs(difference))
    rms_diff = np.sqrt(np.mean(difference**2))
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Solution plot
    plt.subplot(2, 1, 1)
    plt.plot(x_shoot, y_shoot, 'b-', label='Shooting Method')
    plt.plot(x_scipy, y_scipy, 'r--', label='scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of BVP Solutions')
    plt.legend()
    plt.grid(True)
    
    # Difference plot
    plt.subplot(2, 1, 2)
    plt.plot(x_shoot, difference, 'g-')
    plt.xlabel('x')
    plt.ylabel('Difference')
    plt.title('Difference Between Solutions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Return results as dictionary
    results = {
        'x_shooting': x_shoot,
        'y_shooting': y_shoot,
        'x_scipy': x_scipy,
        'y_scipy': y_scipy,
        'max_difference': max_diff,
        'rms_difference': rms_diff
    }
    
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"RMS difference: {rms_diff:.2e}")
    
    return results


# Example usage
if __name__ == "__main__":
    results = compare_methods_and_plot()
