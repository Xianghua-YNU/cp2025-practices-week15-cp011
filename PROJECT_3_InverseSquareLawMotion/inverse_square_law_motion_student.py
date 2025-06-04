"""
Module: InverseSquareLawMotion Solution
File: inverse_square_law_motion_solution.py
Author: Trae AI
Date: $(date +%Y-%m-%d)

Solves for the motion of a particle in an inverse-square law central force field.
This typically describes orbital motion, like a planet around a star.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants (can be adjusted or passed as parameters)
GM = 1.0  # Gravitational constant * Mass of central body (e.g., G*M_sun)
# For simplicity, we can also assume the mass of the orbiting particle m=1.

def derivatives(t, state_vector, gm_val):
    """
    Computes the derivatives for the state vector [x, y, vx, vy].

    The equations of motion in Cartesian coordinates are:
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -GM * x / r^3
    dvy/dt = -GM * y / r^3
    where r = sqrt(x^2 + y^2).

    Args:
        t (float): Current time (not directly used in this autonomous system but required by solve_ivp).
        state_vector (np.ndarray): A 1D array [x, y, vx, vy] representing the current state.
        gm_val (float): The product of gravitational constant G and central mass M.

    Returns:
        np.ndarray: A 1D array of the derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt].
    """
    x, y, vx, vy = state_vector
    r_cubed = (x**2 + y**2)**1.5
    
    # Handle potential division by zero if r is very small (though physically unlikely for bound orbits starting outside r=0)
    if r_cubed < 1e-12: # A small threshold to prevent division by zero
        # This case should ideally not be reached if initial conditions are sensible.
        # For a direct hit or very close approach, the model might break down or need regularization.
        # Returning zeros or very large accelerations might be options, or raising an error.
        # For now, let's assume it implies a very strong force towards the origin if r is tiny.
        # However, a robust solution might involve stopping integration or special handling.
        # For typical orbital simulations, r should not become this small.
        ax = -gm_val * x / (1e-12) if x != 0 else 0
        ay = -gm_val * y / (1e-12) if y != 0 else 0
        return [vx, vy, ax, ay]

    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    return [vx, vy, ax, ay]

def solve_orbit(initial_conditions, t_span, t_eval, gm_val=GM):
    """
    Solves the orbital motion problem using scipy.integrate.solve_ivp.

    Args:
        initial_conditions (list or np.ndarray): [x0, y0, vx0, vy0] at t_start.
        t_span (tuple): (t_start, t_end), the interval of integration.
        t_eval (np.ndarray): Array of time points at which to store the solution.
        gm_val (float, optional): GM value. Defaults to the global GM.

    Returns:
        scipy.integrate.OdeSolution: The solution object from solve_ivp.
                                     Access solution at t_eval via sol.y (transpose for (N_points, N_vars)).
                                     sol.t contains the time points.
    """
    sol = solve_ivp(
        fun=derivatives, 
        t_span=t_span, 
        y0=initial_conditions, 
        t_eval=t_eval, 
        args=(gm_val,),
        method='RK45',  # Explicit Runge-Kutta method of order 5(4)
        rtol=1e-7,      # Relative tolerance
        atol=1e-9       # Absolute tolerance
    )
    return sol

def calculate_energy(state_vector, gm_val=GM, m=1.0):
    """
    Calculates the specific mechanical energy (energy per unit mass) of the particle.
    E/m = 0.5 * v^2 - GM/r

    Args:
        state_vector (np.ndarray): A 2D array where each row is [x, y, vx, vy] or a 1D array for a single state.
        gm_val (float, optional): GM value. Defaults to the global GM.
        m (float, optional): Mass of the orbiting particle. Defaults to 1.0 for specific energy.

    Returns:
        np.ndarray or float: Specific mechanical energy (or total energy if m is not 1).
    """
    is_single_state = state_vector.ndim == 1
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)

    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]

    r = np.sqrt(x**2 + y**2)
    v_squared = vx**2 + vy**2
    
    # Avoid division by zero for r
    # If r is zero, potential energy is undefined (infinite). This should not happen in a valid orbit.
    potential_energy_per_m = np.zeros_like(r)
    non_zero_r_mask = r > 1e-12
    potential_energy_per_m[non_zero_r_mask] = -gm_val / r[non_zero_r_mask]
    # For r=0, it's a singularity. We might assign NaN or raise error, or let it be if m=0 for PE.
    # Here, if r is effectively zero, PE term will be -inf if not handled.
    # For plotting or analysis, such points should be flagged.
    if np.any(~non_zero_r_mask):
        print("Warning: r=0 encountered in energy calculation. Potential energy is singular.")
        potential_energy_per_m[~non_zero_r_mask] = -np.inf # Or some other indicator

    kinetic_energy_per_m = 0.5 * v_squared
    specific_energy = kinetic_energy_per_m + potential_energy_per_m
    
    total_energy = m * specific_energy

    return total_energy[0] if is_single_state else total_energy

def calculate_angular_momentum(state_vector, m=1.0):
    """
    Calculates the specific angular momentum (z-component) of the particle.
    Lz/m = x*vy - y*vx

    Args:
        state_vector (np.ndarray): A 2D array where each row is [x, y, vx, vy] or a 1D array for a single state.
        m (float, optional): Mass of the orbiting particle. Defaults to 1.0 for specific angular momentum.

    Returns:
        np.ndarray or float: Specific angular momentum (or total Lz if m is not 1).
    """
    is_single_state = state_vector.ndim == 1
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)
        
    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]

    specific_Lz = x * vy - y * vx
    total_Lz = m * specific_Lz
    
    return total_Lz[0] if is_single_state else total_Lz


if __name__ == "__main__":
    # --- Demonstration of usage ---
    print("Demonstrating orbital simulations...")

    # Common parameters
    t_start = 0
    t_end_ellipse = 20  # Enough time for a few orbits for typical elliptical case
    t_end_hyperbola = 5 # Hyperbola moves away quickly
    t_end_parabola = 10 # Parabola also moves away
    n_points = 1000
    mass_particle = 1.0 # Assume m=1 for simplicity in E and L calculations

    # Case 1: Elliptical Orbit (E < 0)
    # Initial conditions: x0=1, y0=0, vx0=0, vy0=0.8 (adjust vy0 for different eccentricities)
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]
    t_eval_ellipse = np.linspace(t_start, t_end_ellipse, n_points)
    sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end_ellipse), t_eval_ellipse, gm_val=GM)
    x_ellipse, y_ellipse = sol_ellipse.y[0], sol_ellipse.y[1]
    energy_ellipse = calculate_energy(sol_ellipse.y.T, GM, mass_particle)
    Lz_ellipse = calculate_angular_momentum(sol_ellipse.y.T, mass_particle)
    print(f"Ellipse: Initial E = {energy_ellipse[0]:.3f}, Initial Lz = {Lz_ellipse[0]:.3f}")
    print(f"Ellipse: Final E = {energy_ellipse[-1]:.3f}, Final Lz = {Lz_ellipse[-1]:.3f} (Energy/Ang. Mom. Conservation Check)")

    # Case 2: Parabolic Orbit (E = 0)
    # For E=0, v_escape = sqrt(2*GM/r). If x0=1, y0=0, then vy0 = sqrt(2*GM/1) = sqrt(2)
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2*GM)]
    t_eval_parabola = np.linspace(t_start, t_end_parabola, n_points)
    sol_parabola = solve_orbit(ic_parabola, (t_start, t_end_parabola), t_eval_parabola, gm_val=GM)
    x_parabola, y_parabola = sol_parabola.y[0], sol_parabola.y[1]
    energy_parabola = calculate_energy(sol_parabola.y.T, GM, mass_particle)
    print(f"Parabola: Initial E = {energy_parabola[0]:.3f}")

    # Case 3: Hyperbolic Orbit (E > 0)
    # If vy0 > v_escape, e.g., vy0 = 1.5 * sqrt(2*GM)
    ic_hyperbola = [1.0, 0.0, 0.0, 1.2 * np.sqrt(2*GM)] # Speed greater than escape velocity
    t_eval_hyperbola = np.linspace(t_start, t_end_hyperbola, n_points)
    sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end_hyperbola), t_eval_hyperbola, gm_val=GM)
    x_hyperbola, y_hyperbola = sol_hyperbola.y[0], sol_hyperbola.y[1]
    energy_hyperbola = calculate_energy(sol_hyperbola.y.T, GM, mass_particle)
    print(f"Hyperbola: Initial E = {energy_hyperbola[0]:.3f}")

    # Plotting the orbits
    plt.figure(figsize=(10, 8))
    plt.plot(x_ellipse, y_ellipse, label=f'Elliptical (E={energy_ellipse[0]:.2f})')
    plt.plot(x_parabola, y_parabola, label=f'Parabolic (E={energy_parabola[0]:.2f})')
    plt.plot(x_hyperbola, y_hyperbola, label=f'Hyperbolic (E={energy_hyperbola[0]:.2f})')
    plt.plot(0, 0, 'ko', markersize=10, label='Central Body (Sun)') # Central body
    plt.title('Orbits in an Inverse-Square Law Gravitational Field')
    plt.xlabel('x (arbitrary units)')
    plt.ylabel('y (arbitrary units)')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal') # Crucial for correct aspect ratio of orbits
    plt.show()

    # --- Demonstration for Task 3: Varying Angular Momentum for E < 0 ---
    print("\nDemonstrating varying angular momentum for E < 0...")
    E_target = -0.2 # Target negative energy (must be < 0 for ellipse)
    r0 = 1.5       # Initial distance from center (on x-axis)
    # E = 0.5*m*v0y^2 - GM*m/r0  => v0y = sqrt(2/m * (E_target + GM*m/r0))
    # Ensure (E_target + GM*m/r0) is positive for real v0y
    if E_target + GM * mass_particle / r0 < 0:
        print(f"Error: Cannot achieve E_target={E_target} at r0={r0}. E_target must be > -GM*m/r0.")
        print(f"Required E_target > {-GM*mass_particle/r0}")
    else:
        vy_base = np.sqrt(2/mass_particle * (E_target + GM * mass_particle / r0))
        
        initial_conditions_L = []
        # Lz = m * r0 * vy0. We vary vy0 slightly around vy_base to change Lz while trying to keep E close to E_target.
        # Note: Strictly keeping E constant while varying L means r0 or speed direction must change.
        # Here, we fix r0 and initial velocity direction (along y), so varying vy0 changes both E and L.
        # A more precise way for Task 3 would be to fix E and r_periapsis, then find v_periapsis for different L, 
        # or fix E and vary the launch angle from a fixed r0.
        # For simplicity in this demo, we'll vary vy0, which will slightly alter E too.
        # The project description implies fixing E and varying L. Let's try to achieve that more directly.
        # For a fixed E (<0) and r0, the speed v0 is fixed: v0 = sqrt(2/m * (E + GMm/r0)).
        # We can then vary the angle of v0 to change L = m*r0*v0*sin(alpha), where alpha is angle between r0_vec and v0_vec.
        # Let initial position be (r0, 0). Initial velocity (v0*cos(theta), v0*sin(theta)).
        # Lz = m * (x0*vy0 - y0*vx0) = m * r0 * v0*sin(theta). Energy E = 0.5*m*v0^2 - GMm/r0.
        
        v0_for_E_target = np.sqrt(2/mass_particle * (E_target + GM*mass_particle/r0))
        print(f"For E_target={E_target} at r0={r0}, required speed v0={v0_for_E_target:.3f}")

        plt.figure(figsize=(10, 8))
        plt.plot(0, 0, 'ko', markersize=10, label='Central Body')

        # Launch angles (theta) to vary Lz, keeping v0 (and thus E) constant
        launch_angles_deg = [90, 60, 45] # Degrees from positive x-axis for velocity vector
        
        for i, angle_deg in enumerate(launch_angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            vx0 = v0_for_E_target * np.cos(angle_rad)
            vy0 = v0_for_E_target * np.sin(angle_rad)
            ic = [r0, 0, vx0, vy0]
            
            current_E = calculate_energy(np.array(ic), GM, mass_particle)
            current_Lz = calculate_angular_momentum(np.array(ic), mass_particle)
            print(f"  Angle {angle_deg}deg: Calculated E={current_E:.3f} (Target E={E_target:.3f}), Lz={current_Lz:.3f}")

            sol = solve_orbit(ic, (t_start, t_end_ellipse*1.5), np.linspace(t_start, t_end_ellipse*1.5, n_points), gm_val=GM)
            plt.plot(sol.y[0], sol.y[1], label=f'Lz={current_Lz:.2f} (Launch Angle {angle_deg}°)')

        plt.title(f'Elliptical Orbits with Fixed Energy (E ≈ {E_target:.2f}) and Varying Angular Momentum')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.show()
