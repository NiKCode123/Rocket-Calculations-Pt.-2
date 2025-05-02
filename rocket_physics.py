import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

# Physical constants
GRAVITY = 9.81  # m/s²
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
ATMOSPHERIC_SCALE_HEIGHT = 8500  # m, scale height for simple exponential atmosphere model

def calculate_air_density(altitude):
    """Calculate air density based on altitude using an exponential model"""
    if altitude < 0:
        return AIR_DENSITY_SEA_LEVEL
    return AIR_DENSITY_SEA_LEVEL * np.exp(-altitude / ATMOSPHERIC_SCALE_HEIGHT)

def rocket_dynamics(t, state, params):
    """
    Differential equations describing rocket motion
    
    Parameters:
    -----------
    t : float
        Current time
    state : array
        [x, y, vx, vy, mass]
        x: horizontal position (m)
        y: vertical position (m)
        vx: horizontal velocity (m/s)
        vy: vertical velocity (m/s)
        mass: current mass (kg)
    params : dict
        Dictionary of rocket parameters
        
    Returns:
    --------
    Array of derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt, dmass/dt]
    """
    # Unpack state
    x, y, vx, vy, mass = state
    
    # Unpack parameters
    thrust = params['thrust']
    burn_time = params['burn_time']
    drag_coefficient = params['drag_coefficient']
    frontal_area = params['frontal_area']
    launch_angle_rad = np.radians(params['launch_angle'])
    fuel_mass = params['fuel_mass']
    rocket_mass = params['rocket_mass']
    
    # Compute current speed
    velocity = np.sqrt(vx**2 + vy**2)
    
    # Initialize acceleration components
    ax = 0
    ay = -GRAVITY  # Gravity always acts downward
    
    # Check if we're still burning fuel
    is_burning = t <= burn_time
    
    # Compute mass derivative (fuel consumption rate)
    if is_burning and mass > rocket_mass:
        fuel_consumption_rate = fuel_mass / burn_time
        dmass_dt = -fuel_consumption_rate
    else:
        dmass_dt = 0
        
    # Apply thrust if engines are still burning
    if is_burning and mass > rocket_mass:
        # Thrust direction based on launch angle
        thrust_x = thrust * np.cos(launch_angle_rad)
        thrust_y = thrust * np.sin(launch_angle_rad)
        
        # Add thrust acceleration (F = ma → a = F/m)
        ax += thrust_x / mass
        ay += thrust_y / mass
    
    # Apply drag if rocket is moving
    if velocity > 0 and y >= 0:  # Only apply drag when in atmosphere
        # Calculate air density at current altitude
        air_density = calculate_air_density(y)
        
        # Calculate drag force magnitude (F_drag = 0.5 * rho * Cd * A * v²)
        drag_force = 0.5 * air_density * drag_coefficient * frontal_area * velocity**2
        
        # Compute the unit vector in the direction of velocity
        vx_unit = vx / velocity
        vy_unit = vy / velocity
        
        # Apply drag in the opposite direction of velocity
        drag_ax = -drag_force * vx_unit / mass
        drag_ay = -drag_force * vy_unit / mass
        
        ax += drag_ax
        ay += drag_ay
    
    # Return derivatives
    return [vx, vy, ax, ay, dmass_dt]

def calculate_trajectory(params):
    """
    Calculate rocket trajectory based on input parameters
    
    Parameters:
    -----------
    params : dict
        Dictionary of rocket parameters
        
    Returns:
    --------
    trajectory_data : dict
        Dictionary containing trajectory data arrays
    flight_metrics : dict
        Dictionary of flight performance metrics
    """
    # Set up initial conditions
    total_mass = params['rocket_mass'] + params['fuel_mass']
    launch_angle_rad = np.radians(params['launch_angle'])
    
    # Initial state: [x, y, vx, vy, mass]
    initial_state = [
        0,  # x = 0 (starting at origin)
        0,  # y = 0 (starting at ground level)
        0,  # vx = 0 (starting from rest)
        0,  # vy = 0 (starting from rest)
        total_mass  # initial mass
    ]
    
    # Time spans for simulation
    t_span = (0, params['simulation_time'])
    t_eval = np.arange(0, params['simulation_time'], params['time_step'])
    
    # Solve the differential equations
    solution = solve_ivp(
        fun=lambda t, y: rocket_dynamics(t, y, params),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # Extract solution components
    times = solution.t
    x_positions = solution.y[0]
    y_positions = solution.y[1]
    x_velocities = solution.y[2]
    y_velocities = solution.y[3]
    masses = solution.y[4]
    
    # Calculate derived quantities
    velocities = np.sqrt(x_velocities**2 + y_velocities**2)
    
    # Find index where rocket hits the ground (if it does)
    ground_impact_idx = None
    for i in range(1, len(y_positions)):
        if y_positions[i] <= 0 and y_positions[i-1] > 0:
            ground_impact_idx = i
            break
    
    # If rocket hits the ground, data at that point
    if ground_impact_idx is not None:
        times = times[:ground_impact_idx+1]
        x_positions = x_positions[:ground_impact_idx+1]
        y_positions = y_positions[:ground_impact_idx+1]
        velocities = velocities[:ground_impact_idx+1]
        masses = masses[:ground_impact_idx+1]
    
    # Calculate flight metrics
    max_altitude = np.max(y_positions)
    max_altitude_time = times[np.argmax(y_positions)]
    max_velocity = np.max(velocities)
    max_velocity_time = times[np.argmax(velocities)]
    
    if ground_impact_idx is not None:
        total_flight_time = times[ground_impact_idx]
        total_range = x_positions[ground_impact_idx]
    else:
        total_flight_time = times[-1]
        total_range = x_positions[-1]
    
    # Assemble trajectory data
    trajectory_data = {
        'time': times,
        'horizontal_distance': x_positions,
        'altitude': y_positions,
        'velocity': velocities,
        'mass': masses
    }
    
    # Assemble flight metrics
    flight_metrics = {
        'Maximum Altitude': max_altitude,
        'Time to Apogee': max_altitude_time,
        'Maximum Velocity': max_velocity,
        'Time to Max Velocity': max_velocity_time,
        'Total Flight Time': total_flight_time,
        'Total Range': total_range,
        'Launch Angle': params['launch_angle']
    }
    
    return trajectory_data, flight_metrics
