def get_default_parameters():
    """
    Return default parameters for rocket simulation
    
    Returns:
    --------
    dict
        Dictionary of default parameters
    """
    return {
        'rocket_mass': 50.0,      # kg
        'fuel_mass': 100.0,       # kg
        'thrust': 5000.0,         # N
        'burn_time': 10.0,        # s
        'drag_coefficient': 0.3,  # dimensionless
        'frontal_area': 0.1,      # m²
        'launch_angle': 85.0,     # degrees
        'simulation_time': 120,   # s
        'time_step': 0.1          # s
    }

def validate_inputs(params):
    """
    Validate user inputs for rocket simulation
    
    Parameters:
    -----------
    params : dict
        Dictionary of rocket parameters
        
    Returns:
    --------
    tuple
        (is_valid, message) where is_valid is a boolean and message explains any error
    """
    # Check for positive values where required
    for param_name, param_value in params.items():
        if param_name in ['rocket_mass', 'fuel_mass', 'thrust', 'burn_time', 
                          'frontal_area', 'simulation_time', 'time_step']:
            if param_value <= 0:
                return False, f"{param_name} must be positive"
    
    # Check for reasonable range on drag coefficient
    if params['drag_coefficient'] <= 0 or params['drag_coefficient'] > 2.0:
        return False, "Drag coefficient should be between 0 and 2.0"
    
    # Check launch angle is between 45 and 90 degrees
    if params['launch_angle'] < 45 or params['launch_angle'] > 90:
        return False, "Launch angle should be between 45 and 90 degrees"
        
    # Check time step is reasonable for simulation stability
    if params['time_step'] > 1.0:
        return False, "Time step should be less than 1.0 seconds for accuracy"
    
    # Check simulation time is long enough
    if params['simulation_time'] < params['burn_time']:
        return False, "Simulation time should be longer than burn time"
    
    # Check thrust-to-weight ratio
    total_mass = params['rocket_mass'] + params['fuel_mass']
    thrust_to_weight = params['thrust'] / (total_mass * 9.81)
    if thrust_to_weight < 1.0:
        return False, f"Thrust-to-weight ratio ({thrust_to_weight:.2f}) is less than 1.0. Rocket won't lift off."
    
    return True, "All inputs are valid"
