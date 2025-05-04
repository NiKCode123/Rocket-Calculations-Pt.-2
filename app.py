import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from rocket_physics import calculate_trajectory
from utils import validate_inputs, get_default_parameters
from db_utils import save_simulation, get_all_simulations, get_simulation_by_id, delete_simulation

# Set page configuration
st.set_page_config(
    page_title="Rocket Trajectory Calculator",
    page_icon="🚀",
    layout="wide"
)

# Main title
st.title("🚀 Rocket Trajectory Simulator")
st.write("Calculate and visualize rocket flight trajectories based on physical parameters")

# Sidebar for inputs
st.sidebar.header("Rocket Parameters")

# Get default parameters
default_params = get_default_parameters()

# Create input form
with st.sidebar.form("rocket_parameters_form"):
    # Rocket physical parameters
    rocket_mass = st.number_input(
        "Dry Mass (kg)", 
        min_value=1.0, 
        value=default_params['rocket_mass'],
        help="The mass of the rocket without fuel"
    )
    
    fuel_mass = st.number_input(
        "Fuel Mass (kg)", 
        min_value=0.1, 
        value=default_params['fuel_mass'],
        help="Total mass of the fuel"
    )
    
    thrust = st.number_input(
        "Thrust (N)", 
        min_value=10.0, 
        value=default_params['thrust'],
        help="The thrust force produced by the rocket engine"
    )
    
    burn_time = st.number_input(
        "Burn Time (s)", 
        min_value=0.1, 
        value=default_params['burn_time'],
        help="How long the rocket engine fires"
    )
    
    drag_coefficient = st.number_input(
        "Drag Coefficient", 
        min_value=0.01, 
        max_value=2.0, 
        value=default_params['drag_coefficient'],
        help="Coefficient of aerodynamic drag (typical range: 0.1-1.0)"
    )
    
    frontal_area = st.number_input(
        "Frontal Area (m²)", 
        min_value=0.001, 
        value=default_params['frontal_area'],
        help="Cross-sectional area of the rocket"
    )
    
    launch_angle = st.slider(
        "Launch Angle (degrees)", 
        min_value=45.0, 
        max_value=90.0, 
        value=default_params['launch_angle'],
        help="Angle from horizontal (90° is vertical)"
    )
    
    simulation_time = st.number_input(
        "Simulation Time (s)", 
        min_value=10, 
        max_value=1000, 
        value=default_params['simulation_time'],
        help="Total time to simulate the trajectory"
    )
    
    time_step = st.number_input(
        "Time Step (s)", 
        min_value=0.01, 
        max_value=1.0, 
        value=default_params['time_step'],
        help="Smaller values are more accurate but slower"
    )
    
    submit_button = st.form_submit_button("Calculate Trajectory")

# Create main content
if submit_button or 'trajectory_data' in st.session_state:
    # Collect all parameters
    params = {
        'rocket_mass': rocket_mass,
        'fuel_mass': fuel_mass,
        'thrust': thrust,
        'burn_time': burn_time,
        'drag_coefficient': drag_coefficient,
        'frontal_area': frontal_area,
        'launch_angle': launch_angle,
        'simulation_time': simulation_time,
        'time_step': time_step
    }
    
    # Validate inputs
    validation_result, message = validate_inputs(params)
    
    if not validation_result:
        st.error(f"Invalid inputs: {message}")
    else:
        # Calculate trajectory
        with st.spinner("Calculating trajectory..."):
            trajectory_data, flight_metrics = calculate_trajectory(params)
            st.session_state.trajectory_data = trajectory_data
            st.session_state.flight_metrics = flight_metrics
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Trajectory Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the trajectory
            ax.plot(trajectory_data['horizontal_distance'], trajectory_data['altitude'], 
                    color='blue', linewidth=2)
            
            # Add fuel burnout point
            burnout_idx = int(burn_time / time_step)
            if burnout_idx < len(trajectory_data['horizontal_distance']):
                ax.scatter(
                    trajectory_data['horizontal_distance'][burnout_idx],
                    trajectory_data['altitude'][burnout_idx],
                    color='red', s=100, zorder=5, label='Fuel Burnout'
                )
            
            # Add apogee point
            max_altitude_idx = np.argmax(trajectory_data['altitude'])
            ax.scatter(
                trajectory_data['horizontal_distance'][max_altitude_idx],
                trajectory_data['altitude'][max_altitude_idx],
                color='green', s=100, zorder=5, label='Apogee'
            )
            
            # Add labels and title
            ax.set_xlabel('Horizontal Distance (m)')
            ax.set_ylabel('Altitude (m)')
            ax.set_title('Rocket Trajectory')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set equal aspect ratio and create some buffer space
            ax.set_aspect('equal', adjustable='box')
            
            # Show plot in Streamlit
            st.pyplot(fig)
            
            # Display trajectory data as an interactive table
            st.subheader("Trajectory Data")
            # Create a smaller version of the dataframe for display
            display_df = pd.DataFrame({
                'Time (s)': trajectory_data['time'],
                'Altitude (m)': trajectory_data['altitude'],
                'Distance (m)': trajectory_data['horizontal_distance'],
                'Velocity (m/s)': trajectory_data['velocity']
            })
            # Sample the dataframe if it's too large
            if len(display_df) > 100:
                display_df = display_df.iloc[::len(display_df)//100].reset_index(drop=True)
            
            st.dataframe(display_df)
            
        with col2:
            st.subheader("Flight Metrics")
            
            # Create a nice display for metrics
            metrics_df = pd.DataFrame({
                'Metric': list(flight_metrics.keys()),
                'Value': list(flight_metrics.values())
            })
            
            for i, row in metrics_df.iterrows():
                if "Time" in row['Metric']:
                    value = f"{row['Value']:.2f} seconds"
                elif "Angle" in row['Metric']:
                    value = f"{row['Value']:.1f}°"
                else:
                    value = f"{row['Value']:.2f} meters"
                
                st.metric(
                    label=row['Metric'],
                    value=value
                )
            
            # Display a secondary plot - velocity over time
            st.subheader("Velocity Profile")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(trajectory_data['time'], trajectory_data['velocity'], 
                     color='purple', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig2)
            
            # Save simulation section
            st.subheader("Save Simulation")
            with st.form(key="save_simulation_form"):
                sim_name = st.text_input("Simulation Name", value=f"Simulation {launch_angle}° - {thrust}N")
                sim_description = st.text_area("Description (optional)", height=100)
                save_button = st.form_submit_button("Save to Database")
                
                if save_button:
                    try:
                        sim_id = save_simulation(
                            name=sim_name,
                            description=sim_description,
                            params=params,
                            trajectory_data=trajectory_data,
                            flight_metrics=flight_metrics
                        )
                        st.success(f"Simulation saved successfully with ID: {sim_id}")
                    except Exception as e:
                        st.error(f"Error saving simulation: {str(e)}")

# Add a tab for viewing saved simulations
st.header("Saved Simulations")
all_sims = get_all_simulations()

if not all_sims:
    st.info("No saved simulations yet. Run a simulation and save it to see it here.")
else:
    # Create a dataframe for display
    sims_df = pd.DataFrame([
        {
            "ID": sim.id,
            "Name": sim.name,
            "Date": sim.created_at.strftime("%Y-%m-%d %H:%M"),
            "Max Altitude (m)": f"{sim.max_altitude:.2f}",
            "Max Velocity (m/s)": f"{sim.max_velocity:.2f}",
            "Flight Time (s)": f"{sim.total_flight_time:.2f}",
            "Range (m)": f"{sim.total_range:.2f}"
        }
        for sim in all_sims
    ])
    
    st.dataframe(sims_df)
    
    # Allow loading a simulation
    col1, col2 = st.columns(2)
    
    with col1:
        selected_sim_id = st.selectbox(
            "Select a simulation to load",
            options=[sim.id for sim in all_sims],
            format_func=lambda x: next((s.name for s in all_sims if s.id == x), "")
        )
    
    with col2:
        load_col, delete_col = st.columns(2)
        with load_col:
            if st.button("Load Simulation"):
                sim = get_simulation_by_id(selected_sim_id)
                if sim:
                    # Load parameters
                    loaded_params = json.loads(sim.parameters)
                    st.session_state.loaded_params = loaded_params
                    
                    # Recalculate trajectory with these parameters
                    trajectory_data, flight_metrics = calculate_trajectory(loaded_params)
                    st.session_state.trajectory_data = trajectory_data
                    st.session_state.flight_metrics = flight_metrics
                    
                    st.success(f"Loaded simulation: {sim.name}")
                    st.rerun()
        
        with delete_col:
            if st.button("Delete Simulation", type="primary", use_container_width=True):
                if delete_simulation(selected_sim_id):
                    st.success(f"Simulation {selected_sim_id} deleted successfully")
                    st.rerun()
                else:
                    st.error("Failed to delete simulation")

# Add explanatory information at the bottom
with st.expander("How it works"):
    st.markdown("""
    ## Rocket Physics Simulation
    
    This application uses fundamental principles of physics to simulate rocket flight:
    
    1. **Thrust Force**: Provided by the rocket engine, calculated as constant during burn time
    2. **Weight Force**: Due to gravity, calculated as mass × gravitational acceleration
    3. **Drag Force**: Air resistance, calculated as 0.5 × air density × drag coefficient × area × velocity²
    4. **Acceleration**: Determined by the sum of forces divided by the current mass
    5. **Motion**: Position and velocity updated using numerical integration
    
    ### Assumptions:
    - Earth's gravity is constant at 9.81 m/s²
    - Simplified atmospheric model with exponential density decrease
    - No wind or weather effects
    - No stabilization or control systems
    """)

with st.expander("About"):
    st.markdown("""
    # Rocket Trajectory Simulator
    
    This application was created to help visualize and understand rocket flight dynamics.
    It's intended for educational purposes and rocket enthusiasts.
    
    The simulator uses basic Newtonian physics and a simplified atmospheric model.
                
    Developed by Nikash Nadgir, January to May 2025
    """)
