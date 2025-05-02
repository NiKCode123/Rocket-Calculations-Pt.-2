import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Get database URL from environment variable
DATABASE_URL = os.environ['DATABASE_URL']

# Create database engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define model for storing simulation results
class SimulationResult(Base):
    __tablename__ = 'simulation_results'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    description = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Store parameters as JSON
    parameters = Column(Text)
    
    # Store key metrics
    max_altitude = Column(Float)
    max_velocity = Column(Float)
    total_flight_time = Column(Float)
    total_range = Column(Float)
    
    def __repr__(self):
        return f"<SimulationResult(name='{self.name}', max_altitude={self.max_altitude})>"

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Create session factory
Session = sessionmaker(bind=engine)

def save_simulation(name, description, params, trajectory_data, flight_metrics):
    """
    Save simulation results to database
    
    Parameters:
    -----------
    name : str
        Name for this simulation
    description : str
        Optional description
    params : dict
        Simulation parameters
    trajectory_data : dict
        Full trajectory data
    flight_metrics : dict
        Flight performance metrics
    
    Returns:
    --------
    int
        ID of the saved simulation
    """
    session = Session()
    
    try:
        # Create new simulation result
        sim_result = SimulationResult(
            name=name,
            description=description,
            parameters=json.dumps(params),
            max_altitude=flight_metrics['Maximum Altitude'],
            max_velocity=flight_metrics['Maximum Velocity'],
            total_flight_time=flight_metrics['Total Flight Time'],
            total_range=flight_metrics['Total Range']
        )
        
        # Add and commit to database
        session.add(sim_result)
        session.commit()
        
        # Return the ID of the new record
        return sim_result.id
    
    except Exception as e:
        session.rollback()
        raise e
    
    finally:
        session.close()

def get_all_simulations():
    """
    Get all saved simulations
    
    Returns:
    --------
    list
        List of simulation results
    """
    session = Session()
    
    try:
        results = session.query(SimulationResult).order_by(SimulationResult.created_at.desc()).all()
        return results
    
    finally:
        session.close()

def get_simulation_by_id(simulation_id):
    """
    Get a specific simulation by ID
    
    Parameters:
    -----------
    simulation_id : int
        ID of the simulation to retrieve
    
    Returns:
    --------
    SimulationResult
        Simulation result object or None if not found
    """
    session = Session()
    
    try:
        result = session.query(SimulationResult).filter_by(id=simulation_id).first()
        return result
    
    finally:
        session.close()

def delete_simulation(simulation_id):
    """
    Delete a simulation from the database
    
    Parameters:
    -----------
    simulation_id : int
        ID of the simulation to delete
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    session = Session()
    
    try:
        result = session.query(SimulationResult).filter_by(id=simulation_id).first()
        if result:
            session.delete(result)
            session.commit()
            return True
        return False
    
    except Exception as e:
        session.rollback()
        raise e
    
    finally:
        session.close()