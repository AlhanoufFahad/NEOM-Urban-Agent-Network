"""Predefined reproducible test scenarios."""
from .model import CitySimulation, Config

def charging_conflict():
    return CitySimulation("charging", Config(ev_arrivals=0, randomness=0))

def busy_road_waste():
    return CitySimulation("waste", Config(ev_arrivals=0, bin_fill_rate=0, randomness=0))

def energy_failure():
    return CitySimulation("failure", Config(ev_arrivals=3, randomness=0, max_steps=28))
