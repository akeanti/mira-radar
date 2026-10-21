"""MIRA: a complete MIMO FMCW radar, from chirp to track, in NumPy."""

from .config import PRESETS, RadarConfig, get_preset
from .pipeline import FrameResult, PipelineSettings, RadarPipeline
from .scene import Car, Cyclist, Drone, Maneuver, Pedestrian, Reflector, Scene, Static
from .simulator import FMCWSimulator

__version__ = "0.1.0"

__all__ = [
    "PRESETS", "RadarConfig", "get_preset",
    "FrameResult", "PipelineSettings", "RadarPipeline",
    "Car", "Cyclist", "Drone", "Maneuver", "Pedestrian", "Reflector", "Scene", "Static",
    "FMCWSimulator",
]
