"""
micro_clusters.py
Defines the Core, Potential, and Outlier Micro-Clusters
"""
import numpy as np
from enum import Enum

class MicroClusterType(Enum):
    # diferent types of the clusters for the pruning phase
    POTENTIAL = "potential"
    OUTLIER = "outlier"
    CORE = "core" # might not need this

class MicroCluster:
    def __init__(self, mc_id: int, t_0: int, mc_type: MicroClusterType, point: np.ndarray):
        """
        Initializes the CF tuple (w, CF1, CF2) and timestamp.
        """
        self.mc_id = mc_id        # unique ID for th mircro cluster
        self.t_0 = t_0            # creation time (used for OMC pruning function Xi)
        self.last_update = t_0    # here we track the last time this MC was updated/decayed
        self.mc_type = mc_type
        
        # the Cluster Feature (CF) tuple
        self.weight = 0.0
        # initialize CF1 and CF2 as vectors matching the dimensionality of 'point'
        self.CF1 = np.zeros_like(point) if point is not None else 0.0
        self.CF2 = np.zeros_like(point) if point is not None else 0.0
        
        # if initialized with a point, immediately add it
        if point is not None:
            self.insert_sample(point, 1.0) # weight is 1.0 before decay

    # we need to implement these methods / consider adding more helpers as you see fit
    def insert_sample(self, point: np.ndarray, weight: float):
        """
        TODO: Add a new point to the CF tuple
        Formula: w = w + weight, CF1 = CF1 + point, CF2 = CF2 + (point^2)
        """
        pass

    def apply_decay(self, decay_factor: float, time_diff: int):
        """
        TODO: Apply the exponential fading function to the CF tuple.
        formula: f(t) = 2^(-lambda * dt)
        multiply weight, CF1, and CF2 by this factor, then update self.last_update.
        """
        pass

    def get_center(self) -> np.ndarray:
        """Calculates and returns the center c = CF1 / w"""
        pass

    def get_radius(self) -> float:
        """Calculates and returns the radius r = sqrt((CF2 / w) - (CF1 / w)^2)"""
        pass
    
    def copy(self) -> 'MicroCluster':

        pass