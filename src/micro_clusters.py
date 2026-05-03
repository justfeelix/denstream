"""
micro_clusters.py
Defines the Core, Potential, and Outlier Micro-Clusters
"""
import numpy as np
import math
from enum import Enum

class MicroClusterType(Enum):
    # diferent types of the clusters for the pruning phase
    POTENTIAL = "potential"
    OUTLIER = "outlier"
    #CORE = "core" # might not need this

class MicroCluster:
    def __init__(self, mc_id: int, t_0: int, mc_type: MicroClusterType, point: np.ndarray):
        """
        Initializes the CF tuple (w, CF1, CF2) and timestamp
        """
        self.mc_id = mc_id        # unique ID for th mircro cluster
        self.t_0 = t_0            # creation time (used for OMC pruning function Xi)
        self.last_update = t_0    # here we track the last time this MC was updated/decayed
        self.mc_type = mc_type
        
        # the Cluster Feature (CF) tuple
        self.weight = 0.0
        # initialize CF1 and CF2 as vectors matching the dimensionality of 'point'
        # CF1 is the linear sum -> store the weighted sum of all the coordinates of the points
        # CF2 is the squared sum -> store the weighted squares
        self.CF1 = np.zeros_like(point) if point is not None else 0.0
        self.CF2 = np.zeros_like(point) if point is not None else 0.0
        
        # if initialized with a point, immediately add it
        if point is not None:
            self.insert_sample(point, 1.0) # weight is 1.0 before decay

    # we need to implement these methods / consider adding more helpers as you see fit
    def insert_sample(self, point: np.ndarray, weight: float):
        """
        Adds a new point to the CF tuple, when a point successfully falls within the radius of this micro cluster
        Formula: w = w + weight, CF1 = CF1 + point, CF2 = CF2 + (point^2)
        """
        self.weight += weight
        self.CF1 += point
        self.CF2 += np.square(point)

    def apply_decay(self, decay_factor: float, time_diff: int):
        """
        Apply the exponential fading function to the CF tuple
        formula: f(t) = 2^(-lambda * dt)
        multiply weight, CF1, and CF2 by this factor, then update self.last_update
        """
        fading_fraction = 2 ** (-decay_factor * time_diff)
        self.weight *= fading_fraction
        self.CF1 *= fading_fraction
        self.CF2 *= fading_fraction
        self.last_update = time_diff#TODO



    def get_center(self) -> np.ndarray:
        """Calculates and returns the center c = CF1 / w"""
        return self.CF1 / self.weight

    def get_radius(self) -> float:
        """Calculates and returns the radius r = sqrt((CF2 / w) - (CF1 / w)^2)
        """
        #before a point is allowed to merge into a micro cluster, the Online Phase simulates what the new radius would be. If get_radius() returns a number larger than epsilon, the merge is rejected
        return math.sqrt((self.CF2 / self.weight) - (self.CF1 / self.weight)^2)
    
    def copy(self) -> 'MicroCluster':

        pass