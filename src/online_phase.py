"""
Online phase of DenStream algorithm
"""

import numpy as np
from enum import Enum
import math
from baseframework.stream_clustering import OnlinePhase_StreamClusterer
from baseframework.utils import distance_calc

from .micro_clusters import MicroCluster, MicroClusterType


class DenStreamOnlinePhase(OnlinePhase_StreamClusterer):

    def __init__(self, epsilon: float, mu: float, beta: float, lambda_decay: float):
        # TODO implement and read documentation in super class add argruments as needed
        super().__init__()
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.lambda_decay = lambda_decay
    
        # Calculate Tp (Pruning Trigger) based on the paper's formula (Eq 4.1)
        # self.Tp = ... 
        
        # Dictionaries mapping mc_id -> MicroCluster object
        self.p_micro_clusters = {} 
        self.o_micro_clusters = {}

        self.mc_counter = 0 # use this counter as generator for micro clusters IDs (increase by 1 when creating a new cluster) -> avoid duplicates
        self.current_time = 0

    
    def learn_one(self, x: np.ndarray, t: int) -> tuple[int, int, str]:
        """
        Routes the point to the nearest PMC, OMC, or creates a new OMC.
        should also call _periodic_cleanup()
        """
        self.current_time = t
        event_log = []
        summary_id = -1
        
        # TODO: Implement algorithm 1: find nearest PMC and try merging, if it fails, find nearest OMC and try to merge. when merging check if it can be promoted to PMC. if both fail , create a new OMC 
        # NOTE: pay attention to efficeincy during distance calculation
        
        # TODO: Implement algorithm 2: periodic cleanup NOTE: need to call _periodic_cleanup()
        
        self.current_time = t
        return (summary_id, summary_id, " | ".join(event_log))
    
    def _periodic_cleanup(self, event_log: list):
        """ TODO: Checks OMC weights against Xi, and PMC weights against beta*mu"""
        # look in the paper (Eq 4.2)
        pass
    
    # this can be left as it is. it just calls the learn_one method on multiple instances
    def learn(self, xs, ts) -> list[tuple[int, int, str]]:
        return super().learn(xs, ts)
    
    def get_summary(self) -> dict:
        """
        Return and pass both PMC and OMC to the offline phase as a dictionary
        """
        return {**self.p_micro_clusters, **self.o_micro_clusters}