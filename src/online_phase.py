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
        self.current_time = t # update time
        event_log = []
        summary_id = -1
        
        # TODO: Implement algorithm 1: 
        # Step 0: Update Time
        # step 1: try the PMCs: find nearest PMC and try merging, 
        # step 2: try the OMCs (if step 1 fails): find nearest OMC and try to merge
        # - when merging check if it can be promoted to PMC. 
        # step 3: create new OMC (if both step 1 and 2 fail)
        # step 4: clean up:  call _periodic_cleanup()
         
        # NOTE: pay attention to efficeincy during distance calculation!
        # check the method utils.distance_calc() in the base framework 

        # NOTE: here we pay attention to the initialization phase (section 4.1)
        # we have 2 options:
        # option 1: as the paper suggests, we can take the frist 1000 points of the stream, hold them in memory and then run the standard offline DBSCAN on them (creating the first batch of PMCs)
        # option 2: let the algorithm take care of this. this means the algorithm will try to find closest PMC (it fails), then tries to find closest OMC (it fails) so it will create new OMCs for the first few points and it will merge them quickly and promote them to PMCs


        self.current_time = t
        # return 
        return (summary_id, summary_id, " | ".join(event_log))


    def _periodic_cleanup(self, t: int, event_log: list):
        """
        Algorithm 2, lines 5-15.
        Prune PMCs whose weight fell below beta*mu, and OMCs whose weight
        is below their lower bound xi (Eq 4.2).
        """

        # Prune PMCs: weight < beta*mu means cp faded into outlier territory
        to_delete = []
        for mc_id, mc in self.p_micro_clusters.items():
            self._apply_decay_to(mc, t)
            if mc.weight < self.beta * self.mu:
                to_delete.append(mc_id)
        for mc_id in to_delete:
            del self.p_micro_clusters[mc_id]
            event_log.append(f"pruned PMC {mc_id}")

        # Prune OMCs: weight < xi(tc, t0) means co is unlikely to become a PMC
        # Eq 4.2: xi = (2^(-lambda*(tc - t0 + Tp)) - 1) / (2^(-lambda*Tp) - 1)
        to_delete = []
        for mc_id, mc in self.o_micro_clusters.items():
            self._apply_decay_to(mc, t)
            xi = (
                (2.0 ** (-self.lambda_decay * (t - mc.t_0 + self.Tp)) - 1.0)
                / (2.0 ** (-self.lambda_decay * self.Tp) - 1.0)
            )
            if mc.weight < xi:
                to_delete.append(mc_id)
        for mc_id in to_delete:
            del self.o_micro_clusters[mc_id]
            event_log.append(f"pruned OMC {mc_id}")

    def _apply_decay_to(self, mc: MicroCluster, current_time: int):
        """Lazily apply accumulated decay since mc.last_update."""
        time_diff = current_time - mc.last_update
        if time_diff > 0:
            mc.apply_decay(self.lambda_decay, time_diff)

    # this can be left as it is. it just calls the learn_one method on multiple instances
    def learn(self, xs, ts) -> list[tuple[int, int, str]]:
        return super().learn(xs, ts)

    def get_summary(self) -> dict:
        """
        Return and pass both PMC and OMC to the offline phase as a dictionary
        """
        return {**self.p_micro_clusters, **self.o_micro_clusters}