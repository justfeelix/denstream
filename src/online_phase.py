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
    """
    Online phase of DenStream.
    
    Initialization note: we deliberately skip the paper's batch DBSCAN
    initialization (for now). Instead, the algorithm bootstraps itself:
    early points create OMCs, which are promoted to PMCs once they accumulate
    enough weight (>= beta*mu). This trades a slightly slower start-up for
    significantly simpler implementation and robustness on small datasets.
    
    A future variant could add the batch initialization for closer paper
    fidelity if needed.
    """

    def __init__(self, epsilon: float, mu: float, beta: float, lambda_decay: float):
        super().__init__()
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.lambda_decay = lambda_decay

        # Tp: minimum time span before a PMC fades into outlier territory
        # Requires beta*mu > 1, otherwise log is undefined / negative.
        if beta * mu <= 1.0:
            raise ValueError(
                f"beta*mu must be > 1 for Tp to be well-defined "
                f"(got beta={beta}, mu={mu}, beta*mu={beta*mu})"
            )

        
        # Calculate Tp (Pruning Trigger) based on the paper's formula (Eq 4.1)        
        self.Tp = math.ceil(
        (1.0 / lambda_decay) * math.log2((beta * mu) / (beta * mu - 1.0))
        )


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
        # Step 0: Update Time
        self.current_time = t # update time
        event_log = []
        summary_id = -1
        
        # step 1: try the PMCs: find nearest PMC and try merging, 
        nearest_pmc, _ = self._find_nearest(x, self.p_micro_clusters, t)
        merged = False
        if nearest_pmc is not None:
            if self._try_merge(nearest_pmc, x, t):
                summary_id = nearest_pmc.mc_id
                event_log.append(f"merged_into_pmc_{summary_id}")
                merged = True

        # step 2: try the OMCs (if step 1 fails): find nearest OMC and try to merge
        # - when merging check if it can be promoted to PMC. 
        if not merged:
            nearest_omc, _ = self._find_nearest(x, self.o_micro_clusters, t)
            if nearest_omc is not None:
                if self._try_merge(nearest_omc, x, t):
                    summary_id = nearest_omc.mc_id
                    event_log.append(f"merged_into_omc_{summary_id}")
                    merged = True

                    # promotion check: paper Algorithm 1 line 8
                    if nearest_omc.weight > self.beta * self.mu:
                        self._promote_to_pmc(nearest_omc)
                        event_log.append(f"promoted_{summary_id}")

        # step 3: create new OMC (if both step 1 and 2 fail)
        if not merged:
            new_mc = MicroCluster(
                mc_id=self.mc_counter,
                t_0=t,
                mc_type=MicroClusterType.OUTLIER,
                point=x,
            )
            self.o_micro_clusters[self.mc_counter] = new_mc
            summary_id = self.mc_counter
            self.mc_counter += 1
            event_log.append(f"created_omc_{summary_id}")

        # step 4: clean up:  call _periodic_cleanup()
        if t > 0 and t % self.Tp == 0:
            self._periodic_cleanup(t, event_log)

        return (summary_id, -1, " | ".join(event_log)) 

        # NOTE: pay attention to efficeincy during distance calculation!
        # check the method utils.distance_calc() in the base framework 

        
    
    def _find_nearest(self, x: np.ndarray, clusters_dict: dict, t: int):
        """
        Find the micro-cluster in clusters_dict whose center is closest to x.

        Applies lazy decay to all clusters in the dict before computing distances,
        so centers reflect the current state at time t.
        """
        if not clusters_dict:
            return None, float('inf')

        mc_ids = list(clusters_dict.keys())
        centers = []
        for mc_id in mc_ids:
            mc = clusters_dict[mc_id]
            self._apply_decay_to(mc, t)
            centers.append(mc.get_center())

        centers = np.array(centers)
        # framework helper for euclidean distance from one point to many
        distances = distance_calc(x, centers, dist="euclidean")

        nearest_idx = int(np.argmin(distances))
        nearest_mc = clusters_dict[mc_ids[nearest_idx]]
        return nearest_mc, float(distances[nearest_idx])


    def _try_merge(self, mc: MicroCluster, x: np.ndarray, t: int) -> bool:
        """
        Test whether x can be merged into mc without exceeding epsilon.

        Uses copy-then-test to avoid mutating mc on a failed attempt.
        If the test passes, performs the actual merge on mc.
        """
        # mc has already been decay-updated by _find_nearest, so the copy starts
        # from a current state.
        test_copy = mc.copy()
        test_copy.insert_sample(x)

        if test_copy.get_radius() <= self.epsilon:
            # commit the merge on the original
            mc.insert_sample(x)
            return True

        return False
    

    def _promote_to_pmc(self, mc: MicroCluster):
        """
        Move a micro-cluster from the outlier buffer to the PMC dict
        and update its type tag.

        The mc keeps its mc_id (permanent ID design decision).
        """
        self.p_micro_clusters[mc.mc_id] = mc
        del self.o_micro_clusters[mc.mc_id]
        mc.mc_type = MicroClusterType.POTENTIAL

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