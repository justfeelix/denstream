"""
Offline phase of DenStream algorithm
"""

import numpy as np
from sklearn.cluster import DBSCAN
from baseframework.stream_clustering import OfflinePhase_Clusterer
from .micro_clusters import MicroClusterType


class DenStreamOfflinePhase(OfflinePhase_Clusterer):

    def __init__(self, epsilon: float, mu: float):
        super().__init__()
        self.epsilon = epsilon
        self.mu = mu

    def predict_summaries(self, summaries: dict) -> dict:
        """
        Apply a variant of DBSCAN to the current set of p-micro-clusters to
        produce final macro-cluster labels (Section 4.2).

        Only PMCs are considered. OMCs are labelled noise (-1) by definition.

        Per Definition 4.1, two PMCs are directly density-reachable when
            dist(cp, cq) <= 2 * epsilon
        and the source PMC has weight >= mu (i.e. it is a c-micro-cluster).

        sklearn DBSCAN with sample_weight encodes this exactly: a sample whose
        weight is >= min_samples is by itself a core sample, which matches the
        paper's per-PMC weight threshold.
        """
        # Start with every entry labelled noise; OMCs stay -1.
        result = {mc_id: -1 for mc_id in summaries}

        pmc_ids = [
            mc_id for mc_id, mc in summaries.items()
            if mc.mc_type == MicroClusterType.POTENTIAL
        ]

        if len(pmc_ids) == 0:
            return result

        centers = np.array([summaries[mc_id].get_center() for mc_id in pmc_ids])
        weights = np.array([summaries[mc_id].weight for mc_id in pmc_ids])

        # Definition 4.1: eps = 2*epsilon; core point iff weight >= mu
        db = DBSCAN(eps=2.0 * self.epsilon, min_samples=self.mu, metric='euclidean')
        labels = db.fit_predict(centers, sample_weight=weights)

        for mc_id, label in zip(pmc_ids, labels):
            result[mc_id] = label

        return result

    def predict_assignment(self, assignments: list[int], summaries: dict) -> list[int]:
        # Same logic as the base class, but uses .get(a, -1) instead of [a].
        # The base class would KeyError if a summary_id was pruned between
        # learn_one() and predict() — here we treat pruned IDs as noise (-1).
        summary_clustering = self.predict_summaries(summaries)
        summary_clustering[-1] = -1
        return [summary_clustering.get(a, -1) for a in assignments]
