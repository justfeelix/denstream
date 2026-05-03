"""DenStream: Density-Based Clustering over an Evolving Data Stream
This is the class representing both combination of online and offline phases"""

import numpy as np
from .online_phase import DenStreamOnlinePhase
from .offline_phase import DenStreamOfflinePhase


class DenStream:
    """
    DenStream algorithm implementation for density-based clustering over evolving data streams
    Orechestrates both online and offline phases
    """

    def __init__(self, epsilon=0.1, mu=10.0, beta=0.2, lambda_decay=0.25):
        """
        Initialize DenStream algorithm.

        Parameters match Section 6.1 of the paper (scaled for [0,1] data):
            epsilon      – micro-cluster radius threshold
            mu           – minimum weight for a core micro-cluster
            beta         – outlier weight threshold factor (0 < beta <= 1)
            lambda_decay – fading function decay factor
        """
        self.online_phase = DenStreamOnlinePhase(epsilon=epsilon, mu=mu, beta=beta, lambda_decay=lambda_decay)
        self.offline_phase = DenStreamOfflinePhase(epsilon=epsilon, mu=mu)

    
    # for online
    def learn(self, xs, ts) -> list:
        """Feed a new batch of data into the online phase using the framework's learn method"""
        #leave as it is -> the actual implementation needs to be done in learn_one() in DenStreamOnlinePhase
        return self.online_phase.learn(xs, ts)
    
    # for offline
    def predict(self, assignments=None):
        """Trigger the offline phase to get the current clustering.

        Two modes:
          predict()                   – return macro-cluster label per PMC/OMC
                                        (dict mapping summary_id -> label)
          predict(assignments=[...])  – map a list of summary_ids (as returned
                                        by learn()) to macro-cluster labels.
                                        Each point already has an authoritative
                                        micro-cluster assignment from the online
                                        phase; this simply looks up its label.
        """
        summaries = self.online_phase.get_summary()

        if assignments is None:
            return self.offline_phase.predict_summaries(summaries)

        return self.offline_phase.predict_assignment(assignments, summaries)


