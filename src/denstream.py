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

    #TODO add and pass args as needed
    def __init__(self, epsilon=0.1, mu=5.0, beta=0.2, lambda_decay=0.25): 
        """
        Initialize DenStream algorithm with requried parameters

        Args:
            #TODO
        """
        #TODO correct initialization / change as needed
        self.online_phase = DenStreamOnlinePhase(epsilon=epsilon, mu=mu, beta=beta, lambda_decay=lambda_decay) 
        self.offline_phase = DenStreamOfflinePhase()
        pass

    
    # for online
    def learn(self, xs, ts) -> list:
        """Feed a new batch of data into the online phase using the framework's learn method"""
        #leave as it is -> the actual implementation needs to be done in learn_one() in DenStreamOnlinePhase
        return self.online_phase.learn(xs, ts)
    
    # for offline
    def predict(self, X_batch=None):
        """Trigger the offline phase to get the current clustering"""

        # TODO: fetch summaries and call the framework's predict_assignment method

        # here differntiate between two cases:
        #   1. if the user wants to know how the bubbles are clustered, they call predict() with no argument and return predict_summaries()
        #   2. if the user wants to evaluate the algorithm against a specific batch of raw points (for example when evaluating with cmm) they can call predict(X_batch)

        # fetch summaries
        summaries = self.online_phase.get_summary()
        # 1st case
        if X_batch is None:
            return self.offline_phase.predict_summaries(summaries)
        #2nd case
        else:
            # The user wants to cluster specific raw data points (X_batch)
            assignments = []
            #TODO:  assign a raw point to a cluster -> find its nearest PMC

            # at the end call the self.offline_phase.predict_assignment(assignments, summaries)
            # and return ouput


        pass


