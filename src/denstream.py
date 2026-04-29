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
    
        summaries = self.online_phase.get_summary()
        #...
        pass


