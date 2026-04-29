"""
Offline phase of DenStream algorithm
"""

import numpy as np
from baseframework.offline_clustering.adjacency import adjacency_clustering
from baseframework.offline_clustering.density_peak_clustering import DensityPeakClustering
from baseframework.stream_clustering import OfflinePhase_Clusterer



class DenStreamOfflinePhase(OfflinePhase_Clusterer):
    # NOTE: you can work on this seperately form online here, becasue we know that we receive a dictionary containing 2 lists with pmc and omc objects. look at get_summary()

    def __init__(self):
        super().__init__()
        #TODO initialize as needed. look at documentation of super class!
        pass

    def predict_summaries(self, summaries: dict) -> dict:
        # TODO implement clustering based on the summaries
        # read documentation of super
        # - keep only OMCs
        # - extract the centers 
        # - define the status array for each cluster, check if its 'dense' or 'noram' 
        # - call adjacency_clustering
        return super().predict_summaries(summaries) # remove this 
    
    def predict_assignment(self, assignments: list[int], summaries: dict) -> list[int]:
        #TODO implement - read documentation in super
    
        #summary_clustering = self.predict_summaries(summaries)
        return super().predict_assignment(assignments, summaries) # remove this
    
    #TODO: addition methods/helpers as required
