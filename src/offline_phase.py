"""
Offline phase of DenStream algorithm
"""

import numpy as np
from baseframework.offline_clustering.adjacency import adjacency_clustering
from baseframework.offline_clustering.density_peak_clustering import DensityPeakClustering
from baseframework.stream_clustering import OfflinePhase_Clusterer



class DenStreamOfflinePhase(OfflinePhase_Clusterer):
    # NOTE: you can work on this seperately form online here, becasue we know that we receive 2 disctionaries with pmc and omc objects. look at get_summary()

    def __init__(self):
        super().__init__()
        #TODO initialize as needed. look at documentation of super class!
        pass

    def predict_summaries(self, summaries: dict) -> dict:
        # TODO implement clustering based on the summaries
        # read documentation of super
        # this function looks at the Potential Micro-Clusters (PMCs), checks which ones are overlapping/touching, and groups them into final macro-clusters
        # - keep only PMCs 
        # - extract the centers 
        # - call DBSCAN

        # input: dictionary of summaries (micro-clusters)
        # output: a dictionary mapping each micro cluster's ID to a final cluster label (for example: {mc_id_0: 1, mc_id_1: 1, mc_id_2: -1} )
        return super().predict_summaries(summaries) # remove this 
    
    def predict_assignment(self, assignments: list[int], summaries: dict) -> list[int]:
        #TODO implement - read documentation in super
        # this function predicts the final macro-cluster labels for raw data points (assignments) -> use the summaries as a lookup table
        #  it  is responsible for mapping the assignments list to the dictionary output of predict_summaries

        # input: list of assignments for example [0, 0, 2] means we have 3 data points where the first two point belongs to mc_id_0 and the last to mc_id_2
        # ouput: a list of final cluster labels for those raw points based on the loopup table (for example [1,1,-1])
        # super class already implements the logic - change if needed though
        return super().predict_assignment(assignments, summaries)
    
    #TODO: addition methods/helpers as required
