"""
Main entry point for running and experimenting with DenStream
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'baseframework/')

import argparse
import numpy as np
from src.denstream import DenStream
from baseframework.datahandler import load_data
try:
    from baseframework.metrics.cmm import get_cmm
    from baseframework.metrics.TSindex import tempsil
except ImportError:
    print("Warning: Could not import metrics!")


def main():
    parser = argparse.ArgumentParser(description="Run DenStream Clustering Simulation")
    # TODO: add and define all the configurable parameters with default values
    # - add clustering parameters (for dbscan: eps, min_pts, lambda, mu and beta)
    # - add decay parameters (decay base and decay factor)
    # - additional parameters as you see fit
    # check here for a list of the configuration optoins: https://infosys.beckhoff.com/english.php?content=../content/1033/te3520_tc3_analytics_service_tool/7678085899.html&id=

    # data loading configs
    parser.add_argument('--dataset', type=str, default='complex9', help='Name of the dataset (e.g., complex9, rbf3, har)')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of instances to process per batch')
    # - add data configuration paramters for testing (batch size, timeout maybe?, initialization params etc...)

    args = parser.parse_args()
    print(f"Loading data '{args.dataset}'...")
    X, y = load_data(args.dataset)


    den_stream = DenStream() # TODO pass needed args for denstream here

    num_batches = len(X) // args.batch_size

    print(f"\n=======================================================")
    print(f"STARTING STREAM PROCESSING: {num_batches} batches of size {args.batch_size}")
    print(f"PARAMETERS USED: ...") # TODO add and pass paramters used for printing
    print(f"=======================================================\n")

    #here we simulate a stream from the provided dataset
    for i in range(num_batches + 1):
        # get the next chunk of data  -> simulate the stream
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(X))
        batch = X[start_idx:end_idx]

        if len(batch) == 0:
            break

        # timestamp for this batch
        timestamps = np.arange(start_idx, end_idx)  # here use the batch size for the timestamps, so for example [0, 1, 2, 3 ... 499]
        
        # ONLINE PHASE (learn)
        results = den_stream.learn(batch, timestamps)

        # here we can add option to only call offlien phase when needed or each x number of steps
        # OFFLINE PHASE (predict assignment)
        labels = den_stream.predict(X_batch=batch)

        #num_clusters = len(np.unique(labels[labels != -1]))
        #print(f"   [Result] Batch {i} finished: {num_clusters} clusters were found")
    

    #TODO: Evaluation on final clustering with CMM + TSIndex



if __name__ == '__main__':
    main()


