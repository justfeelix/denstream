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


def _evaluate(X_acc, y_acc, labels_acc, ts_acc):
    """Compute and print CMM and Temporal Silhouette on accumulated data."""
    labels_arr = np.array(labels_acc)
    unique_clusters = np.unique(labels_arr[labels_arr != -1])
    if len(unique_clusters) < 2:
        print("   [Eval] Skipped: fewer than 2 non-noise clusters found.")
        return

    w = np.ones(len(X_acc))
    cmm = get_cmm(np.array(X_acc), np.array(y_acc), labels_arr, w, k=5)
    _, _, ts = tempsil(np.array(ts_acc), np.array(X_acc), labels_arr)
    print(f"   [Eval] CMM={cmm:.4f}  |  TS={ts:.4f}  |  "
          f"n={len(labels_arr)}  k={len(unique_clusters)}")


def main():
    parser = argparse.ArgumentParser(description="Run DenStream Clustering Simulation")

    # Dataset / batching
    parser.add_argument('--dataset', type=str, default='complex9',
                        help='Dataset name: complex9, rbf3, har, stream_densired')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Number of instances per batch')

    # DenStream parameters — paper defaults, with per-point lambda scaling
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Micro-cluster radius threshold (epsilon)')
    parser.add_argument('--mu', type=float, default=10.0,
                        help='Minimum weight for a core micro-cluster (mu)')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Outlier weight threshold factor (beta, 0 < beta <= 1)')
    parser.add_argument('--lambda_decay', type=float, default=0.00025,
                        help='Fading function decay factor (per-point lambda; paper uses 0.25 per 1000-point unit)')

    # Offline phase / evaluation
    parser.add_argument('--offline_interval', type=int, default=1,
                        help='Run offline clustering + evaluation every N batches (0 = only at end of stream)')

    args = parser.parse_args()

    print(f"Loading data '{args.dataset}'...")
    X, y = load_data(args.dataset)

    den_stream = DenStream(
        epsilon=args.epsilon,
        mu=args.mu,
        beta=args.beta,
        lambda_decay=args.lambda_decay,
    )

    num_batches = len(X) // args.batch_size

    print(f"\n=======================================================")
    print(f"STARTING STREAM PROCESSING: {num_batches} batches of size {args.batch_size}")
    print(f"PARAMETERS: epsilon={args.epsilon}  mu={args.mu}  "
          f"beta={args.beta}  lambda={args.lambda_decay}")
    print(f"=======================================================\n")

    # Accumulators — store every point's authoritative summary_id from the online phase
    all_X, all_y, all_assignments, all_ts = [], [], [], []

    for i in range(num_batches + 1):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(X))
        batch = X[start_idx:end_idx]

        if len(batch) == 0:
            break

        timestamps = np.arange(start_idx, end_idx)

        # ONLINE PHASE — each result is (summary_id, cluster_id, event_log)
        results = den_stream.learn(batch, timestamps)
        batch_assignments = [r[0] for r in results]

        all_X.extend(batch)
        all_y.extend(y[start_idx:end_idx])
        all_assignments.extend(batch_assignments)
        all_ts.extend(timestamps)

        print(f"   Batch {i:>4d} | points so far: {len(all_X):>6d}")

        # OFFLINE PHASE — only on demand
        run_offline = (
            args.offline_interval > 0 and (i + 1) % args.offline_interval == 0
        )
        if run_offline:
            labels = den_stream.predict(assignments=batch_assignments)
            num_clusters = len(np.unique(np.array(labels)[np.array(labels) != -1]))
            print(f"          [Offline] clusters this batch: {num_clusters}")
            _evaluate(list(batch), list(y[start_idx:end_idx]), labels, list(timestamps))

    # Final offline phase + evaluation — uses stored authoritative assignments
    # for the full stream, no nearest-PMC approximation needed.
    print(f"\n--- Final offline clustering + evaluation ---")
    final_labels = den_stream.predict(assignments=all_assignments)
    final_arr = np.array(final_labels)
    unique_clusters = np.unique(final_arr[final_arr != -1])
    print(f"    Total points: {len(all_X)}  |  clusters: {len(unique_clusters)}")
    _evaluate(all_X, all_y, final_labels, all_ts)



if __name__ == '__main__':
    main()


