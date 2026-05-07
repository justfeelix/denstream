"""
Run DenStream from the River library and evaluate with baseframework metrics.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'baseframework/')

import argparse
import numpy as np
from river import cluster

from baseframework.datahandler import load_data

try:
    from baseframework.metrics.cmm import get_cmm
    from baseframework.metrics.TSindex import tempsil
except ImportError:
    print("Warning: Could not import metrics!")


def _to_river_point(x: np.ndarray) -> dict[int, float]:
    """Convert numpy row vectors to River's dict-based input format."""
    return {i: float(v) for i, v in enumerate(x)}


def _evaluate(x_all, y_all, labels_all, ts_all):
    """Compute and print CMM and Temporal Silhouette on accumulated data."""
    labels_arr = np.array(labels_all)
    unique_clusters = np.unique(labels_arr[labels_arr != -1])
    if len(unique_clusters) < 2:
        print("   [Eval] Skipped: fewer than 2 non-noise clusters found.")
        return

    w = np.ones(len(x_all))
    cmm = get_cmm(np.array(x_all), np.array(y_all), labels_arr, w, k=5)
    _, _, ts = tempsil(np.array(ts_all), np.array(x_all), labels_arr)
    print(f"   [Eval] CMM={cmm:.4f}  |  TS={ts:.4f}  |  "
          f"n={len(labels_arr)}  k={len(unique_clusters)}")


def _predict_all(model: cluster.DenStream, x_all: np.ndarray) -> list[int]:
    """
    Predict final labels for all points from the current River DenStream state.

    We trigger River's offline clustering once via predict_one and then classify
    all points against the generated macro-clusters.
    """
    if len(x_all) == 0 or not model.initialized:
        return [-1] * len(x_all)

    # Trigger generation of macro-clusters from current p-micro-clusters.
    model.predict_one(_to_river_point(x_all[0]))
    clusters = getattr(model, "clusters", {})
    if len(clusters) == 0:
        return [-1] * len(x_all)

    return [
        model._get_closest_cluster_key(_to_river_point(x), clusters)
        for x in x_all
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run River DenStream Clustering Simulation"
    )

    # Dataset / progress printing cadence
    parser.add_argument('--dataset', type=str, default='complex9',
                        help='Dataset name: complex9, rbf3, har, stream_densired')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Progress print interval in number of processed points')

    # Parameters aligned with the local project defaults where possible.
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Micro-cluster radius threshold (epsilon)')
    parser.add_argument('--mu', type=float, default=10.0,
                        help='Minimum weight for a core micro-cluster (mu)')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Outlier weight threshold factor (beta, 0 < beta <= 1)')
    parser.add_argument('--lambda_decay', type=float, default=0.00025,
                        help='Fading function decay factor')

    # River-specific controls.
    parser.add_argument('--n_samples_init', type=int, default=1000,
                        help='Number of initial points buffered for River initialization')
    parser.add_argument('--stream_speed', type=int, default=1,
                        help='Points per River time unit (1 means per-point time updates)')

    args = parser.parse_args()

    print(f"Loading data '{args.dataset}'...")
    x_all, y_all = load_data(args.dataset)

    model = cluster.DenStream(
        decaying_factor=args.lambda_decay,
        beta=args.beta,
        mu=args.mu,
        epsilon=args.epsilon,
        n_samples_init=args.n_samples_init,
        stream_speed=args.stream_speed,
    )

    print("\n=======================================================")
    print(f"STARTING STREAM PROCESSING: {len(x_all)} points")
    print(f"PARAMETERS: epsilon={args.epsilon}  mu={args.mu}  "
          f"beta={args.beta}  lambda={args.lambda_decay}  "
          f"n_samples_init={args.n_samples_init}  stream_speed={args.stream_speed}")
    print("=======================================================\n")

    for i, x in enumerate(x_all):
        model.learn_one(_to_river_point(x))
        if (i + 1) % args.batch_size == 0 or (i + 1) == len(x_all):
            print(f"   Points processed: {i + 1:>6d}")

    print("\n--- Final offline clustering + evaluation (River) ---")
    labels = _predict_all(model, x_all)
    labels_arr = np.array(labels)
    unique_clusters = np.unique(labels_arr[labels_arr != -1])
    print(f"    Total points: {len(x_all)}  |  clusters: {len(unique_clusters)}")

    timestamps = np.arange(len(x_all))
    _evaluate(x_all, y_all, labels, timestamps)


if __name__ == '__main__':
    main()
