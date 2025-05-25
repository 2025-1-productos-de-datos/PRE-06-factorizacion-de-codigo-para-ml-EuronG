import argparse


def parse_argument():

    parser = argparse.ArgumentParser(description="Train a sklearn model.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type: 'elasticnet' or 'knn'",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Model type: 'elasticnet' or 'knn'",
    )

    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,
        help="Model type: 'elasticnet' or 'knn'",
    )

    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=5,
        help="Model type: 'elasticnet' or 'knn'",
    )

    return parser.parse_args()