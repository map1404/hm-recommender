"""
training.py
-----------
ML Developer A (Tianshi) module.

Trains an implicit-feedback ALS model on the user-item purchase matrix.

Outputs:
  models/als_model.pkl  — serialised ALS model
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

MODELS_DIR = Path("models")


def train(
    factors: int = 64,
    iterations: int = 20,
    regularization: float = 0.01,
    alpha: float = 40.0,
    random_state: int = 42,
):
    import implicit

    print("Loading user-item matrix...")
    user_item = load_npz(MODELS_DIR / "user_item_matrix.npz")
    item_user = (user_item.T.tocsr() * alpha).astype(np.float32)

    print(f"Training ALS  factors={factors}  iters={iterations}  reg={regularization}...")
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=random_state,
        use_gpu=False,
    )
    model.fit(item_user)

    with open(MODELS_DIR / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("ALS model saved to models/als_model.pkl")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=40.0)
    args = parser.parse_args()
    train(args.factors, args.epochs, args.regularization, args.alpha)
