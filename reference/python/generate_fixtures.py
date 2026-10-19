"""Generate golden fixtures for the TypeScript engine's parity tests.

Every case fixes inputs, targets AND initial weights explicitly, so no random-number-generator
behaviour has to match between NumPy and TypeScript: only the maths (forward, backward, loss,
optimizer updates) is compared.

Run from the repo root:  python reference/python/generate_fixtures.py
Output: packages/engine/fixtures/parity.json (committed; the Python code can be deleted afterwards).
"""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from neural_network import NeuralNetwork  # noqa: E402

OUT = HERE.parents[1] / "packages" / "engine" / "fixtures" / "parity.json"

HIDDEN_ACTIVATIONS = ["relu", "leaky_relu", "tanh", "sigmoid", "elu", "swish", "linear"]


def to_list(a):
    return np.asarray(a, dtype=float).tolist()


def snapshot_weights(net):
    return [{"W": to_list(l.weights), "b": to_list(l.biases)[0]} for l in net.layers]


def make_case(
    name,
    layer_sizes,
    activations,
    problem,
    loss_type,
    optimizer,
    lr,
    steps,
    skip=False,
    l1=0.0,
    l2=0.0,
    seed=0,
    m=14,
):
    rng = np.random.default_rng(seed)
    n_in = layer_sizes[0]
    X = rng.normal(size=(m, n_in))
    if problem == "classification":
        k = layer_sizes[-1]
        labels = rng.integers(0, k, size=m)
        y = np.eye(k)[labels]
    else:
        y = rng.normal(size=(m, 1)) * 2.0  # large enough to exercise Huber's linear branch

    net = NeuralNetwork(
        layer_sizes, activations, [f"f{i}" for i in range(n_in)], problem, skip, loss_type
    )
    for layer in net.layers:
        layer.weights = rng.normal(size=layer.weights.shape) * 0.6
        layer.biases = rng.normal(size=layer.biases.shape) * 0.2

    initial = snapshot_weights(net)
    first_out = None
    first_grads = None
    losses = []

    for step in range(steps):
        out = net.forward(X)
        if step == 0:
            first_out = to_list(out)
        losses.append(float(net.compute_loss(out, y, l1, l2)))
        net.backward(X, y, out)
        if step == 0:
            first_grads = [{"dW": to_list(l.dW), "db": to_list(l.db)[0]} for l in net.layers]
        net.update_weights(lr, optimizer=optimizer, l1_rate=l1, l2_rate=l2)

    final_out = net.forward(X)
    return {
        "name": name,
        "config": {
            "layerSizes": layer_sizes,
            "activations": activations,
            "problem": problem,
            "lossType": loss_type,
            "skipConnections": skip,
        },
        "train": {"optimizer": optimizer, "lr": lr, "steps": steps, "l1": l1, "l2": l2},
        "X": to_list(X),
        "y": to_list(y),
        "initialWeights": initial,
        "expected": {
            "firstOutput": first_out,
            "firstGrads": first_grads,
            "losses": losses,
            "finalOutput": to_list(final_out),
            "finalLoss": float(net.compute_loss(final_out, y, l1, l2)),
            "finalMetric": float(net.compute_accuracy(final_out, y)),
            "finalWeights": snapshot_weights(net),
        },
    }


def schedules():
    lr = 0.03
    epochs = [0, 1, 7, 50, 100, 250, 499, 500, 731, 1500]
    t_max = 500
    lr_min = lr * 0.01
    return {
        "baseLr": lr,
        "epochs": epochs,
        "exponential": [lr * (0.96 ** (e / 100)) for e in epochs],
        "cosine": [
            lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * (e % t_max) / t_max))
            for e in epochs
        ],
    }


def main():
    cases = []

    for act in HIDDEN_ACTIVATIONS:
        cases.append(
            make_case(
                f"activation-{act}",
                [3, 5, 4, 3],
                [act, act, "softmax"],
                "classification",
                "default",
                "sgd",
                0.05,
                8,
                seed=1,
            )
        )

    for opt in ["sgd", "momentum", "rmsprop", "adam"]:
        cases.append(
            make_case(
                f"optimizer-{opt}-classification",
                [2, 6, 6, 3],
                ["tanh", "tanh", "softmax"],
                "classification",
                "default",
                opt,
                0.03,
                25,
                seed=2,
            )
        )
        cases.append(
            make_case(
                f"optimizer-{opt}-regression",
                [2, 5, 5, 1],
                ["relu", "relu", "linear"],
                "regression",
                "mse",
                opt,
                0.02,
                25,
                seed=3,
            )
        )

    for loss in ["mse", "mae", "huber"]:
        cases.append(
            make_case(
                f"loss-{loss}",
                [2, 6, 4, 1],
                ["elu", "tanh", "linear"],
                "regression",
                loss,
                "adam",
                0.02,
                12,
                seed=4,
            )
        )

    cases.append(
        make_case(
            "skip-classification",
            [3, 5, 4, 2],
            ["tanh", "relu", "softmax"],
            "classification",
            "default",
            "adam",
            0.02,
            12,
            skip=True,
            seed=5,
        )
    )
    cases.append(
        make_case(
            "skip-regression",
            [2, 4, 4, 4, 1],
            ["swish", "tanh", "leaky_relu", "linear"],
            "regression",
            "huber",
            "momentum",
            0.01,
            12,
            skip=True,
            seed=6,
        )
    )

    for label, l1, l2 in [("l1", 0.01, 0.0), ("l2", 0.0, 0.05), ("l1-l2", 0.01, 0.05)]:
        cases.append(
            make_case(
                f"regularization-{label}",
                [2, 6, 6, 2],
                ["relu", "tanh", "softmax"],
                "classification",
                "default",
                "adam",
                0.02,
                15,
                l1=l1,
                l2=l2,
                seed=7,
            )
        )

    cases.append(
        make_case(
            "deep-tanh",
            [2, 6, 6, 6, 6, 2],
            ["tanh", "tanh", "tanh", "tanh", "softmax"],
            "classification",
            "default",
            "adam",
            0.01,
            20,
            seed=8,
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": "reference/python/generate_fixtures.py",
        "numpy": np.__version__,
        "schedules": schedules(),
        "cases": cases,
    }
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
