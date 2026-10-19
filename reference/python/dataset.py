import numpy as np
from sklearn.datasets import make_blobs

def generate_classification_data(dataset_type, n_samples=200, noise=0.0, num_classes=2):
    """
    Generates synthetic classification datasets: 'circle', 'xor', 'gaussian', 'spiral'
    Supports up to 5 classes natively.
    """
    np.random.seed(42)
    
    if dataset_type == 'circle':
        # Concentric circles for num_classes
        samples_per_class = n_samples // num_classes
        X_list, y_list = [], []
        for i in range(num_classes):
            r = (i + 1) * (5.0 / num_classes)
            t = np.linspace(0, 2 * np.pi, samples_per_class)
            x1 = r * np.sin(t) + np.random.randn(samples_per_class) * noise * 0.5
            x2 = r * np.cos(t) + np.random.randn(samples_per_class) * noise * 0.5
            X_list.append(np.column_stack([x1, x2]))
            y_list.append(np.full(samples_per_class, i))
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
    elif dataset_type == 'gaussian':
        # Gaussian clusters based on num_classes
        cluster_std = 0.5 + noise * 1.5
        X, y = make_blobs(n_samples=n_samples, centers=num_classes, cluster_std=cluster_std, random_state=42)
        # Scale to roughly -5 to 5
        X = (X - X.mean(axis=0)) / X.std(axis=0) * 2.5
        
    elif dataset_type == 'xor':
        # Multi-class checkerboard
        X = np.random.uniform(-5, 5, (n_samples, 2))
        # Grid size dependent on classes to ensure enough variation
        grid_scale = 1.0 + (num_classes * 0.2)
        y = (np.floor(X[:, 0] / grid_scale) + np.floor(X[:, 1] / grid_scale)) % num_classes
        # Add noise
        X += np.random.normal(0, noise * 0.5, X.shape)
        if noise > 0:
            flip_mask = np.random.rand(n_samples) < (noise * 0.1)
            y[flip_mask] = np.random.randint(0, num_classes, np.sum(flip_mask))
            
    elif dataset_type == 'spiral':
        # Multi-arm spiral
        samples_per_arm = n_samples // num_classes
        X_list, y_list = [], []
        
        for j in range(num_classes):
            i = np.arange(samples_per_arm)
            r = i / samples_per_arm * 5.0
            t = 1.75 * i / samples_per_arm * 2 * np.pi + (j * 2 * np.pi / num_classes) + np.random.randn(samples_per_arm) * noise * 0.25
            x1 = r * np.sin(t)
            x2 = r * np.cos(t)
            X_list.append(np.column_stack([x1, x2]))
            y_list.append(np.full(samples_per_arm, j))
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
    else:
        raise ValueError(f"Unknown classification dataset type: {dataset_type}")

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices].astype(int)

def generate_regression_data(dataset_type, n_samples=200, noise=0.0):
    """
    Generates synthetic regression datasets: 'plane', 'saddle'
    """
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (n_samples, 2))
    
    if dataset_type == 'plane':
        # y is a flat plane tilted along X1 and X2
        y = 0.5 * X[:, 0] + 0.5 * X[:, 1]
        
    elif dataset_type == 'saddle':
        # y is a sinusoidal saddle-like function
        # Scale coords to fit sine wavelength well
        y = np.sin(X[:, 0] * 0.6) * np.cos(X[:, 1] * 0.6) * 3.0
        
    else:
        raise ValueError(f"Unknown regression dataset type: {dataset_type}")
        
    # Add noise
    y += np.random.normal(0, noise * 0.5, y.shape)
    
    # Return y as standard array
    return X, y

def extract_features(X, selected_features):
    """
    Given a dataset X of shape (N, 2), maps it to a feature matrix based on selected_features list.
    """
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    feature_dict = {
        "X1": X1,
        "X2": X2,
        "X1^2": X1**2,
        "X2^2": X2**2,
        "X1*X2": X1 * X2,
        "sin(X1)": np.sin(X1),
        "sin(X2)": np.sin(X2),
        "cos(X1)": np.cos(X1),
        "cos(X2)": np.cos(X2),
        "sgn(X1)": np.sign(X1),
        "sgn(X2)": np.sign(X2),
        "RBF_Gaussian": np.exp(-(X1**2 + X2**2) / 4.0)
    }
    
    cols = []
    for f in selected_features:
        if f in feature_dict:
            cols.append(feature_dict[f])
        else:
            raise ValueError(f"Unknown feature: {f}")
            
    return np.column_stack(cols)
