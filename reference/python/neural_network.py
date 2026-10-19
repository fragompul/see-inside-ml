import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialization
        # He initialization for Relu-family and Linear, Xavier/Glorot for Tanh, Sigmoid, Swish, ELU, Softmax
        if activation in ['relu', 'leaky_relu', 'linear']:
            std = np.sqrt(2.0 / input_dim)
            self.weights = np.random.randn(input_dim, output_dim) * std
        else:
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
            
        self.biases = np.zeros((1, output_dim))
        
        # Gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.biases)
        
        # Optimizer states
        # First moments (Adam)
        self.mW = np.zeros_like(self.weights)
        self.mb = np.zeros_like(self.biases)
        # Second moments (RMSprop, Adam) or velocities (Momentum)
        self.vW = np.zeros_like(self.weights)
        self.vb = np.zeros_like(self.biases)
        # Timestep counter for bias correction
        self.t = 0
        
        # Cache for backpropagation
        self.inputs = None
        self.z = None
        self.a = None

    def forward(self, inputs):
        self.inputs = inputs
        # z = X*W + b
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activate(self.z)
        return self.a

    def activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0.0, z)
        elif self.activation == 'leaky_relu':
            return np.where(z > 0.0, z, 0.02 * z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            z_clipped = np.clip(z, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-z_clipped))
        elif self.activation == 'softmax':
            # Numerically stable softmax
            z_shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif self.activation == 'elu':
            z_clipped = np.clip(z, -500.0, 500.0)
            return np.where(z > 0.0, z, 1.0 * (np.exp(z_clipped) - 1.0))
        elif self.activation == 'swish':
            z_clipped = np.clip(z, -500.0, 500.0)
            sig = 1.0 / (1.0 + np.exp(-z_clipped))
            return z * sig
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def activate_derivative(self, z, a):
        if self.activation == 'relu':
            return (z > 0.0).astype(float)
        elif self.activation == 'leaky_relu':
            return np.where(z > 0.0, 1.0, 0.02)
        elif self.activation == 'tanh':
            return 1.0 - a ** 2
        elif self.activation == 'sigmoid':
            return a * (1.0 - a)
        elif self.activation == 'elu':
            return np.where(z > 0.0, 1.0, a + 1.0)
        elif self.activation == 'swish':
            z_clipped = np.clip(z, -500.0, 500.0)
            sig = 1.0 / (1.0 + np.exp(-z_clipped))
            return sig + a * (1.0 - sig)
        elif self.activation == 'linear':
            return np.ones_like(z)
        elif self.activation == 'softmax':
            # Handled directly at the output layer combining Loss + Softmax derivative
            return np.ones_like(z) 
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def backward(self, d_out):
        """
        d_out is dA_l = dL / dA_l
        Returns d_inputs (dA_l-1) for the previous layer
        """
        d_z = d_out * self.activate_derivative(self.z, self.a)
        m = self.inputs.shape[0]
        self.dW = np.dot(self.inputs.T, d_z) / m
        self.db = np.sum(d_z, axis=0, keepdims=True) / m
        d_inputs = np.dot(d_z, self.weights.T)
        return d_inputs


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, input_features, problem_type='classification', skip_connections=False, loss_type='default'):
        """
        layer_sizes: list of sizes, e.g. [input_dim, h1, h2, output_dim]
        activations: list of activation names
        input_features: list of string names of selected features
        problem_type: 'classification' or 'regression'
        skip_connections: if True, concatenates original input features to the input of each hidden layer
        loss_type: 'cce' (Categorical Cross Entropy), 'mse', 'mae', 'huber'
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.input_features = input_features
        self.problem_type = problem_type
        self.skip_connections = skip_connections
        self.loss_type = loss_type
        self.layers = []
        
        # Build layers
        input_dim = layer_sizes[0]
        for i in range(len(layer_sizes) - 1):
            act = activations[i] if i < len(activations) else 'linear'
            
            # Skip connections augment the input dimension with the original feature set
            if skip_connections and i > 0:
                current_input_dim = layer_sizes[i] + input_dim
            else:
                current_input_dim = layer_sizes[i]
                
            self.layers.append(DenseLayer(current_input_dim, layer_sizes[i+1], act))
            
    def forward(self, X):
        self.X_orig = X # Store for skip connections
        a = X
        for i, layer in enumerate(self.layers):
            if self.skip_connections and i > 0:
                layer_input = np.concatenate([a, self.X_orig], axis=1)
            else:
                layer_input = a
            a = layer.forward(layer_input)
        return a
        
    def compute_output_gradient(self, out, y):
        """
        Computes dZ_L = dL/dZ_L (pre-activation derivative of the output layer).
        Combines loss function derivative and output activation derivative for numerical stability.
        """
        if self.problem_type == 'classification':
            # For multi-class classification using Softmax and Categorical Cross Entropy
            # dZ = out - y (where y is one-hot encoded)
            return out - y
        else:
            # Regression (Linear output activation)
            d = out - y
            if self.loss_type == 'mae':
                return np.sign(d)
            elif self.loss_type == 'huber':
                delta = 1.0
                grad = np.copy(d)
                mask_large = np.abs(d) > delta
                grad[mask_large] = delta * np.sign(d[mask_large])
                return grad
            else:
                # Default: MSE
                return d
        
    def backward(self, X, y, out):
        """
        X: inputs of shape (m, input_dim)
        y: target labels of shape (m, num_classes) or (m, 1)
        out: network output of shape (m, num_classes) or (m, 1)
        """
        m = X.shape[0]
        
        # 1. Output layer pre-activation gradient
        d_z = self.compute_output_gradient(out, y)
        
        # 2. Backpropagate through the output layer
        last_layer = self.layers[-1]
        last_layer.dW = np.dot(last_layer.inputs.T, d_z) / m
        last_layer.db = np.sum(d_z, axis=0, keepdims=True) / m
        d_inputs = np.dot(d_z, last_layer.weights.T)
        
        # Slice output skip connection gradient if enabled
        if self.skip_connections and len(self.layers) > 1:
            n_a_prev = self.layers[-2].output_dim
            d_inputs = d_inputs[:, :n_a_prev]
            
        # 3. Backpropagate through hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            d_inputs = layer.backward(d_inputs)
            
            # Slice skip connection gradient for deeper hidden layers
            if self.skip_connections and i > 0:
                n_a_prev = self.layers[i-1].output_dim
                d_inputs = d_inputs[:, :n_a_prev]
                
    def update_weights(self, lr, optimizer='adam', momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, l1_rate=0.0, l2_rate=0.0):
        """
        Updates parameters of all layers using the specified optimizer.
        """
        for layer in self.layers:
            # Regularization gradients
            reg_w = 0.0
            if l1_rate > 0:
                reg_w += l1_rate * np.sign(layer.weights)
            if l2_rate > 0:
                reg_w += l2_rate * layer.weights
                
            dW_total = layer.dW + reg_w
            db_total = layer.db
            
            if optimizer == 'sgd':
                layer.weights -= lr * dW_total
                layer.biases -= lr * db_total
                
            elif optimizer == 'momentum':
                layer.vW = momentum * layer.vW + dW_total
                layer.vb = momentum * layer.vb + db_total
                layer.weights -= lr * layer.vW
                layer.biases -= lr * layer.vb
                
            elif optimizer == 'rmsprop':
                layer.vW = momentum * layer.vW + (1.0 - momentum) * (dW_total ** 2)
                layer.vb = momentum * layer.vb + (1.0 - momentum) * (db_total ** 2)
                layer.weights -= lr * dW_total / (np.sqrt(layer.vW) + epsilon)
                layer.biases -= lr * db_total / (np.sqrt(layer.vb) + epsilon)
                
            elif optimizer == 'adam':
                layer.t += 1
                layer.mW = beta1 * layer.mW + (1.0 - beta1) * dW_total
                layer.mb = beta1 * layer.mb + (1.0 - beta1) * db_total
                layer.vW = beta2 * layer.vW + (1.0 - beta2) * (dW_total ** 2)
                layer.vb = beta2 * layer.vb + (1.0 - beta2) * (db_total ** 2)
                
                mW_hat = layer.mW / (1.0 - beta1 ** layer.t)
                mb_hat = layer.mb / (1.0 - beta1 ** layer.t)
                vW_hat = layer.vW / (1.0 - beta2 ** layer.t)
                vb_hat = layer.vb / (1.0 - beta2 ** layer.t)
                
                layer.weights -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
                layer.biases -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
            
    def compute_loss(self, out, y, l1_rate=0.0, l2_rate=0.0):
        """
        Computes the loss + L1/L2 weight regularization penalty.
        """
        m = y.shape[0]
        
        if self.problem_type == 'classification':
            # Categorical Cross-Entropy Loss
            out_clipped = np.clip(out, 1e-15, 1.0 - 1e-15)
            loss = -np.mean(np.sum(y * np.log(out_clipped), axis=1))
        else:
            # Regression Loss
            d = out - y
            if self.loss_type == 'mae':
                loss = np.mean(np.abs(d))
            elif self.loss_type == 'huber':
                delta = 1.0
                abs_d = np.abs(d)
                huber_loss = np.where(abs_d <= delta, 0.5 * (d**2), delta * (abs_d - 0.5 * delta))
                loss = np.mean(huber_loss)
            else:
                # Default: MSE
                loss = 0.5 * np.mean(d ** 2)
                
        # Add regularization penalty
        reg_penalty = 0.0
        for layer in self.layers:
            if l1_rate > 0:
                reg_penalty += l1_rate * np.sum(np.abs(layer.weights))
            if l2_rate > 0:
                reg_penalty += 0.5 * l2_rate * np.sum(layer.weights ** 2)
                
        return loss + reg_penalty
        
    def compute_accuracy(self, out, y):
        """
        Computes classification accuracy (or MSE for regression).
        """
        if self.problem_type == 'classification':
            # Predictions are the index of the max probability
            predictions = np.argmax(out, axis=1)
            # Targets are one-hot, convert back to index
            targets = np.argmax(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y.flatten()
            return np.mean(predictions == targets)
        else:
            # Regression MSE
            return np.mean((out - y) ** 2)
