"""
Neural Network metamodels for regression using PyTorch.

This module provides neural network-based metamodel regressors with single hidden layer MLP architecture.
"""

import numpy as np
from typing import Optional, Callable, Tuple
from surroptim.meta_models import metamodel

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required for neural network metamodels")


class MLP(torch.nn.Module):
    """Multi-layer perceptron with single hidden layer and sigmoid activation."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        """
        Initialize MLP.

        Args:
            n_in: Input dimension
            n_hidden: Number of hidden units
            n_out: Output dimension
        """
        super().__init__()
        self.linear_in = torch.nn.Linear(n_in, n_hidden)
        self.activation = torch.nn.Sigmoid()
        self.linear_out = torch.nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        out = self.linear_in(x)
        out = self.activation(out)
        out = self.linear_out(out)
        return out


class neural_net_regressor(metamodel):
    """Neural network regressor with numpy input/output interface."""

    def __init__(self, n_hidden: int = 100):
        """
        Initialize neural network regressor.

        Args:
            n_hidden: Number of hidden units in the MLP
        """
        super().__init__()
        self.model: Optional[MLP] = None
        self.n_hidden = n_hidden

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 1.0e-2,
        epochs: int = 10000,
        plot_func: Optional[Callable] = None,
        restart: bool = True,
    ) -> None:
        """
        Train the neural network model.

        Args:
            X: Training samples, shape (n_samples, n_dims)
            y: Training targets, shape (n_samples,) or (n_samples, n_outputs)
            lr: Learning rate for Adam optimizer
            epochs: Number of training epochs
            plot_func: Optional callback function for visualization during training
            restart: If True, reinitialize model; if False, continue training existing model
        """
        # Ensure numpy arrays are double precision to enable zero-copy conversion
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        super().train_init(X, y)

        # Reinitialize model if restart=True or model doesn't exist
        if (self.model is None) or restart:
            self.model = MLP(n_in=self.dim, n_hidden=self.n_hidden, n_out=self.n_out)
            # enforce double precision for model parameters
            self.model.double()

        # Convert to torch tensors (zero-copy when NumPy is float64)
        X_torch = torch.from_numpy(X).requires_grad_(False)
        y_torch = torch.from_numpy(y).requires_grad_(False)

        # Setup optimizer
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        for epoch in range(1, epochs + 1):
            y_pred = self.model.forward(X_torch)
            loss = loss_fn(y_torch.squeeze(), y_pred.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if plot_func is not None:
                plot_func(epoch, X_torch, y_pred, self.predict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new samples.

        Args:
            X: Input samples, shape (n_samples, n_dims)

        Returns:
            Predictions, shape (n_samples,) or (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        X = np.asarray(X, dtype=np.float64)
        X_torch = torch.from_numpy(X)
        return self.model.forward(X_torch).detach().numpy()

    def predict_and_grad(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return predictions and input gradients using PyTorch autograd.

        Returns
        -------
        y_np : np.ndarray
            Model predictions, shape (n_samples, n_out)
        grads_np : np.ndarray
            Gradients d y / d x with shape (n_samples, n_out, n_in)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X = np.asarray(X, dtype=np.float64)
        X_torch = torch.from_numpy(X).requires_grad_(True)
        y_torch = self.model.forward(X_torch)

        # Ensure 2D outputs (N, n_out)
        if y_torch.dim() == 1:
            y_torch = y_torch.unsqueeze(1)

        N, n_out = y_torch.shape
        n_in = X_torch.shape[1]

        y_np = y_torch.detach().numpy()

        grads = np.zeros((N, n_out, n_in), dtype=float)

        for i in range(N):
            for j in range(n_out):
                grad_full = torch.autograd.grad(
                    y_torch[i, j], X_torch, retain_graph=True, allow_unused=True
                )[0]
                if grad_full is None:
                    grads[i, j, :] = 0.0
                else:
                    grads[i, j, :] = grad_full[i].detach().numpy()

        return y_np, grads


class neural_net_regressor_pt(metamodel):
    """Neural network regressor with PyTorch tensor input/output interface."""

    def __init__(self, n_hidden: int = 100):
        """
        Initialize neural network regressor (PyTorch interface).

        Args:
            n_hidden: Number of hidden units in the MLP
        """
        super().__init__()
        self.model: Optional[MLP] = None
        self.n_hidden = n_hidden

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 1.0e-2,
        epochs: int = 10000,
        plot_func: Optional[Callable] = None,
        restart: bool = True,
    ) -> None:
        """
        Train the neural network model (PyTorch tensors).

        Args:
            X: Training samples tensor, shape (n_samples, n_dims)
            y: Training targets tensor, shape (n_samples,) or (n_samples, n_outputs)
            lr: Learning rate for Adam optimizer
            epochs: Number of training epochs
            plot_func: Optional callback function for visualization during training
            restart: If True, reinitialize model; if False, continue training existing model
        """
        # For compatibility with metamodel base class, store as numpy temporarily
        # ensure incoming tensors are double precision
        if isinstance(X, torch.Tensor):
            X = X.double()
        if isinstance(y, torch.Tensor):
            y = y.double()
        X_np = X.detach().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().numpy() if isinstance(y, torch.Tensor) else y
        super().train_init(X_np, y_np)

        # Reinitialize model if restart=True or model doesn't exist
        if (self.model is None) or restart:
            self.model = MLP(n_in=self.dim, n_hidden=self.n_hidden, n_out=self.n_out)
            # enforce double precision for model parameters
            self.model.double()

        # Setup optimizer
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_fn = torch.nn.MSELoss()

        # Training loop (X and y are already torch tensors)
        for epoch in range(1, epochs + 1):
            y_pred = self.model.forward(X)
            loss = loss_fn(y.squeeze(), y_pred.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if plot_func is not None:
                plot_func(epoch, X, y_pred, self.predict)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict on new samples (PyTorch tensors).

        Args:
            X: Input samples tensor, shape (n_samples, n_dims)

        Returns:
            Predictions tensor, shape (n_samples,) or (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.forward(X.double())
