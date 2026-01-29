"""
Neural network metamodels for regression using PyTorch.

This module provides neural network-based metamodel regressors with a
single-hidden-layer MLP architecture. Importing this module does not
raise when PyTorch is absent; instead lightweight stubs are provided
that raise helpful ImportError messages at use-time.
"""

import numpy as np
from typing import Optional, Callable, Tuple, Any
from surroptim.meta_models import metamodel

# Optional import of torch. Do NOT raise on ImportError here so test
# collection can inspect the module even when PyTorch is not installed.
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class MLP(torch.nn.Module):
        """Multi-layer perceptron with single hidden layer and sigmoid activation."""

        def __init__(self, n_in: int, n_hidden: int, n_out: int):
            super().__init__()
            self.linear_in = torch.nn.Linear(n_in, n_hidden)
            self.activation = torch.nn.Sigmoid()
            self.linear_out = torch.nn.Linear(n_hidden, n_out)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            out = self.linear_in(x)
            out = self.activation(out)
            out = self.linear_out(out)
            return out


    class neural_net_regressor(metamodel):
        """Neural network regressor with NumPy input/output interface."""

        def __init__(self, n_hidden: int = 100):
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
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            super().train_init(X, y)

            if (self.model is None) or restart:
                self.model = MLP(n_in=self.dim, n_hidden=self.n_hidden, n_out=self.n_out)
                self.model.double()

            X_torch = torch.from_numpy(X).requires_grad_(False)
            y_torch = torch.from_numpy(y).requires_grad_(False)

            params = self.model.parameters()
            optimizer = torch.optim.Adam(params, lr=lr)
            loss_fn = torch.nn.MSELoss()

            for epoch in range(1, epochs + 1):
                y_pred = self.model.forward(X_torch)
                loss = loss_fn(y_torch.squeeze(), y_pred.squeeze())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if plot_func is not None:
                    plot_func(epoch, X_torch, y_pred, self.predict)

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self.model is None:
                raise ValueError("Model not trained yet")
            X = np.asarray(X, dtype=np.float64)
            X_torch = torch.from_numpy(X)
            return self.model.forward(X_torch).detach().numpy()

        def predict_and_grad(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if self.model is None:
                raise ValueError("Model not trained yet")

            X = np.asarray(X, dtype=np.float64)
            X_torch = torch.from_numpy(X).requires_grad_(True)
            y_torch = self.model.forward(X_torch)

            if y_torch.dim() == 1:
                y_torch = y_torch.unsqueeze(1)

            N, n_out = y_torch.shape
            n_in = X_torch.shape[1]

            y_np = y_torch.detach().numpy()
            grads = np.zeros((N, n_out, n_in), dtype=float)

            for j in range(n_out):
                retain = j < (n_out - 1)
                grad_full = torch.autograd.grad(
                    y_torch[:, j].sum(), X_torch, retain_graph=retain, allow_unused=True
                )[0]
                if grad_full is None:
                    continue
                grads[:, j, :] = grad_full.detach().cpu().numpy()

            return y_np, grads


    class neural_net_regressor_pt(metamodel):
        """Neural network regressor with PyTorch tensor interface."""

        def __init__(self, n_hidden: int = 100):
            super().__init__()
            self.model: Optional[MLP] = None
            self.n_hidden = n_hidden

        def train(self, X: 'torch.Tensor', y: 'torch.Tensor', lr: float = 1.0e-2, epochs: int = 10000, plot_func: Optional[Callable] = None, restart: bool = True) -> None:
            X_np = X.detach().numpy() if isinstance(X, torch.Tensor) else X
            y_np = y.detach().numpy() if isinstance(y, torch.Tensor) else y
            super().train_init(X_np, y_np)

            if (self.model is None) or restart:
                self.model = MLP(n_in=self.dim, n_hidden=self.n_hidden, n_out=self.n_out)
                self.model.double()

            # Ensure input tensors use the same dtype as the model (double)
            if isinstance(X, torch.Tensor):
                X = X.double()
            if isinstance(y, torch.Tensor):
                y = y.double()

            params = self.model.parameters()
            optimizer = torch.optim.Adam(params, lr=lr)
            loss_fn = torch.nn.MSELoss()

            for epoch in range(1, epochs + 1):
                y_pred = self.model.forward(X)
                loss = loss_fn(y.squeeze(), y_pred.squeeze())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if plot_func is not None:
                    plot_func(epoch, X, y_pred, self.predict)

        def predict(self, X: 'torch.Tensor') -> 'torch.Tensor':
            if self.model is None:
                raise ValueError("Model not trained yet")
            return self.model.forward(X.double())

else:
    # PyTorch not available: provide lightweight stubs that raise helpful
    # ImportError messages when instantiated or used. This prevents
    # import-time failures while still giving useful diagnostics at runtime.
    class MLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MLP; install PyTorch to use neural network metamodels")

    class neural_net_regressor(metamodel):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural network metamodels")

    class neural_net_regressor_pt(metamodel):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural network metamodels")
    
