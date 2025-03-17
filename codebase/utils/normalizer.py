import numpy as np
import torch

class RunningNormalizer:
    def __init__(self, shape, momentum=0.001, epsilon=1e-8):
        # The backend will be determined on the first call to update() or normalize()
        self.momentum = momentum
        self.epsilon = epsilon
        self.count = 0
        self.shape = shape
        self.backend = None  # Will be set to "numpy" or "torch"
        self.mean = None
        self.var = None

    def _init_backend(self, batch):
        if isinstance(batch, np.ndarray):
            self.backend = "numpy"
            self.mean = np.zeros(self.shape, dtype=np.float32)
            self.var = np.ones(self.shape, dtype=np.float32)
        elif isinstance(batch, torch.Tensor):
            self.backend = "torch"
            
            # Use the batch device for our internal tensors
            device = batch.device
            self.mean = torch.zeros(self.shape, dtype=torch.float32, device=device)
            self.var = torch.ones(self.shape, dtype=torch.float32, device=device)
        else:
            raise TypeError("Unsupported data type: Expected numpy.ndarray or torch.Tensor.")

    def update(self, batch):
        # Initialize backend if not set yet
        if self.backend is None:
            self._init_backend(batch)
        
        # Check that batch type matches the backend
        if self.backend == "numpy":
            if not isinstance(batch, np.ndarray):
                raise TypeError("Expected numpy.ndarray for update, but got a different type.")
            
            # Compute statistics over the batch (along axis 0)
            batch_mean = np.mean(batch, axis=0)
            batch_var = np.var(batch, axis=0)
            batch_count = batch.shape[0]

            # Exponential moving average update
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.var = (1 - self.momentum) * self.var + self.momentum * batch_var
            self.count += batch_count
        elif self.backend == "torch":
            if not isinstance(batch, torch.Tensor):
                raise TypeError("Expected torch.Tensor for update, but got a different type.")
            batch_mean = torch.mean(batch, dim=0)
            batch_var = torch.var(batch, dim=0, unbiased=False)  # unbiased=False to mimic np.var
            batch_count = batch.shape[0]
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.var = (1 - self.momentum) * self.var + self.momentum * batch_var
            self.count += batch_count
        else:
            raise RuntimeError("Unsupported backend state.")

    def normalize(self, batch):
        # Initialize backend if not set yet
        if self.backend is None:
            self._init_backend(batch)
        if self.backend == "numpy":
            if not isinstance(batch, np.ndarray):
                raise TypeError("Expected numpy.ndarray for normalize, but got a different type.")
            return (batch - self.mean) / (np.sqrt(self.var) + self.epsilon)
        elif self.backend == "torch":
            if not isinstance(batch, torch.Tensor):
                raise TypeError("Expected torch.Tensor for normalize, but got a different type.")
            return (batch - self.mean) / (torch.sqrt(self.var) + self.epsilon)
        else:
            raise RuntimeError("Unsupported backend state.")