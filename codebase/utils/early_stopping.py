import numpy as np

class EarlyStopping:
    """
    Early stopping to terminate training if the evaluation reward does not improve
    by at least min_delta for patience consecutive evaluations, or if the evaluation 
    reward is zero for zero_patience consecutive evaluations.
    """
    def __init__(self, patience=10, min_delta=1e-3, verbose=True, zero_patience=3):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.zero_patience = zero_patience
        self.best_score = -np.inf
        self.counter = 0
        self.zero_counter = 0
        self.early_stop = False

        print(f"EarlyStopping: Patience: {patience}, Min delta: {min_delta}, Zero patience: {zero_patience}")

    def step(self, current_score):
        # Check if the current evaluation reward is exactly zero.
        if current_score == 0:
            self.zero_counter += 1
            if self.verbose:
                print(f"EarlyStopping: Current score is zero. Zero counter: {self.zero_counter} out of {self.zero_patience}")
        else:
            self.zero_counter = 0

        # Check for improvement.
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0  # Reset counter if improvement is seen
            if self.verbose:
                print(f"EarlyStopping: Improved evaluation score to {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        
        # Early stop if either no improvement or zero reward for enough consecutive evaluations.
        if self.counter >= self.patience or self.zero_counter >= self.zero_patience:
            self.early_stop = True
        
        return self.early_stop
