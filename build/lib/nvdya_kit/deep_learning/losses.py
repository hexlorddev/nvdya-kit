"""
Custom Loss Function APIs for Deep Neural Networks.
"""

class CrossEntropyLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        print(f"Initializing CrossEntropyLoss with reduction: {reduction}")

    def compute(self, y_true, y_pred):
        # Placeholder for loss computation
        print(f"Computing CrossEntropyLoss for y_true: {y_true}, y_pred: {y_pred}")
        return {"loss": "sample loss value"}


class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        print(f"Initializing MSELoss with reduction: {reduction}")

    def compute(self, y_true, y_pred):
        # Placeholder for loss computation
        print(f"Computing MSELoss for y_true: {y_true}, y_pred: {y_pred}")
        return {"loss": "sample loss value"} 