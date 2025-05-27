"""
Custom Layer APIs for Deep Neural Networks.
"""

class DenseLayer:
    def __init__(self, units, activation='relu'):
        self.units = units
        self.activation = activation
        print(f"Initializing Dense layer with {units} units and activation {activation}")

    def build(self, input_shape):
        # Placeholder for layer building logic
        print(f"Building Dense layer with input shape: {input_shape}")
        return self

    def call(self, inputs):
        # Placeholder for forward pass
        print(f"Calling Dense layer with inputs: {inputs}")
        return {"output": "sample output"}


class ConvLayer:
    def __init__(self, filters, kernel_size, activation='relu'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        print(f"Initializing Conv layer with {filters} filters, kernel size {kernel_size}, and activation {activation}")

    def build(self, input_shape):
        # Placeholder for layer building logic
        print(f"Building Conv layer with input shape: {input_shape}")
        return self

    def call(self, inputs):
        # Placeholder for forward pass
        print(f"Calling Conv layer with inputs: {inputs}")
        return {"output": "sample output"} 