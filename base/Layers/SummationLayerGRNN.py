import numpy as np


class SummationLayerGRNN:

    def __init__(self):
        pass

    def forward(self, inputs, y):
        valid_mask = np.isfinite(inputs) & np.isfinite(y)

        filtered_inputs = inputs[valid_mask]
        filtered_y = y[valid_mask]

        marginal = np.sum(filtered_inputs)
        weighted_y = np.sum(filtered_inputs * filtered_y)

        
        if marginal == 0 or np.isnan(marginal):
            raise ZeroDivisionError('The marginal probability is 0. Try set higher sigma or tau parameters')

        return weighted_y, marginal
