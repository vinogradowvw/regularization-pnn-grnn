import numpy as np
from Distance.distance import distance_l1, distance_l2
from Kernels.GaussianKernel import GaussianKernel
from Kernels.LaplaceKernel import LaplaceKernel


class RegularizationLayerPNN():

    def __init__(self, classes, regularization_type: str, tau: float, distance_metric: str):
        self.__classes = classes
        self.__distance_metric = distance_metric
        if regularization_type == 'l1':
            self.__prior_kernel = LaplaceKernel(tau)
        if regularization_type == 'l2':
            self.__prior_kernel = GaussianKernel(tau)
        self.distances = {}
        self.__tau = tau

    def fit(self, X, y):
        """Fitting the train data to the Regularization layer."""
        # computing distances from each pattern to other patterns
        for c in self.__classes:
            self.distances[c] = []
            class_mask = (y == c)
            X_class = X[class_mask]

            if self.__distance_metric == 'l1':
                for i in range(len(X_class)):
                    D_i = []
                    for j in range(len(X_class)):
                        D_i.append(distance_l1(X_class[i], X_class[j]))
                    self.distances[c].append(D_i)

            elif self.__distance_metric == 'l2':
                for i in range(len(X_class)):
                    D_i = []
                    for j in range(len(X_class)):
                        D_i.append(distance_l2(X[i], X[j]))
                    self.distances[c].append(D_i)

    def __compute_prior_on_distance(self, distance, c):
        pattern_distances = []
        for i in range(len(self.distances[c])):
            for j in range(len(self.distances[c][0])):
                pattern_distances.append(self.distances[c][i][j])
        return self.__prior_kernel(distance, pattern_distances)

    def forward(self, pattern_kernels, y, distances):
        priors = []
        for c in self.__classes:
            class_mask = (y == c)
            distances_class = distances[class_mask]
            prior = np.array([self.__compute_prior_on_distance(d, c) for d in distances_class])
            pattern_kernels[class_mask] *= prior
            priors.extend(prior)
        return pattern_kernels
