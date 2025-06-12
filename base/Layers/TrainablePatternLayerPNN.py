import torch
import torch.nn as nn
import torch.optim as optim


class TrainablePatternLayerPNN(nn.Module):

    def __init__(self, sigma, tau, regularization: str, n_classes):
        super().__init__()
        self.sigma = sigma
        self.tau = tau
        self.regularization = regularization
        self.n_classes = n_classes

    def fit(self, X, y, epochs=1000, lr=0.001):
        self.W = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)
        self.priors = torch.bincount(self.y_train) / len(y)

        n_samples = self.W.shape[0]
        self.beta_logits = nn.Parameter(torch.randn(n_samples))

        with torch.no_grad():
            self.initial_beta_logits = self.beta_logits.detach().cpu()
            self.initial_beta = torch.softmax(self.initial_beta_logits, dim=0).detach().cpu().numpy().flatten()
            self.initial_beta_logits = self.beta_logits.detach().cpu().numpy().flatten()

        optimizer = optim.Adam([self.beta_logits], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = self._loss(self.W, self.y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            self.beta = torch.softmax(self.beta_logits, dim=0)

    def _pdf(self, X, W):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if not torch.is_tensor(W):
            W = torch.tensor(W, dtype=torch.float32)
        dists = torch.cdist(X, W)
        coef = 1 / (torch.sqrt(torch.tensor(2 * torch.pi)) * self.sigma)
        return coef * torch.exp(-dists**2 / (2 * self.sigma**2))

    def _log_likelihood(self, X, y):

        X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        y = torch.tensor(y, dtype=torch.long) if not torch.is_tensor(y) else y

        n = X.shape[0]
        W = self.W
        y_train = self.y_train
        beta = torch.softmax(self.beta_logits, dim=0)
        priors = self.priors

        N_x = self._pdf(X, W)

        mask_not_i = ~torch.eye(n, dtype=torch.bool, device=X.device)

        y_i_eq_yj = y[:, None] == y_train[None, :] 
        priors_yi = priors[y]

        beta_exp = beta.unsqueeze(0)

        numer_mask = y_i_eq_yj & mask_not_i
        numer = (beta_exp * N_x * numer_mask * priors_yi[:, None]).sum(dim=1)

        denom = torch.zeros_like(numer)
        for k in range(self.priors.shape[0]):
            class_mask = (y_train == k)
            class_mask_exp = class_mask.unsqueeze(0).expand(n, -1)
            denom += (beta_exp * N_x * class_mask_exp * mask_not_i * priors[k]).sum(dim=1)

        numer = torch.clamp(numer, min=1e-8)
        denom = torch.clamp(denom, min=1e-8)
        return torch.sum(torch.log(numer / denom))

    def _prior(self):
        self.beta_logits
        if self.regularization == 'l2':
            return -torch.sum(self.beta_logits**2) / (2 * self.tau**2)
        elif self.regularization == 'l1':
            return -torch.sum(torch.abs(self.beta_logits)) / self.tau
        return 0.0

    def _loss(self, X, y):
        likelihood = self._log_likelihood(X, y)
        prior = self._prior()
        return -(likelihood + prior)

    def forward(self, input):

        with torch.no_grad():
            if not torch.is_tensor(input):
                input = torch.tensor(input, dtype=torch.float32)
            if input.ndim == 1:
                input = input.unsqueeze(0)

            W = self.W
            y = self.y_train
            beta = self.beta

            weighted_N_values_list = []
            y_list = []

            for class_label in torch.unique(y):
                class_mask = (y == class_label)
                W_class = W[class_mask]
                b_class = beta[class_mask]
                weigherd_N_values = self._pdf(input, W_class) * b_class
                weighted_N_values_list.append(weigherd_N_values)
                y_list.append(class_label.repeat(weigherd_N_values.shape[1]))

            weigherd_N = torch.cat(weighted_N_values_list, dim=1)
            y_values = torch.cat(y_list)
            return (
                weigherd_N.detach().cpu().numpy()[0],
                y_values.detach().cpu().numpy(),
                beta.detach().cpu().numpy()
            )
