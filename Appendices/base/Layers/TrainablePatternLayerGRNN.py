import torch
import torch.nn as nn
import torch.optim as optim


class TrainablePatternLayerGRNN(nn.Module):

    def __init__(self, sigma, tau, regularization: str):
        super().__init__()
        self.sigma = sigma
        self.tau = tau
        self.regularization = regularization

    def fit(self, X, y, epochs=1000, lr=0.001):
        self.W = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
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
        W = self.W

        y = torch.tensor(y, dtype=torch.float32) if not torch.is_tensor(y) else y
        y = y.reshape(-1, 1)

        y_train = self.y_train.reshape(-1, 1)

        n = X.shape[0]
        mask_not_i = ~torch.eye(n, dtype=torch.bool, device=X.device)

        beta = torch.softmax(self.beta_logits, dim=0)

        N_x = self._pdf(X, W)
        N_y = self._pdf(y, y_train)

        numer = (beta * N_x * N_y * mask_not_i).sum(dim=1)
        denom = (beta * N_x * mask_not_i).sum(dim=1)

        numer = torch.clamp(numer, min=1e-8)
        denom = torch.clamp(denom, min=1e-8)

        return torch.sum(torch.log(numer / denom))

    def _prior(self):
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
            beta = self.beta * 1e10

            N = self._pdf(input, W)
            weighted_N = N * beta 

            return (
                weighted_N.detach().cpu().numpy()[0],
                y.detach().cpu().numpy(),
                beta.detach().cpu().numpy()
            )
