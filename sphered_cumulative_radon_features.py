import torch
import torch.nn.functional as F
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


class TimeSeriesFeatureExtractor():
    def __init__(self, n_channels, n_timesteps, window_size=9):
        self.window_size = window_size
        self.n_resolutions = min(n_timesteps // window_size, 10)

    def extract(self, X):
        Xs = []
        for r in range(1, self.n_resolutions + 1):
            p = (r * self.window_size - r)//2
            window = F.unfold(torch.Tensor(X).unsqueeze(3),
                              (self.window_size, 1),
                              dilation=(r, 1),
                              padding=(p, 0))
            Xs.append(window)
        Xs = torch.cat(Xs, 1)
        return Xs


class CumulativeRadonFeatures(torch.nn.Module):
    def __init__(self, n_channels, n_projections=100, n_quantiles=20):
        self.n_channels = n_channels
        self.n_projections = n_projections
        self.n_quantiles = n_quantiles
        self.projections = torch.randn(self.n_projections,  self.n_channels, 1)

    def fit(self, X):
        a = F.conv1d(X, self.projections).permute((0, 2, 1)).reshape((-1, self.n_projections))
        self.min_vals = torch.quantile(a, 0.01, dim=0)
        self.max_vals = torch.quantile(a, 0.99, dim=0)

    def forward(self, X):
        a = F.conv1d(X, self.projections)
        cdf = torch.zeros((a.shape[0], a.shape[1], self.n_quantiles))
        for q in range(self.n_quantiles):
            threshold = self.min_vals + (self.max_vals - self.min_vals) * (q + 1) / (self.n_quantiles + 1)
            cdf[:, :, q] = (a < threshold.unsqueeze(0).unsqueeze(2)).float().mean(2)
        return cdf.reshape((X.shape[0], -1)).numpy()


class ZCA_Sphering():
    def __init__(self):
        pass

    def fit(self, X):
        cov = LedoitWolf().fit(X).covariance_
        u, s, vh = np.linalg.svd(cov, hermitian=True, full_matrices=True)
        self.W = np.matmul(np.diag(1/np.sqrt(s)), vh)
        self.mu = X.mean(0)[None, :]

    def transform(self, X):
        X_sph = np.matmul(X - self.mu, self.W.T)
        return X_sph


def anomaly_score(train, test, test_labels):
    # Extract basic feature from the time-series
    ts_extractor = TimeSeriesFeatureExtractor(train.shape[1], train.shape[2], 9)
    train_feats = ts_extractor.extract(train)
    test_feats = ts_extractor.extract(test)
    # Extract Cumulative-Radon features from the input set
    radon_extractor = CumulativeRadonFeatures(train_feats.shape[1])
    radon_extractor.fit(train_feats)
    train_radon = radon_extractor.forward(train_feats)
    test_radon = radon_extractor.forward(test_feats)
    # Sphere the Radon features
    zca_sphere = ZCA_Sphering()
    zca_sphere.fit(train_radon)
    test_sph = zca_sphere.transform(test_radon)
    # Simple distance to center anomaly detection
    auc = roc_auc_score(test_labels, np.power(test_sph, 2).sum(1))
    return auc
