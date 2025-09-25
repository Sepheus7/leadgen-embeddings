from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from leadgen.config import TABULAR_PCA_COMPONENTS


@dataclass
class TabularEmbeddingArtifacts:
    scaler: StandardScaler
    pca: PCA


class TabularEmbedder:
    def __init__(self, n_components: int = TABULAR_PCA_COMPONENTS) -> None:
        self.scaler = StandardScaler()
        self.requested_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit(self, X: np.ndarray) -> None:
        Z = self.scaler.fit_transform(X)
        n_features = Z.shape[1]
        n_comp = min(self.requested_components, n_features)
        if getattr(self.pca, 'n_components', None) != n_comp:
            self.pca = PCA(n_components=n_comp, random_state=42)
        self.pca.fit(Z)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        E = self.pca.transform(Z)
        return E.astype(np.float32)

    def get_artifacts(self) -> TabularEmbeddingArtifacts:
        return TabularEmbeddingArtifacts(self.scaler, self.pca)

