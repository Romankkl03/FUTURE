import torch

from recurrence.pdist import torch_pdist


def _as_2d_time_series(time_series: torch.Tensor) -> torch.Tensor:
    if not isinstance(time_series, torch.Tensor):
        time_series = torch.as_tensor(time_series)
    time_series = time_series.float().squeeze()
    if time_series.ndim == 1:
        return time_series.unsqueeze(0)
    if time_series.ndim == 2:
        return time_series
    raise ValueError(
        "time_series must be 1D or 2D after squeeze, "
        f"got shape={tuple(time_series.shape)}"
    )


class TorchTSTransformer:
    def __init__(
        self,
        time_series: torch.Tensor,
        rec_metric: str = "cosine",
        p: float = 3,
        device: torch.device | None = None,
    ):
        """
        Transformer for recurrence matrix generation.
        Args:
            time_series: torch.Tensor of shape (features, timesteps)
            rec_metric: one of ['cosine', 'euclidean', 'canberra']
        """
        self.time_series = _as_2d_time_series(time_series)
        self.device = device or self.time_series.device
        self.time_series = self.time_series.to(self.device)
        self.recurrence_matrix = None
        self.threshold_baseline = [0.95, 0.7]
        self.min_signal_ratio = 0.6
        self.max_signal_ratio = 0.85
        self.rec_metric = rec_metric
        self.p = p

    def ts_to_recurrence_matrix(self, threshold: float | None = None) -> torch.Tensor:
        distance_matrix = torch_pdist(self.time_series.T, metric=self.rec_metric, p=self.p)
        distance_matrix = 1 - (distance_matrix / distance_matrix.max().clamp_min(1e-12))
        self.recurrence_matrix = self.binarization(distance_matrix, threshold)
        return self.recurrence_matrix

    def binarization(
        self,
        distance_matrix: torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        if threshold is not None:
            return (distance_matrix >= threshold).float()
        ths = torch.tensor(self.threshold_baseline,
                           device=distance_matrix.device,
                           dtype=distance_matrix.dtype)
        tmp = (distance_matrix.unsqueeze(0) >= ths[:, None, None]).float()
        signal_ratio = torch.mean((tmp == 0).float(), dim=(1, 2))

        mask = (signal_ratio > float(self.min_signal_ratio)) & (signal_ratio < float(self.max_signal_ratio))
        if mask.any():
            neg_one = torch.tensor(-1.0,
                                   device=distance_matrix.device,
                                   dtype=signal_ratio.dtype)
            candidate_scores = torch.where(mask, signal_ratio, neg_one)
            best_idx = torch.argmax(candidate_scores)
            return tmp[best_idx].float()
        else:
            return tmp[0].float()

    def _colorise_torch(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.to(torch.float64)
        min_val = matrix.min()
        max_val = matrix.max()
        normalized = (matrix - min_val) / (max_val - min_val + 1e-8)
        rounded = torch.floor(normalized * 255 + 0.5)
        return rounded.to(torch.uint8)

    def ts_to_3d_recurrence_matrix(self) -> torch.Tensor:
        cos = torch_pdist(self.time_series.T, metric='cosine')
        euc = torch_pdist(self.time_series.T, metric='euclidean')
        can = torch_pdist(self.time_series.T, metric='canberra')
        colorised = [self._colorise_torch(m) for m in [cos, euc, can]]
        stacked = torch.stack(colorised, dim=0)
        return stacked
