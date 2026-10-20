import numpy as np
import pyroomacoustics as pra

DOA_ALGORITHMS = {
    "MUSIC": pra.doa.MUSIC,
    "SRP": pra.doa.SRP,
    "NORMMUSIC": pra.doa.NormMUSIC,
}

class DOAEstimator:
    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: int = 16000,
        nfft: int = 512,
        algorithm: str = "MUSIC",
        n_grids: int = 600
    ):
        self.mic_positions = mic_positions
        self.fs = fs
        self.nfft = nfft
        self.algorithm = algorithm.upper()
        self.n_grids = n_grids

        if self.mic_positions.ndim != 2:
            raise ValueError("mic_positions must be a 2D array")

        if self.mic_positions.shape[0] != 3:
            raise ValueError("mic_positions must have shape (3, n_mics)")

        if self.mic_positions.shape[1] < 4:
            raise ValueError("At least 4 microphones are recommended")

        if self.algorithm not in DOA_ALGORITHMS:
            raise ValueError(
                f"Unsupported DOA algorithm: {self.algorithm}"
            )

    def estimate(self, signals: np.ndarray) -> tuple[float, float]:
        signals = np.asarray(signals, dtype=float)
        n_mics, n_samples = signals.shape

        if signals.ndim != 2:
            raise ValueError("Signals must have shape (n_mics, n_samples)")

        if n_samples < self.nfft:
            raise ValueError("Signals are shorter than nfft")

        if self.mic_positions.shape[1] != n_mics:
            raise ValueError("Microphone positions and signals do not match")

        if self.algorithm not in {"MUSIC", "SRP", "NORMMUSIC"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        microphone_stft = np.stack([
            pra.transform.stft.analysis(
                signal.astype(float),
                self.nfft,
                self.nfft // 2
            ).T
            for signal in signals
        ])

        doa_algorithm = self._create_doa_algorithm()

        doa_algorithm.locate_sources(
            microphone_stft,
            freq_range = [300, 4000]
        )

        estimate_azimuth = doa_algorithm.azimuth_recon[0]
        estimate_colatitude = doa_algorithm.colatitude_recon[0]

        estimate_elevation = np.pi / 2 - estimate_colatitude

        return estimate_azimuth, estimate_elevation

    def _create_doa_algorithm(self) -> pra.doa.DOA:
        algorithm_class = DOA_ALGORITHMS.get(self.algorithm)

        if algorithm_class is None:
            raise ValueError(
                f"Unsupported DOA algorithm: {self.algorithm}"
            )

        return algorithm_class(
            self.mic_positions,
            self.fs,
            self.nfft,
            c=343.0,
            dim=3,
            n_grids=self.n_grids 
        )