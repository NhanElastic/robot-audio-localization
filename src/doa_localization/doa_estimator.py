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
    ):
        self.mic_positions = mic_positions
        self.fs = fs
        self.nfft = nfft
        self.algorithm = algorithm.upper()

    def estimate(self, signals: np.ndarray) -> float:
        n_mics = signals.shape[0]

        if signals.ndim != 2:
            raise ValueError("Signals must have shape (n_mics, n_samples)")

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
        return float(estimate_azimuth)

    def _create_doa_algorithm(self):
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
        )