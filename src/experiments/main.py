import numpy as np

from doa_localization.doa_simulator import DOASimulator
from doa_localization.doa_estimator import DOAEstimator

def main():
    fs = 16000
    duration = 2.0
    n_samples = int(fs * duration)

    rng = np.random.default_rng(42)
    source_signal = rng.standard_normal(n_samples)

    # 3. Create an 8-microphone circular array
    angles = np.arange(8) * 2 * np.pi / 8

    mic_positions = np.asarray([
        0.15 * np.cos(angles),
        0.15 * np.sin(angles),
    ])

    simulator = DOASimulator(
        mic_positions=mic_positions,
        fs=fs,
        room_size=(10.0, 10.0),
        max_order=100,
    )

    true_azimuth = np.deg2rad(191)

    microphone_signals = simulator.simulate(
        source_signal=source_signal,
        source_azimuth=true_azimuth,
        distance=2.0,
    )

    estimator = DOAEstimator(
        mic_positions = mic_positions,
        fs = fs
    )

    estimated_azimuth = estimator.estimate(microphone_signals)

    print(f"Microphone signals shape: {microphone_signals.shape}")
    print(f"Number of microphones: {microphone_signals.shape[0]}")
    print(f"Input samples: {n_samples}")
    print(f"True direction: {np.rad2deg(true_azimuth):.1f} degrees")
    print(f"Estimated direction: {np.rad2deg(estimated_azimuth):.1f} degrees")

    error = abs(
    (np.rad2deg(estimated_azimuth - true_azimuth) + 180) % 360 - 180
)

    print(f"Angular error: {error:.2f} degrees")

main()
