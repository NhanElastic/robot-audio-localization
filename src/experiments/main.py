import numpy as np

from doa_localization.doa_simulator import DOASimulator
from doa_localization.doa_estimator import DOAEstimator

from experiments.visual import plot_doa_3d

def create_3d_microphone_array(
    radius: float = 0.15,
) -> np.ndarray:
    directions = np.array([
        [ 1,  1,  1, -1, -1,  1, -1, -1],
        [ 1, -1,  1,  1, -1, -1,  1, -1],
        [ 1, -1, -1, -1,  1,  1,  1, -1],
    ], dtype=float)

    directions /= np.linalg.norm(directions, axis=0)
    return radius * directions


def angular_error_deg(
    estimated: float,
    actual: float,
) -> float:
    difference = np.rad2deg(estimated - actual)
    return abs((difference + 180) % 360 - 180)


def main():
    fs = 16000
    duration = 2.0
    n_samples = int(fs * duration)

    rng = np.random.default_rng(42)
    source_signal = rng.standard_normal(n_samples)

    mic_positions = create_3d_microphone_array(radius=0.15)

    simulator = DOASimulator(
        mic_positions=mic_positions,
        fs=fs,
        room_size=(10.0, 10.0, 10.0),
        max_order=5,
    )

    estimator = DOAEstimator(
        mic_positions=mic_positions,
        fs=fs,
        nfft=512,
        algorithm="MUSIC",
        n_grids=600,
    )

    test_cases = [
        (0, 0),
        (45, 15),
        (120, 30),
        (225, -20),
    ]

    visualization_results = []

    for azimuth_deg, elevation_deg in test_cases:
        true_azimuth = np.deg2rad(azimuth_deg)
        true_elevation = np.deg2rad(elevation_deg)

        microphone_signals = simulator.simulate(
            source_signal=source_signal,
            source_azimuth=true_azimuth,
            source_elevation=true_elevation,
            distance=2.0,
        )

        estimated_azimuth, estimated_elevation = estimator.estimate(
            microphone_signals
        )

        estimated_azimuth_deg = np.rad2deg(estimated_azimuth) % 360
        estimated_elevation_deg = np.rad2deg(estimated_elevation)

        azimuth_error = angular_error_deg(
            estimated_azimuth,
            true_azimuth,
        )
        elevation_error = abs(
            estimated_elevation_deg - elevation_deg
        )

        print(
            f"True: ({azimuth_deg:6.1f}°, {elevation_deg:6.1f}°) | "
            f"Estimated: ({estimated_azimuth_deg:6.1f}°, "
            f"{estimated_elevation_deg:6.1f}°) | "
            f"Errors: ({azimuth_error:.1f}°, "
            f"{elevation_error:.1f}°)"
        )

        visualization_results.append(
            (
                true_azimuth,
                true_elevation,
                estimated_azimuth,
                estimated_elevation,
            )
        )

    plot_doa_3d(mic_positions, visualization_results)

main()