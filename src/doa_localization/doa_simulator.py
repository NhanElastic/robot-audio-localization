import numpy as np
import pyroomacoustics as pra


class DOASimulator:
    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: int,
        room_size: tuple[float, float, float] = (10.0, 10.0, 10.0),
        max_order: int = 1,
    ):
        self.mic_positions = np.asarray(mic_positions, dtype=float)
        self.fs = fs
        self.room_size = np.asarray(room_size, dtype=float)
        self.max_order = max_order

        if self.mic_positions.ndim != 2:
            raise ValueError("mic_positions must have shape (3, n_mics)")

        if self.mic_positions.shape[0] != 3:
            raise ValueError("This simulator currently supports 3D only")

        if self.mic_positions.shape[1] < 4:
            raise ValueError("At least 4 microphones are recommended")

        if self.room_size.shape != (3,) or np.any(self.room_size <= 0):
            raise ValueError("room_size must contain three positive dimensions")

        if fs <= 0:
            raise ValueError("fs must be positive")

        if max_order < 0:
            raise ValueError("max_order must be non-negative")

        self.room_center = self.room_size / 2
        room_mic_positions = (
            self.room_center[:, np.newaxis] + self.mic_positions
        )

        if np.any(room_mic_positions < 0) or np.any(
            room_mic_positions > self.room_size[:, np.newaxis]
        ):
            raise ValueError("Microphone positions must fit inside room_size")

    def simulate(
        self,
        source_signal: np.ndarray,
        source_azimuth: float,
        source_elevation: float = 0.0,
        distance: float = 0.1,
    ) -> np.ndarray:
        source_signal = np.asarray(source_signal, dtype=float)

        if source_signal.ndim != 1 or source_signal.size == 0:
            raise ValueError("source_signal must be a non-empty 1D array")

        if distance <= 0:
            raise ValueError("distance must be positive")

        room = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            max_order=self.max_order,
            materials=pra.Material(energy_absorption=0.3)
        )

        direction = np.array([
            np.cos(source_elevation) * np.cos(source_azimuth),
            np.cos(source_elevation) * np.sin(source_azimuth),
            np.sin(source_elevation),
        ])

        source_position = self.room_center + distance * direction
        
        if np.any(source_position < 0) or np.any(source_position > self.room_size):
            raise ValueError("Source position must be inside room_size")

        room.add_source(
            source_position,
            signal=source_signal,
        )
        room.add_microphone_array(self.room_center[:, np.newaxis] + self.mic_positions)
        room.simulate()

        return room.mic_array.signals.copy()
