import numpy as np
import pyroomacoustics as pra


class DOASimulator:
    def __init__(
        self,
        mic_positions: np.ndarray,
        fs: int,
        room_size: tuple[float, float] = (10.0, 10.0),
        max_order: int = 1,
    ):
        self.mic_positions = np.asarray(mic_positions, dtype=float)
        self.fs = fs
        self.room_size = np.asarray(room_size, dtype=float)
        self.max_order = max_order

        if self.mic_positions.ndim != 2:
            raise ValueError(
                "mic_positions must have shape (dimension, n_mics)"
            )

        if self.mic_positions.shape[0] != 2:
            raise ValueError("This simulator currently supports 2D only")

        if self.room_size.shape != (2,) or np.any(self.room_size <= 0):
            raise ValueError("room_size must contain two positive dimensions")

        if max_order < 0:
            raise ValueError("max_order must be non-negative")

        room_center = self.room_size / 2
        room_mic_positions = room_center[:, np.newaxis] + self.mic_positions
        if np.any(room_mic_positions < 0) or np.any(
            room_mic_positions > self.room_size[:, np.newaxis]
        ):
            raise ValueError("Microphone positions must fit inside room_size")

    def simulate(
        self,
        source_signal: np.ndarray,
        source_azimuth: float,
        distance: float = 0.1,
    ) -> np.ndarray:
        source_signal = np.asarray(source_signal, dtype=float)

        if source_signal.ndim != 1:
            raise ValueError("source_signal must be a 1D array")

        if source_signal.size == 0:
            raise ValueError("source_signal must not be empty")

        if distance <= 0:
            raise ValueError("distance must be positive")

        room = pra.ShoeBox(
            self.room_size,
            fs=self.fs,
            max_order=self.max_order,
        )

        room_center = self.room_size / 2
        source_position = room_center + distance * np.array(
            [
                np.cos(source_azimuth),
                np.sin(source_azimuth),
            ]
        )
        if np.any(source_position < 0) or np.any(source_position > self.room_size):
            raise ValueError("Source position must be inside room_size")

        room.add_source(
            source_position,
            signal=source_signal,
        )
        room.add_microphone_array(room_center[:, np.newaxis] + self.mic_positions)
        room.simulate()

        return room.mic_array.signals.copy()
