import matplotlib.pyplot as plt
import numpy as np


def plot_doa_3d(mic_positions, results):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Microphones
    ax.scatter(
        *mic_positions,
        color="blue",
        s=60,
        label="Microphones",
    )

    for i in range(mic_positions.shape[1]):
        ax.text(*mic_positions[:, i], f" M{i + 1}", fontsize=8)

    # Array center
    ax.scatter(0, 0, 0, color="black", marker="x", s=80,
               label="Array center")

    colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown"]

    for i, result in enumerate(results):
        true_az, true_el, est_az, est_el = result
        color = colors[i % len(colors)]

        def direction(az, el):
            return np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ])

        true_vec = direction(true_az, true_el)
        est_vec = direction(est_az, est_el)

        # True direction: solid arrow
        ax.quiver(
            0, 0, 0, *true_vec,
            color=color, length=1.0, normalize=True,
            arrow_length_ratio=0.12,
            label=f"True {i + 1}",
        )

        # Estimated direction: dashed line
        ax.plot(
            [0, est_vec[0]],
            [0, est_vec[1]],
            [0, est_vec[2]],
            linestyle="--",
            color=color,
            label=f"Estimated {i + 1}",
        )

    limit = 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D DOA: All Test Cases")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()