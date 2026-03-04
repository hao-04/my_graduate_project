"""
骨骼步态可视化脚本
==================
随机抽取若干 trial，生成 3D 骨骼动画 GIF，
坐标轴标注为 dim0 / dim1 / dim2，供确认轴向含义。

用法:
    python utils/visualize_skeleton.py
"""

import pickle
import random
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---- 骨骼连接 (与 model/gcn_bert.py 一致) ----
SKELETON_EDGES = [
    (0, 1), (0, 2), (2, 3), (1, 4), (1, 16), (4, 16),
    (2, 5), (5, 6), (6, 7), (7, 8), (7, 9),
    (4, 10), (10, 11), (11, 12), (12, 13), (13, 14), (13, 15),
    (2, 17), (17, 18), (18, 19), (19, 20), (19, 21),
    (16, 22), (22, 23), (23, 24), (24, 25), (25, 26), (25, 27),
]

MARKER_NAMES = [
    "C7", "T10", "CLAV", "STRN",
    "L_ASI", "L_SHO", "L_ELB", "L_WRA", "L_WRB", "L_FIN",
    "L_THI", "L_KNE", "L_TIB", "L_ANK", "L_HEE", "L_TOE",
    "R_ASI", "R_SHO", "R_ELB", "R_WRA", "R_WRB", "R_FIN",
    "R_THI", "R_KNE", "R_TIB", "R_ANK", "R_HEE", "R_TOE",
]


def load_random_trials(pkl_path: str, n_trials: int = 1, seed: int = None):
    """从 pkl 文件中随机抽取 trial，返回 (frames, marker_names, info_str)"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if seed is not None:
        random.seed(seed)

    results = []
    n_subjects = len(data["data"])
    for _ in range(n_trials):
        subj_idx = random.randint(0, n_subjects - 1)
        person_data = data["data"][subj_idx]  # (28, n_trials_i, 1001, 3)
        n_t = person_data.shape[1]
        trial_idx = random.randint(0, n_t - 1)
        # (28, 1001, 3) → (1001, 28, 3)
        frames = person_data[:, trial_idx, :, :].transpose(1, 0, 2)
        info = f"subj={subj_idx}_trial={trial_idx}"
        results.append((frames, data["marker_names"], info))

    return results


def make_skeleton_gif(
    frames: np.ndarray,
    marker_names: list,
    output_path: str,
    title: str = "",
    fps: int = 30,
    step: int = 5,
    elev: float = 15,
    azim: float = -60,
):
    """
    生成 3D 骨骼动画 GIF。

    frames: (F, 28, 3)  — 原始坐标
    step: 每隔 step 帧取一帧（加速）
    """
    frames_sub = frames[::step]
    n_frames = len(frames_sub)

    # 计算全局坐标范围
    all_coords = frames_sub.reshape(-1, 3)
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    centers = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2 * 1.1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        ax.cla()
        skeleton = frames_sub[frame_idx]  # (28, 3)

        # 画关节点
        ax.scatter(
            skeleton[:, 0], skeleton[:, 1], skeleton[:, 2],
            c="red", s=20, depthshade=True,
        )

        # 画骨骼连线
        for i, j in SKELETON_EDGES:
            ax.plot(
                [skeleton[i, 0], skeleton[j, 0]],
                [skeleton[i, 1], skeleton[j, 1]],
                [skeleton[i, 2], skeleton[j, 2]],
                c="steelblue", linewidth=1.5,
            )

        # 标注几个关键点名称
        for idx in [0, 2, 4, 11, 14, 15, 16, 23, 26, 27]:
            ax.text(
                skeleton[idx, 0], skeleton[idx, 1], skeleton[idx, 2],
                f" {marker_names[idx]}", fontsize=6, color="black",
            )

        ax.set_xlabel("dim0", fontsize=14, fontweight="bold", labelpad=10)
        ax.set_ylabel("dim1", fontsize=14, fontweight="bold", labelpad=10)
        ax.set_zlabel("dim2", fontsize=14, fontweight="bold", labelpad=10)

        ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
        ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
        ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(
            f"{title}  frame {frame_idx * step}/{len(frames)}",
            fontsize=12,
        )

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"已保存: {output_path}  ({n_frames} 帧)")


def main():
    out_dir = Path("output/visualize")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    count = 0

    for pkl_name, label in [
        ("data/normal_LsideSegm_28markers.pkl", "normal"),
        ("data/stroke_NsideSegm_28markers.pkl", "stroke"),
    ]:
        trials = load_random_trials(pkl_name, n_trials=1, seed=seed + count)
        for frames, mnames, info in trials:
            fname = f"{label}_{info}.gif"
            make_skeleton_gif(
                frames,
                mnames,
                str(out_dir / fname),
                title=f"{label} {info}",
                fps=20,
                step=5,
            )
            count += 1

    # 额外生成一个不同视角的，方便从多角度确认
    trials = load_random_trials(
        "data/normal_LsideSegm_28markers.pkl", n_trials=1, seed=99
    )
    for frames, mnames, info in trials:
        make_skeleton_gif(
            frames,
            mnames,
            str(out_dir / f"normal_{info}_topview.gif"),
            title=f"normal {info} (top view)",
            fps=20,
            step=5,
            elev=80,
            azim=-90,
        )

    print(f"\n所有动画已保存到 {out_dir}/")


if __name__ == "__main__":
    main()
