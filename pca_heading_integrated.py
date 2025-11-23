#!/usr/bin/env python3
"""
RMPCA 기반 실시간 헤딩 시각화 도구.

센서 로그(CSV)의 쿼터니언(rot_x~rot_w)과 선형 가속도(linear_x~linear_z)를
이용해 글로벌 좌표계의 XY 평면에서 이동 방향을 추정한다.
현재 구현은 “두 스텝” 단위 윈도우(기본 2 step interval)를 기준으로
Rolling PCA를 수행하고, 헤딩을 실시간 애니메이션으로 표현한다.

스크립트 하단 CONFIG 딕셔너리를 수정해서 파라미터를 제어한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


CONFIG = {
    # "CSV_PATH": "/Users/yundong-geun/Desktop/SJmama/logs/gca/sensor_data_2025_11_20_19_18.csv",
    "CSV_PATH": "/Users/yundong-geun/Desktop/SJmama/logs/largesquare.csv",
    "STEPS_PER_WINDOW": 1,   # 윈도우에 포함할 step interval 개수
    "STEP_STRIDE": 1,        # 슬라이딩 시 이동할 step interval 수
    "MIN_SAMPLES": 25,
    "LOWPASS_ALPHA": 0.05,
    "HEADING_SMOOTH_ALPHA": 0.4,
    "INTERVAL_MS": 45,
    "SUBSAMPLE": 1,
    "START_TIME": None,
    "END_TIME": None,
    "MAX_ROWS": None,
    "EXPORT_CSV": False,
    "CSV_OUT": None,
    "SHOW_ANIMATION": True,
}


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """쿼터니언([x, y, z, w])을 3x3 회전 행렬로 변환."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def exponential_lowpass(data: np.ndarray, alpha: float) -> np.ndarray:
    """간단한 1차 IIR 로우패스 (NaN은 직전 값으로 대체)."""
    if alpha <= 0 or len(data) == 0:
        return data
    smoothed = np.zeros_like(data)
    smoothed[0] = np.where(np.isfinite(data[0]), data[0], 0.0)
    for i in range(1, len(data)):
        prev = smoothed[i - 1]
        curr = np.where(np.isfinite(data[i]), data[i], prev)
        smoothed[i] = alpha * curr + (1.0 - alpha) * prev
    return smoothed


# --- Heading smoothing helper (wrap-around aware) ---
def smooth_heading_deg(angles_deg: np.ndarray, alpha: float) -> np.ndarray:
    """각도(deg) 시퀀스를 wrap-around를 고려하여 지수 스무딩."""
    if alpha <= 0 or angles_deg.size == 0:
        return angles_deg
    ang_rad = np.deg2rad(angles_deg)
    cos_vals = np.cos(ang_rad)
    sin_vals = np.sin(ang_rad)
    cos_s = exponential_lowpass(cos_vals, alpha)
    sin_s = exponential_lowpass(sin_vals, alpha)
    smoothed_rad = np.arctan2(sin_s, cos_s)
    smoothed_deg = (np.rad2deg(smoothed_rad) + 360.0) % 360.0
    return smoothed_deg


def parse_bool_series(series: pd.Series) -> pd.Series:
    """다양한 타입의 isstep 컬럼을 bool 시리즈로 변환."""
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return num != 0
    tokens = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "y", "yes"}
    return tokens.isin(truthy)


class RMPCAHeadingAnimator:
    """Rolling PCA 기반 헤딩 추정 + 애니메이션."""

    def __init__(
        self,
        csv_path: Path,
        steps_per_window: int = 2,
        step_stride: int = 1,
        min_samples: int = 25,
        lowpass_alpha: float = 0.05,
        subsample: int = 2,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_rows: Optional[int] = None,
        heading_smooth_alpha: float = 0.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.steps_per_window = max(1, int(steps_per_window))
        self.step_stride = max(1, int(step_stride))
        self.min_samples = int(min_samples)
        self.lowpass_alpha = float(lowpass_alpha)
        self.subsample = max(1, int(subsample))
        self.start_time = start_time
        self.end_time = end_time
        self.max_rows = max_rows
        self.heading_smooth_alpha = float(heading_smooth_alpha)

        self.df = self._load_dataframe()
        self._compute_world_linear()
        self.valid_df = self._build_valid_dataframe()
        if self.valid_df.empty:
            raise SystemExit("Valid samples not found (check timestamp/linear/quaternion columns).")
        self.heading_df = self._compute_rmpca_heading()
        if self.heading_df.empty:
            raise SystemExit(
                "RMPCA heading results empty. Check isstep detections or lower MIN_SAMPLES/STEPS_PER_WINDOW."
            )

        # heading 스무딩 적용 (옵션)
        if self.heading_smooth_alpha > 0.0:
            raw_heading = self.heading_df["heading_deg"].to_numpy(dtype=float)
            smoothed_heading = smooth_heading_deg(raw_heading, self.heading_smooth_alpha)
            # 각도 원본 보존
            self.heading_df["heading_raw_deg"] = self.heading_df["heading_deg"]
            self.heading_df["heading_deg"] = smoothed_heading
            # 스무딩된 heading 기반으로 단위 방향 벡터도 생성
            sm_rad = np.deg2rad(smoothed_heading)
            self.heading_df["pc1_east_smooth"] = np.cos(sm_rad)
            self.heading_df["pc1_north_smooth"] = np.sin(sm_rad)

        # heading unwrap (for plotting용 연속 그래프)
        ang_rad = np.deg2rad(self.heading_df["heading_deg"].to_numpy(dtype=float))
        unwrapped_rad = np.unwrap(ang_rad)
        unwrapped_deg = np.rad2deg(unwrapped_rad)
        self.heading_df["heading_unwrapped_deg"] = unwrapped_deg

        self.xy_vectors = self.valid_df[["world_linear_x", "world_linear_y"]].to_numpy()
        norms = np.linalg.norm(self.xy_vectors, axis=1)
        finite_norms = norms[np.isfinite(norms)]
        percentile = np.percentile(finite_norms, 98) if finite_norms.size else 1.0
        self.xy_limit = max(percentile * 1.4, 0.5)
        self.arrow_scale = self.xy_limit * 0.9

    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "timestamp" not in df.columns:
            raise SystemExit("CSV must contain 'timestamp' column.")
        df = df.sort_values("timestamp").reset_index(drop=True)
        if self.start_time is not None:
            df = df[df["timestamp"] >= self.start_time]
        if self.end_time is not None:
            df = df[df["timestamp"] <= self.end_time]
        if self.max_rows is not None:
            df = df.iloc[: self.max_rows]
        return df.reset_index(drop=True)

    def _compute_world_linear(self) -> None:
        required = ["rot_x", "rot_y", "rot_z", "rot_w", "linear_x", "linear_y", "linear_z"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise SystemExit(f"CSV missing columns required for RMPCA: {missing}")

        quats = self.df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy(dtype=float)
        locals_lin = self.df[["linear_x", "linear_y", "linear_z"]].to_numpy(dtype=float)
        world = np.full_like(locals_lin, np.nan)

        for idx, (q, acc) in enumerate(zip(quats, locals_lin)):
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(acc)):
                continue
            R = quaternion_to_rotation_matrix(q)
            world[idx] = R @ acc

        world = exponential_lowpass(world, self.lowpass_alpha)
        self.df["world_linear_x"] = world[:, 0]
        self.df["world_linear_y"] = world[:, 1]
        self.df["world_linear_z"] = world[:, 2]

    def _build_valid_dataframe(self) -> pd.DataFrame:
        base_cols = ["timestamp", "world_linear_x", "world_linear_y"]
        mask = np.all(np.isfinite(self.df[base_cols].to_numpy()), axis=1)
        valid = self.df.loc[mask].copy()
        valid.reset_index(drop=False, inplace=True)
        valid.rename(columns={"index": "source_index"}, inplace=True)

        if "isstep" in valid.columns:
            valid["isstep_bool"] = parse_bool_series(valid["isstep"]).to_numpy()
        else:
            valid["isstep_bool"] = False

        if "label" not in valid.columns:
            valid["label"] = ""
        valid["label"] = valid["label"].fillna("").astype(str)
        return valid

    def _detect_step_edges(self, step_mask: np.ndarray) -> np.ndarray:
        """isstep Bool 시리즈에서 상승 에지(새로운 step) 위치를 찾는다."""
        if step_mask.size == 0:
            return np.array([], dtype=int)
        prev = np.concatenate(([False], step_mask[:-1]))
        edges = np.where(step_mask & ~prev)[0]
        if edges.size == 0:
            edges = np.where(step_mask)[0]
        return edges

    def _compute_rmpca_heading(self) -> pd.DataFrame:
        timestamps = self.valid_df["timestamp"].to_numpy(dtype=float)
        xy = self.valid_df[["world_linear_x", "world_linear_y"]].to_numpy(dtype=float)
        isstep_arr = self.valid_df["isstep_bool"].to_numpy(dtype=bool)
        labels_arr = self.valid_df["label"].to_numpy(dtype=object)

        step_edges = self._detect_step_edges(isstep_arr)
        if len(step_edges) < self.steps_per_window + 1:
            return pd.DataFrame()

        records = []
        last_vec = None

        for offset in range(0, len(step_edges) - self.steps_per_window, self.step_stride):
            start_idx = step_edges[offset]
            end_idx = step_edges[offset + self.steps_per_window]
            if end_idx <= start_idx:
                continue

            window = xy[start_idx : end_idx + 1]
            window_len = len(window)
            if window_len < self.min_samples:
                continue

            centered = window - window.mean(axis=0, keepdims=True)
            cov = np.cov(centered, rowvar=False)
            if not np.isfinite(cov).all():
                continue

            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            if vals[0] <= 1e-8 or vals.sum() <= 0:
                continue

            first_vec = vecs[:, 0]
            norm = np.linalg.norm(first_vec)
            if norm == 0:
                continue
            first_vec = first_vec / norm
            if last_vec is not None and np.dot(first_vec, last_vec) < 0:
                first_vec = -first_vec

            heading_rad = np.arctan2(first_vec[1], first_vec[0])
            heading_deg = (np.degrees(heading_rad) + 360.0) % 360.0
            first_evr = vals[0] / vals.sum()

            records.append(
                {
                    "timestamp": float(timestamps[end_idx]),
                    "heading_deg": float(heading_deg),
                    "pc1_east": float(first_vec[0]),
                    "pc1_north": float(first_vec[1]),
                    "first_var": float(vals[0]),
                    "first_evr": float(first_evr),
                    "samples": int(window_len),
                    "window_start": int(start_idx),
                    "window_end": int(end_idx),
                    "isstep": bool(isstep_arr[end_idx]),
                    "label": str(labels_arr[end_idx]),
                }
            )
            last_vec = first_vec

        return pd.DataFrame(records)

    def export_heading_csv(self, out_path: Path) -> None:
        self.heading_df.to_csv(out_path, index=False)
        print(f"[RMPCA] heading CSV saved to: {out_path}")

    # --- Animation helpers -------------------------------------------------

    def _setup_axes(self):
        self.fig, (self.ax_xy, self.ax_heading) = plt.subplots(1, 2, figsize=(15, 7))
        # XY plane
        self.ax_xy.set_title("RMPCA heading on world XY plane")
        self.ax_xy.set_xlabel("World X (East)")
        self.ax_xy.set_ylabel("World Y (North)")
        self.ax_xy.set_aspect("equal", adjustable="box")
        limit = self.xy_limit
        self.ax_xy.set_xlim(-limit, limit)
        self.ax_xy.set_ylim(-limit, limit)
        self.ax_xy.grid(True, alpha=0.3, linestyle="--")
        self.ax_xy.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
        self.ax_xy.axvline(0, color="gray", linewidth=0.8, alpha=0.6)

        self.window_scatter = self.ax_xy.scatter([], [], s=25, c="tab:blue", alpha=0.35)
        (self.arrow_line,) = self.ax_xy.plot([0, 0], [0, 0], color="tab:red", linewidth=3)
        (self.head_history,) = self.ax_xy.plot([], [], color="tab:orange", linewidth=1.5, alpha=0.6)
        self.info_text = self.ax_xy.text(
            0.02,
            0.98,
            "",
            transform=self.ax_xy.transAxes,
            va="top",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray"),
        )

        # Heading timeline
        self.ax_heading.set_title("Heading timeline (deg) + PC1 strength")
        self.ax_heading.set_xlabel("Timestamp (s)")
        self.ax_heading.set_ylabel("Heading (deg, unwrapped)")
        self.ax_heading.grid(True, alpha=0.3)
        timestamps = self.heading_df["timestamp"].to_numpy()
        heading_vals = self.heading_df["heading_unwrapped_deg"].to_numpy()
        (self.heading_curve,) = self.ax_heading.plot(timestamps, heading_vals, color="tab:purple")
        (self.heading_marker,) = self.ax_heading.plot([], [], "o", color="tab:red")
        # Auto-scale y-limits based on unwrapped heading values
        if heading_vals.size > 0 and np.all(np.isfinite(heading_vals)):
            h_min = float(heading_vals.min())
            h_max = float(heading_vals.max())
            margin = max(10.0, 0.1 * max(abs(h_min), abs(h_max), 1.0))
            self.ax_heading.set_ylim(h_min - margin, h_max + margin)
        else:
            self.ax_heading.set_ylim(-10, 370)

        self.ax_strength = self.ax_heading.twinx()
        self.ax_strength.set_ylabel("PC1 EVR")
        self.ax_strength.set_ylim(0, 1.05)
        strength_vals = self.heading_df["first_evr"].to_numpy()
        (self.strength_curve,) = self.ax_strength.plot(
            timestamps, strength_vals, color="tab:green", alpha=0.5
        )
        (self.strength_marker,) = self.ax_strength.plot([], [], "o", color="tab:green")

    def _update_frame(self, frame_idx: int):
        row = self.heading_df.iloc[frame_idx]
        start = int(row["window_start"])
        end = int(row["window_end"])
        window_pts = self.xy_vectors[start : end + 1]
        self.window_scatter.set_offsets(window_pts)

        # 스무딩된 방향 벡터가 있으면 그것을 사용하고, 없으면 원본 pc1 사용
        east_key = "pc1_east_smooth" if "pc1_east_smooth" in self.heading_df.columns else "pc1_east"
        north_key = "pc1_north_smooth" if "pc1_north_smooth" in self.heading_df.columns else "pc1_north"

        arrow_x = [0, row[east_key] * self.arrow_scale]
        arrow_y = [0, row[north_key] * self.arrow_scale]
        self.arrow_line.set_data(arrow_x, arrow_y)

        history = self.heading_df.iloc[: frame_idx + 1]
        tips = history[[east_key, north_key]].to_numpy() * self.arrow_scale * 0.5
        self.head_history.set_data(tips[:, 0], tips[:, 1])

        self.heading_marker.set_data(row["timestamp"], row["heading_unwrapped_deg"])
        self.strength_marker.set_data(row["timestamp"], row["first_evr"])

        info_lines = [
            f"t = {row['timestamp']:.3f} s",
            f"heading = {row['heading_deg']:6.1f}°",
            f"pc1 = ({row['pc1_east']:+.3f}, {row['pc1_north']:+.3f})",
            f"λ1 = {row['first_var']:.5f}",
            f"EVR = {row['first_evr'] * 100:5.1f} %",
            f"samples = {row['samples']}",
        ]
        label = row.get("label", "")
        if label:
            info_lines.append(f"label = {label}")
        if row.get("isstep"):
            info_lines.append("step detected ✓")

        self.info_text.set_text("\n".join(info_lines))
        return (
            self.window_scatter,
            self.arrow_line,
            self.head_history,
            self.heading_marker,
            self.strength_marker,
            self.info_text,
        )

    def save_heading_figure(self, out_path: Optional[Path] = None) -> None:
        """헤딩 타임라인 그래프만 별도 이미지로 저장 (heading만, PC1 EVR 축 없이)."""
        timestamps = self.heading_df["timestamp"].to_numpy(dtype=float)
        heading_vals = self.heading_df["heading_unwrapped_deg"].to_numpy(dtype=float)

        fig, ax_heading = plt.subplots(figsize=(10, 6))
        ax_heading.set_title("Heading timeline (deg, unwrapped)")
        ax_heading.set_xlabel("Timestamp (s)")
        ax_heading.set_ylabel("Heading (deg, unwrapped)")
        ax_heading.grid(True, alpha=0.3)
        ax_heading.plot(timestamps, heading_vals)

        if out_path is None:
            base = self.csv_path.with_suffix("")
            out_path = base.with_name(base.name + "_heading.png")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[RMPCA] heading figure saved to: {out_path}")

    def animate(self, interval_ms: int = 40, repeat: bool = True):
        """한 스텝마다 키보드 입력(스페이스/→)으로 넘기는 인터랙티브 모드."""
        self._setup_axes()
        # 헤딩 그래프만 별도 파일로 저장 (CSV 이름 기반)
        self.save_heading_figure()

        # 사용할 프레임 인덱스 (서브샘플 반영)
        self._frame_indices = list(range(0, len(self.heading_df), self.subsample))
        self._current_frame_pos = 0

        # 첫 프레임 초기화
        if self._frame_indices:
            first_idx = self._frame_indices[0]
            self._update_frame(first_idx)
            self._current_frame_pos = 1
            self.fig.canvas.draw_idle()

        def on_key(event):
            # 스페이스바나 오른쪽 화살표를 누르면 다음 스텝으로 진행
            if event.key in (" ", "right"):
                if self._current_frame_pos >= len(self._frame_indices):
                    # 더 이상 진행할 프레임이 없으면 무시
                    return
                frame_idx = self._frame_indices[self._current_frame_pos]
                self._update_frame(frame_idx)
                self._current_frame_pos += 1
                self.fig.canvas.draw_idle()

        # 키보드 이벤트 핸들러 등록
        self.fig.canvas.mpl_connect("key_press_event", on_key)

        plt.tight_layout()
        plt.show()
        # 더 이상 FuncAnimation 객체를 반환하지 않고 None 반환
        return None


def default_output_path(csv_path: Path) -> Path:
    base = csv_path.with_suffix("")
    return base.with_name(base.name + ".heading_pca.csv")


def main():
    cfg = CONFIG
    csv_path = Path(cfg["CSV_PATH"])
    animator = RMPCAHeadingAnimator(
        csv_path=csv_path,
        steps_per_window=cfg["STEPS_PER_WINDOW"],
        step_stride=cfg.get("STEP_STRIDE", 1),
        min_samples=cfg["MIN_SAMPLES"],
        lowpass_alpha=cfg["LOWPASS_ALPHA"],
        subsample=cfg["SUBSAMPLE"],
        start_time=cfg["START_TIME"],
        end_time=cfg["END_TIME"],
        max_rows=cfg["MAX_ROWS"],
        heading_smooth_alpha=cfg.get("HEADING_SMOOTH_ALPHA", 0.0),
    )

    if cfg.get("EXPORT_CSV"):
        out_cfg = cfg.get("CSV_OUT")
        out_path = Path(out_cfg) if out_cfg else default_output_path(csv_path)
        animator.export_heading_csv(out_path)

    if cfg.get("SHOW_ANIMATION", True):
        animator.animate(interval_ms=cfg["INTERVAL_MS"], repeat=True)
    else:
        print("Animation disabled by configuration (SHOW_ANIMATION=False).")


if __name__ == "__main__":
    main()
