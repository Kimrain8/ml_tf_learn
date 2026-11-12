import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.datasets import load_iris
import matplotlib
# print(matplotlib.matplotlib_fname())
# print(matplotlib.matplotlib_fname())
matplotlib.use('QtAgg')   # 或 'TkAgg'/'Qt5Agg' 需要安装PyQt5或PyQt6 / 'MacOSX'
import matplotlib.pyplot as plt
# print(plt.rcParams['font.sans-serif'])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 修复负号显示成方块
from IPython import embed
from utils.MyTools import MyTools as mt
# print(matplotlib.get_backend())

"""使用分段线性回归, 模拟曲线方程"""

class LinearSimCurve():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def _segment_angles(self,x, y):
        """每个相邻点连线的角度（度），长度 n-1。theta[i] 对应点 i -> i+1。"""
        dx, dy = np.diff(x), np.diff(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.degrees(np.arctan2(dy, dx))
        return np.where(np.isfinite(theta), theta, 0.0)

    def _make_segments(self,theta, threshold, mode="cumulative", min_seg_len=2):
        """按角度变化切段，返回 inclusive 点区间 [(s, e), ...]，相邻段共享切点。"""
        n = len(theta) + 1
        if len(theta) == 0:
            return [(0, n - 1)]

        cuts = [0]
        if mode == "adjacent":
            for i in range(1, len(theta)):
                if abs(theta[i] - theta[i - 1]) > threshold:
                    cuts.append(i)
        elif mode == "cumulative":
            seg_min = seg_max = theta[0]
            for i in range(1, len(theta)):
                new_min = min(seg_min, theta[i])
                new_max = max(seg_max, theta[i])
                if new_max - new_min > threshold:
                    cuts.append(i)
                    seg_min = seg_max = theta[i]
                else:
                    seg_min, seg_max = new_min, new_max
        else:
            raise ValueError(f"未知 mode: {mode!r}")
        cuts.append(n - 1)

        segs = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]

        changed = True
        while changed:
            changed = False
            for k, (s, e) in enumerate(segs):
                if e - s + 1 < min_seg_len:
                    if k > 0:
                        segs[k - 1] = (segs[k - 1][0], e)
                        segs.pop(k)
                    elif k < len(segs) - 1:
                        segs[k] = (s, segs[k + 1][1])
                        segs.pop(k + 1)
                    changed = True
                    break
        return segs

    def segmented_linear_regression(self,
        x, y,
        angle_diff_deg=10.0,
        angle_mode="cumulative",
        mse_threshold=0.01,
        max_iter=100,
        shrink_factor=0.8,
        min_angle_diff_deg=1e-4,
        min_seg_len=2,
        debug=False, 
    ):
        """
        分段线性回归（按角度变化切段）

        参数
        ----
        angle_diff_deg     : 初始角度阈值（度），触发切段
        angle_mode         : "cumulative" (默认) 段内角度极差 max-min > 阈值 → 切
                            "adjacent"          |θ[i]-θ[i-1]| > 阈值 → 切
        mse_threshold      : 目标整体 MSE; 不达标则 shrink_factor 缩阈值重来
        max_iter           : 最多迭代轮数
        shrink_factor      : 阈值每轮收缩比例 (0,1)
        min_angle_diff_deg : 阈值下限，低于此值停止
        min_seg_len        : 每段最少点数，过短段自动合并
        debug              : False → 只返回 {y_hat, mse}
                            True  → 返回完整诊断信息

        返回
        ----
        debug=False:
            {"y_hat": ndarray, "mse": float}
        debug=True:
            {"y_hat", "mse", "segments", "angle_diff_threshold_deg",
            "n_iterations", "converged", "target_mse", "angle_mode", "history"}
        angle_diff_threshold_deg: 最终生效的角度阈值(可能经过若干次收缩)
        n_iterations: 实际迭代轮数(1 表示一次就达标)
        converged: 是否在 max_iter 内达到 mse_threshold
        target_mse: 用户设定的目标 MSE(回显, 方便日志)
        angle_mode: 回显本次使用的切段模式
        history: 每轮 {iteration, angle_diff_threshold_deg, n_segments, mse}，画"阈值-段数-MSE 收敛曲线"很方便
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("x, y 需等长且至少 2 个点")
        if angle_mode not in ("cumulative", "adjacent"):
            raise ValueError("angle_mode 必须为 'cumulative' 或 'adjacent'")

        theta = self._segment_angles(x, y)
        history, best, y_hat_best = [], None, None
        threshold = float(angle_diff_deg)
        converged = False

        for it in range(max_iter):
            segs = self._make_segments(theta, threshold, angle_mode, min_seg_len)

            fits = []
            y_hat = np.full_like(y, np.nan)
            for idx, (s, e) in enumerate(segs):
                xs, ys = x[s:e + 1], y[s:e + 1]
                if len(xs) >= 2:
                    a, b = np.polyfit(xs, ys, 1)
                else:
                    a, b = 0.0, float(ys.mean())
                y_hat[s:e + 1] = a * xs + b          # 共享切点由后一段覆盖
                fits.append({
                    "index": idx, "start": int(s), "end": int(e),
                    "n_points": int(e - s + 1),
                    "slope": float(a), "intercept": float(b),
                    "angle_deg": float(np.degrees(np.arctan(a))),
                })

            mse = float(np.mean((y - y_hat) ** 2))
            history.append({
                "iteration": it, "angle_diff_threshold_deg": threshold,
                "n_segments": len(segs), "mse": mse,
            })

            if best is None or mse < best["mse"]:
                best = {
                    "mse": mse, "segments": fits,
                    "angle_diff_threshold_deg": threshold,
                    "n_iterations": it + 1,
                }
                y_hat_best = y_hat.copy()

            if mse <= mse_threshold:
                converged = True
                break

            threshold *= shrink_factor
            if threshold < min_angle_diff_deg:
                break

        if not debug:
            return {"y_hat": y_hat_best, "mse": best["mse"]}
        return {
            **best,
            "y_hat": y_hat_best,
            "converged": converged,
            "target_mse": mse_threshold,
            "angle_mode": angle_mode,
            "history": history,
        }

def seriesAngles():
    x = np.linspace(-np.pi,10*np.pi,700)
    y = np.sin(x)

    angles = [0]
    angles2 = [0]
    for i in range(2,len(x)+1):
        angles.append(mt.myVectorAngle(x[:i],y[:i]))
        angles2.append(mt.myVectorAngle(np.ones_like(x)[:i],y[:i])) 

    arr = np.column_stack((x,y,angles,angles2))

    df = pd.DataFrame(arr,columns=['x','y','angles','angle2'])
    # embed()
    df.to_pickle("tmp_20261110.pkl")

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.linspace(0, np.pi / 2, 60)
    y = np.sin(x) + rng.normal(0, 0.002, x.size)
    lsc = LinearSimCurve(x,y)

    # 日常使用：只要 y_hat 和 mse
    out = lsc.segmented_linear_regression(x, y, angle_diff_deg=10, mse_threshold=1e-5)
    print("mse:", out["mse"])
    # plt.plot(x, out["y_hat"])

    # 调试：看切段和迭代
    out = lsc.segmented_linear_regression(x, y, angle_diff_deg=10,
                                    mse_threshold=1e-5, debug=True)
    print("converged:", out["converged"],
        "阈值:", out["angle_diff_threshold_deg"],
        "轮数:", out["n_iterations"],
        "段数:", len(out["segments"]))
    for f in out["segments"]:
        print(f"  [{f['start']:>2},{f['end']:>2}] 点数={f['n_points']:>2} "
            f"角度={f['angle_deg']:.2f}°")
    embed()


