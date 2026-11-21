import pandas as pd
import numpy as np
import weakref
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

class RegHook:
    """查看神经网络 model 运行时的输入输出数据, 或查看参数和梯度。
    使用方法：
    with RegHook(model, layers=[0, 3, 7], to_cpu=False) as rh:
        out = model(x)
        print(rh.intermediate.keys())
        params_info = RegHook.collect_param_and_grad(model)
    参数layers 也可以使用: 'encoder.layers.1.norm2'
    """
    def __init__(self, model, layers=None, *, to_cpu=True,
                 name_prefix="layer", inout="out"):
        self.handles = []
        self.intermediate = {}
        self.to_cpu = to_cpu
        self._closed = False

        if inout not in ("in", "out", "both"):
            raise ValueError(
                f"inout must be 'in', 'out', or 'both', got {inout!r}"
            )
        self.inout = inout

        # layers 支持 int(索引) / str(完整路径名) / 集合 / None
        if layers is None:
            target = None
        elif isinstance(layers, (int, str)):
            target = {layers}
        elif isinstance(layers, (list, tuple, set)):
            target = set(layers)
        else:
            raise TypeError(
                "layers must be int / str / list / tuple / set / None"
            )

        # 同时拿到 (name, idx, module)
        if hasattr(model, "net"):
            items = [(f"net.{i}", i, m) for i, m in enumerate(model.net)]
        else:
            items = [(n, i, m) for i, (n, m) in enumerate(model.named_modules())]

        for name, idx, layer in items:
            # 筛选：int 匹配 idx，str 匹配 name
            if target is not None and idx not in target and name not in target:
                continue

            cls_name = (layer._get_name()
                        if hasattr(layer, "_get_name")
                        else type(layer).__name__)

            # 根模块 name 为空字符串，退化为旧格式
            key_base = name if name else f"{name_prefix}_{idx}_{cls_name}"

            if self.inout in ("in", "both"):
                self.handles.append(layer.register_forward_pre_hook(self._make_pre_hook(key_base)))
            if self.inout in ("out", "both"):
                self.handles.append(layer.register_forward_hook(self._make_post_hook(key_base)))
            # self.handles.append(layer.register_forward_hook(self._make_hook(key_base)) )
    # ---- hook 相关 ----
    def _make_hook(self, key_base):
        ref = weakref.ref(self)
        def hook(module, inp, out):
            obj = ref()
            if obj is None or obj._closed:
                return
            if obj.inout in ("in", "both"):
                obj.intermediate[f"{key_base}_in"] = obj._pack(inp)
            if obj.inout in ("out", "both"):
                obj.intermediate[f"{key_base}_out"] = obj._pack(out)
        return hook
    def _make_pre_hook(self, key_base):
        ref = weakref.ref(self)
        def hook(module, inp):
            obj = ref()
            if obj is None or obj._closed:
                return
            obj.intermediate[f"{key_base}_in"] = obj._pack(inp)
        return hook
    def _make_post_hook(self, key_base):
        ref = weakref.ref(self)
        def hook(module, inp, out):
            obj = ref()
            if obj is None or obj._closed:
                return
            obj.intermediate[f"{key_base}_out"] = obj._pack(out)
        return hook

    def _pack(self, out):
        if isinstance(out, torch.Tensor):
            t = out.detach()
            return t.cpu() if self.to_cpu else t.clone()
        if isinstance(out, (list, tuple)):
            return type(out)(self._pack(o) for o in out)
        if isinstance(out, dict):
            return {k: self._pack(v) for k, v in out.items()}
        return out

    # ---- 生命周期 ----
    def close_handle(self):
        if self._closed:
            return
        self._closed = True
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close_handle()

    def __del__(self):
        try:
            self.close_handle()
        except Exception:
            pass

    # ---- 参数 / 梯度 ----
    @staticmethod
    def collect_param_and_grad(model, *, layers=None, with_grad_norm=True):
        """收集参数和梯度。
        Args:
            model: 目标模型
            layers:
                - None: 收集全部参数(默认，行为不变)
                - str: 只收集该层(含其所有子层)的参数，如 "encoder.layers.0"
                - list/tuple/set: 收集多个层的参数
            with_grad_norm: 是否计算 grad 的 L2 范数
        """
        # 规范化 layers
        if layers is None:
            target = None
        elif isinstance(layers, str):
            target = {layers}
        elif isinstance(layers, (list, tuple, set)):
            target = set(layers)
        else:
            raise TypeError("layers must be str / list / tuple / set / None")

        def _match(name):
            if target is None:
                return True
            # 精确匹配 "output_proj" 本身（理论上参数不会正好是层名）
            # 或前缀匹配 "output_proj.weight" / "output_proj.bias"
            for t in target:
                if name == t or name.startswith(t + "."):
                    return True
            return False

        info = {}
        for name, p in model.named_parameters():
            if not _match(name):
                continue
            grad = p.grad
            info[name] = {
                "param_data": p.detach().clone(),
                "grad": grad.detach().clone() if grad is not None else None,
                "grad_norm": (grad.norm().item()
                            if (with_grad_norm and grad is not None) else None),
                "shape": tuple(p.shape),
            }
        return info
    @staticmethod
    def get_attn_man(mha, x):
        """
        mha: nn.MultiheadAttention (batch_first=True) # model1.encoder.layers[i].self_attn
        x  : [B, L, E]  #为了获取正确的输入, 需要使用RegHook来获取真实的输入。
        返回每头的 Q, K, V, 注意力权重
        """
        B, L, E = x.shape
        H  = mha.num_heads
        Dh = E // H
        assert E == mha.embed_dim

        # 1) 取出合并的 QKV 权重和 bias
        W = mha.in_proj_weight          # [3E, E]
        b = mha.in_proj_bias            # [3E]
        Wq, Wk, Wv = W.chunk(3, dim=0)  # 各 [E, E]
        bq, bk, bv = b.chunk(3, dim=0)  # 各 [E]

        # 2) 线性投影
        q = F.linear(x, Wq, bq)         # [B, L, E]
        k = F.linear(x, Wk, bk)
        v = F.linear(x, Wv, bv)

        # 3) 拆头
        def reshape(t):
            return t.view(B, L, H, Dh).transpose(1, 2)   # [B, H, L, Dh]

        q_h, k_h, v_h = reshape(q), reshape(k), reshape(v)

        # 4) 每头注意力
        scale  = Dh ** 0.5
        scores = (q_h @ k_h.transpose(-2, -1)) / scale   # [B, H, L, L]
        attn   = scores.softmax(dim=-1)                  # [B, H, L, L]
        return q_h, k_h, v_h, attn
    @staticmethod
    def get_attn_sys(rh=None, model=None,layers=None, *, x=None, 
                    average_attn_weights=False,
                    key_padding_mask=None,
                    attn_mask=None,
                    return_output=False,
                    preserve_train_state=True):
        """提取 Transformer 每层注意力权重。
        ── 用法 A(推荐, 从已有 RegHook 提取，不重复 forward)──
            targets = [f"encoder.layers.{i}.self_attn" for i in range(model.encoder.num_layers)]
            with RegHook(model, layers=targets, inout="in", to_cpu=False) as rh:
                model(x)
                attn = RegHook.get_attn_sys(rh, model=model)
            # attn = {"encoder.layers.0.self_attn": [B,H,L,L], ...}

        ── 用法 B(便捷, 内部用 RegHook 跑一次 forward) ──
            attn = RegHook.get_attn_sys(model=model1, x=x,layers="encoder.layers.0.self_attn")
        Args:
            rh:  RegHook 实例, forward 已运行, inout 需包含 'in',
                且 layers 覆盖所有需要的 self_attn / multihead_attn。
                提供后 x 被忽略，不再重跑 forward。
            model: 模型对象（两个用法都必传，用于取 mha 模块引用）。
            x:   model 的输入（仅用法 B 使用）。
            layers: str/list/tuple/set/None, 仅筛选层,语义同 RegHook。
            average_attn_weights: False → [B,H,L,L]; True → [B,L,L]。
            key_padding_mask: 透传给 mha。
            return_output: True 时返回 (attn_out, attn_w)，否则仅返回 attn_w。
            preserve_train_state: 用法 B 下是否恢复 model 的 train 状态。

        Returns:
            dict: {层名: attn_w} 或 {层名: (attn_out, attn_w)}
        """
        if model is None:
            raise ValueError("model 必传")
        if rh is None and x is None:
            raise ValueError("必须提供 rh, 或提供 x(内部跑 forward)")

        # 1) 候选 attention 模块
        candidates = []
        encd = getattr(model, "encoder", None)
        if encd is not None:
            for i, layer in enumerate(encd.layers):
                candidates.append((f"encoder.layers.{i}.self_attn",layer.self_attn, "self"))
        decd = getattr(model, "decoder", None)
        if decd is not None:
            for i, layer in enumerate(decd.layers):
                candidates.append((f"decoder.layers.{i}.self_attn",layer.self_attn, "self"))
                if hasattr(layer, "multihead_attn"):
                    candidates.append((f"decoder.layers.{i}.multihead_attn",layer.multihead_attn, "cross"))

        # 2) 层筛选
        if layers is not None:
            if isinstance(layers, str):
                target = {layers}
            elif isinstance(layers, (list, tuple, set)):
                target = set(layers)
            else:
                raise TypeError("layers must be str/list/tuple/set/None")

            def _pick(name):
                return any(name == t or name.startswith(t + ".")
                        for t in target)
            candidates = [c for c in candidates if _pick(c[0])]

        if not candidates:
            return {}

        # 3) 准备 rh（数据源）
        if rh is None:
            # 用法 B：内部用 RegHook 抓一次
            target_layers = [n for n, _, _ in candidates]
            was_training = model.training
            try:
                with RegHook(model, layers=target_layers, inout="in",
                            to_cpu=False) as _rh:
                    with torch.no_grad():
                        model.eval()      # 避免 forward 阶段 dropout 污染
                        _ = model(x)
            finally:
                if preserve_train_state:
                    model.train(was_training)
            rh = _rh
        else:
            # 用法 A：检查 rh 是否覆盖了所有候选层
            missing = [n for n, _, _ in candidates
                    if f"{n}_in" not in rh.intermediate]
            if missing:
                raise KeyError(
                    f"rh.intermediate 缺少以下层的输入: {missing}。"
                    f"请确保创建 RegHook 时 inout='in' 或 'both'，"
                    f"且 layers 覆盖这些层。"
                )

        # 4) 从 rh 提取输入，算注意力
        def _run(m, q, k=None, v=None):
            k = q if k is None else k
            v = k if v is None else v

            # 设备对齐：rh 里可能是 CPU 数据（to_cpu=True 时）
            dev = m.in_proj_weight.device
            orig_dev = q.device
            q = q.to(dev) if q.device != dev else q
            k = k.to(dev) if k.device != dev else k
            v = v.to(dev) if v.device != dev else v

            # 临时 eval，避免 attention dropout 污染结果
            was_train = m.training
            m.eval()
            try:
                with torch.no_grad():
                    out, w = m(
                        q, k, v,
                        need_weights=True,
                        average_attn_weights=average_attn_weights,
                        key_padding_mask=key_padding_mask,
                        attn_mask=attn_mask,        # <--- 新增这一行
                    )
            finally:
                m.train(was_train)

            if orig_dev != dev:
                out = out.to(orig_dev)
                w = w.to(orig_dev)
            return (out, w) if return_output else w

        rst = {}
        for name, m, kind in candidates:
            inp = rh.intermediate[f"{name}_in"]
            if kind == "self":
                q = inp[0]
                rst[name] = _run(m, q)
            else:                    # cross: inp = (q, memory, ...)
                q, mem = inp[0], inp[1]
                rst[name] = _run(m, q, k=mem, v=mem)
        return rst
    @staticmethod
    def get_attn_one(mha, x, *, average_attn_weights=False,
                    key_padding_mask=None, return_output=False):
        """对单个 MultiheadAttention 用给定输入算注意力权重。
        使用：
        w = RegHook.get_attn_one(
            model1.encoder.layers[0].self_attn,
            rh2.intermediate['encoder.layers.0.self_attn_in'][0],
            )
        """
        was_train = mha.training
        mha.eval()
        try:
            with torch.no_grad():
                out, w = mha(
                    x, x, x,
                    need_weights=True,
                    average_attn_weights=average_attn_weights,
                    key_padding_mask=key_padding_mask,
                )
        finally:
            mha.train(was_train)
        return (out, w) if return_output else w
    @staticmethod
    def plot_attn(attn, batch_idx=0, title=""):
        """绘制第batch_idx 批次的 的注意力权重
        attn: 需要显示的注意力权重[batch,num_headers,L,L]
        batch_idx: 默认为第1批次
        """
        attn = attn.detach().cpu()
        H = attn.size(1)
        fig, axes = plt.subplots(1, H, figsize=(4 * H, 4))
        if H == 1: axes = [axes]
        for h in range(H):
            im = axes[h].imshow(attn[batch_idx, h], aspect="auto", cmap="viridis")
            axes[h].set_title(f"head {h}")
            axes[h].set_xlabel("key pos"); axes[h].set_ylabel("query pos")
        fig.suptitle(title); fig.colorbar(im, ax=axes, fraction=0.02); plt.show()
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

if __name__ == "__main__":
    # rng = np.random.default_rng(0)
    # x = np.linspace(0, np.pi / 2, 60)
    # y = np.sin(x) + rng.normal(0, 0.002, x.size)
    # lsc = LinearSimCurve(x,y)

    # # 日常使用：只要 y_hat 和 mse
    # out = lsc.segmented_linear_regression(x, y, angle_diff_deg=10, mse_threshold=1e-5)
    # print("mse:", out["mse"])
    # # plt.plot(x, out["y_hat"])

    # # 调试：看切段和迭代
    # out = lsc.segmented_linear_regression(x, y, angle_diff_deg=10,
    #                                 mse_threshold=1e-5, debug=True)
    # print("converged:", out["converged"],
    #     "阈值:", out["angle_diff_threshold_deg"],
    #     "轮数:", out["n_iterations"],
    #     "段数:", len(out["segments"]))
    # for f in out["segments"]:
    #     print(f"  [{f['start']:>2},{f['end']:>2}] 点数={f['n_points']:>2} "
    #         f"角度={f['angle_deg']:.2f}°")
    pass