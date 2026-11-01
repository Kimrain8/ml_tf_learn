import pandas as pd
import numpy as np
import re

import time
from Levenshtein import distance as levenshtein_distance  # 需安装：pip install python-Levenshtein
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from IPython.display import clear_output
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.cm import ScalarMappable
import mplcursors

import torch
from matplotlib.animation import FuncAnimation

# 全局字体设置（建议放在脚本开头，这里保留以确保独立性）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'Liberation Sans', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

class MyTools:
    @staticmethod
    def myFindSimilar(df1: pd.DataFrame,df2: pd.DataFrame,exclude_cols: list = None,
        enable_similarity: str = 'Y',ret_similar='N') -> pd.DataFrame:
        """
        功能：从df2中找到和df1每行最相似的3条记录，支持「精确相等」和「带相似度」两种模式
        
        参数：
            df1: 基准DataFrame
            df2: 待匹配DataFrame（列结构与df1一致）
            exclude_cols: 可选，无需比较的列名列表（如['id', 'create_time']）
            enable_similarity: 可选，比较模式：
                - 'Y'（默认）：执行文本/数值带相似度的比较
                - 'N'：执行精确相等比较，仅当值完全相等且非双空时相似度为1，否则为0（自动转大写判断，兼容'y'/'n'）
            ret_similar: 是返回相似矩阵，还是返回Top3的最相似的记录。Y表示返回相似矩阵，N表示返回top3的相似记录行号。默认为N
        返回：
            pd.DataFrame: 行数与df1一致，包含top1/top2/top3三列，值为df2中对应行号（float类型，含NaN）
        """
        pd.set_option('future.no_silent_downcasting', True)
        # -------------------------- 1. 输入校验与预处理 --------------------------
        # 校验列名一致性
        if not df1.columns.equals(df2.columns):
            raise ValueError("两个DataFrame的列名必须完全一致！")
        # 校验df2是否为空
        if df2.empty:
            raise ValueError("待匹配的df2不能为空！")
        # 校验enable_similarity参数
        enable_similarity = enable_similarity.strip().upper()
        if enable_similarity not in ['Y', 'N']:
            raise ValueError("enable_similarity参数只能是'Y'或'N'！")
        # 初始化排除列
        exclude_cols = exclude_cols or []
        # 校验排除列是否存在
        invalid_cols = [col for col in exclude_cols if col not in df1.columns]
        if invalid_cols:
            raise ValueError(f"排除列 {invalid_cols} 不存在于DataFrame中！")
        # 保留需要比较的列
        compare_cols = [col for col in df1.columns if col not in exclude_cols]
        if not compare_cols:
            raise ValueError("没有需要比较的列！请检查exclude_cols参数。")
        # 预处理：空字符串统一转NaN（两种模式通用）
        df1_clean = df1[compare_cols].replace('', np.nan).infer_objects(copy=False).copy()
        df2_clean = df2[compare_cols].replace('', np.nan).infer_objects(copy=False).copy()
        # -------------------------- 2. 计算相似度矩阵 --------------------------
        # 初始化相似度矩阵（行：df1行数，列：df2行数，值0-1）
        similarity_matrix = np.zeros((len(df1_clean), len(df2_clean)), dtype=np.float32)
        if enable_similarity == 'Y':
            # -------------------------- 2.1 带相似度的比较模式 --------------------------
            # 自动区分文本列和数值列
            text_cols = []
            numeric_cols = []
            for col in compare_cols:
                # 先尝试将列转为数值，能转的视为数值列（处理object类型的数值）
                # 边界处理：列全是NaN时，默认视为文本列
                temp_series1 = df1_clean[col].dropna()
                temp_series2 = df2_clean[col].dropna()
                if temp_series1.empty and temp_series2.empty:
                    text_cols.append(col)
                    continue
                try:
                    pd.to_numeric(temp_series1, errors='raise')
                    pd.to_numeric(temp_series2, errors='raise')
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    text_cols.append(col)
            # --- 2.1.1 文本列相似度计算（Levenshtein编辑距离归一化）---
            if text_cols:
                # 预转为字符串（处理NaN）
                df1_text = df1_clean[text_cols].fillna('').astype(str)
                df2_text = df2_clean[text_cols].fillna('').astype(str)
                # 向量化计算每对行的文本相似度（平均各列）
                for col in text_cols:
                    # 计算该列所有行对的编辑距离
                    arr1 = df1_text[col].values
                    arr2 = df2_text[col].values
                    # 广播计算：(len(df1), len(df2)) 的编辑距离矩阵
                    # 优化：如果行数超过10000，分块计算避免OOM
                    if len(arr1) * len(arr2) > 1e8:
                        chunk_size = 1000
                        lev_matrix = np.zeros((len(arr1), len(arr2)), dtype=np.float32)
                        for i in range(0, len(arr1), chunk_size):
                            end_i = min(i + chunk_size, len(arr1))
                            lev_matrix[i:end_i, :] = np.vectorize(
                                lambda x, y: levenshtein_distance(x, y),
                                otypes=[np.float32]
                            )(arr1[i:end_i, None], arr2[None, :])
                    else:
                        lev_matrix = np.vectorize(
                            lambda x, y: levenshtein_distance(x, y),
                            otypes=[np.float32]
                        )(arr1[:, None], arr2[None, :])
                    # 归一化到 [0,1]：1 - (编辑距离 / max(len(x), len(y), 1))
                    len1 = np.vectorize(len, otypes=[np.float32])(arr1)
                    len2 = np.vectorize(len, otypes=[np.float32])(arr2)
                    max_len = np.maximum(len1[:, None], len2[None, :])
                    max_len = np.maximum(max_len, 1.0)  # 避免除零
                    col_similarity = 1.0 - (lev_matrix / max_len)
                    
                    # 处理双空值情况（原空值已转空字符串，需还原为0）
                    nan_mask1 = df1_clean[col].isna().values
                    nan_mask2 = df2_clean[col].isna().values
                    # 双空值位置相似度设为0
                    col_similarity[np.ix_(nan_mask1, nan_mask2)] = 0.0
                    
                    # 累加到总相似度矩阵
                    similarity_matrix += col_similarity
            # --- 2.1.2 数值列相似度计算（绝对差归一化）---
            if numeric_cols:
                # 预转为数值（处理object类型的数值）
                df1_num = df1_clean[numeric_cols].apply(pd.to_numeric, errors='coerce')
                df2_num = df2_clean[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # 向量化计算每对行的数值相似度（平均各列）
                for col in numeric_cols:
                    arr1 = df1_num[col].values.astype(np.float32)
                    arr2 = df2_num[col].values.astype(np.float32)
                    
                    # 计算绝对差矩阵
                    abs_diff = np.abs(arr1[:, None] - arr2[None, :])
                    
                    # 归一化到 [0,1]：1 - (绝对差 / max(|x|, |y|, 1))
                    max_abs = np.maximum(np.abs(arr1[:, None]), np.abs(arr2[None, :]))
                    max_abs = np.maximum(max_abs, 1.0)  # 避免除零
                    col_similarity = 1.0 - (abs_diff / max_abs)
                    
                    # 处理空值情况（单/双空值相似度设为0）
                    nan_mask1 = df1_num[col].isna().values
                    nan_mask2 = df2_num[col].isna().values
                    # 只要有一个空值，相似度设为0
                    col_similarity[np.ix_(nan_mask1, np.ones(len(nan_mask2), dtype=bool))] = 0.0
                    col_similarity[np.ix_(np.ones(len(nan_mask1), dtype=bool), nan_mask2)] = 0.0
                    
                    # 累加到总相似度矩阵
                    similarity_matrix += col_similarity
            
            # --- 2.1.3 平均各列相似度（得到最终相似度矩阵）---
            similarity_matrix /= len(compare_cols)
        else:
            # -------------------------- 2.2 精确相等比较模式 --------------------------
            # 逐列逐行比较：完全相等且非双空 → 1.0，否则 → 0.0
            for col in compare_cols:
                # 生成「完全相等」的布尔矩阵
                equal_mask = (df1_clean[col].values[:, None] == df2_clean[col].values[None, :])
                # 生成「双空值」的布尔矩阵
                double_nan_mask = (df1_clean[col].isna().values[:, None] & df2_clean[col].isna().values[None, :])
                # 生成「有效相等」的布尔矩阵（相等且非双空）
                valid_equal_mask = equal_mask & (~double_nan_mask)
                # 转成float32（1.0/0.0），累加到总相似度矩阵
                similarity_matrix += valid_equal_mask.astype(np.float32)
            
            # --- 2.2.1 平均各列有效相等数（得到最终相似度矩阵）---
            similarity_matrix /= len(compare_cols)
        if ret_similar == 'Y':
            return similarity_matrix
        else:
            # -------------------------- 3. 筛选top3相似行号 --------------------------
            # 按相似度降序排序，取前3个索引（行号）
            # argsort默认升序，取负实现降序
            top3_indices = np.argsort(-similarity_matrix, axis=1)[:, :3].astype(np.float32)  # 先转float，再填充NaN
            
            # 处理df2行数不足3的情况（填充NaN）
            pad_width = max(0, 3 - top3_indices.shape[1])
            if pad_width > 0:
                top3_indices = np.pad(
                    top3_indices,
                    ((0, 0), (0, pad_width)),
                    mode='constant',
                    constant_values=np.nan
                )
            # -------------------------- 4. 格式化输出 --------------------------
            result_df = pd.DataFrame(
                top3_indices,
                columns=['top1', 'top2', 'top3'],
                index=df1.index
            )
            return result_df

    @staticmethod
    def myPlot3dDF(data, row_slice="", col_slice="", cmap="viridis", p='Y', point_size=20, figsize=None):
        """
        用3D的方式显示data的数据，其中x轴是df的列，y轴是df的行
        data：数据类型tensor 或者numpy.array，
        增加参数：
        figsize: (宽度, 高度)，如 (12,7)
        默认自动根据数据大小设置，不会太小也不会太大
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
        # 解析切片
        def parse_slice(s, max_len):
            s = str(s).strip()
            if not s:
                return 0, max_len
            parts = s.split(',')
            start = parts[0].strip()
            end = parts[1].strip() if len(parts) > 1 else ""
    
            s_idx = 0
            if start:
                s_idx = int(start)
    
            e_idx = max_len
            if end:
                e_idx = int(end)
    
            return s_idx, e_idx
        
        # 数据转换
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.array(data)
        
        n_rows, n_cols = arr.shape
        r1, r2 = parse_slice(row_slice, n_rows)
        c1, c2 = parse_slice(col_slice, n_cols)
        sliced = arr[r1:r2, c1:c2]
        
        # 关闭所有旧图
        plt.close('all')
    
        # ======================
        # ✅ 智能默认画布大小
        # ======================
        if figsize is None:
            h, w = sliced.shape
            base_w = min(14, max(8, w * 0.3 + 4))   # 宽度自适应
            base_h = min(9, max(5, h * 0.3 + 3))    # 高度自适应
            figsize = (base_w, base_h)
    
        if p == 'N':
            y = range(sliced.shape[0])
            x = range(sliced.shape[1])
            X, Y = np.meshgrid(x, y)
            Z = sliced
            
            # 使用传入/自动计算的画布大小
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(
                X, Y, Z,
                cmap='viridis',
                edgecolor='none',
                alpha=0.8
            )
            
            ax.set_title('3D 曲面图 z = x × y（可交互旋转）', fontsize=14, pad=20)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            plt.show()
        else:
            num_rows, num_cols = sliced.shape
            x_coords = []
            y_coords = []
            z_coords = []
        
            for y in range(num_rows):
                for x in range(num_cols):
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(sliced[y, x])
    
            # 使用传入/自动计算的画布大小
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            f1 = ax.scatter(
                x_coords, y_coords, z_coords,
                c=z_coords,
                cmap=cmap,
                s=point_size
            )
            ax.set_xlabel(f'列 {c1}-{c2-1}')
            ax.set_ylabel(f'行 {r1}-{r2-1}')
            fig.colorbar(f1, ax=ax, shrink=0.5)
            plt.title(f"切片 行[{row_slice}] 列[{col_slice}]")
            plt.tight_layout()
            plt.show()
        return sliced

    @staticmethod
    def myPlot3dPoints(data, color_col=3, figsize=(8, 6), cmap='viridis',xlim=None, ylim=None, zlim=None, fig=None, ax=None,show=True):
        """
        绘制给定x,y,z位置的三维散点图，支持指定颜色列或直接传入颜色数据。
        参数
        ----------
        data : ndarray, DataFrame 或 Tensor
            输入数据，形状 (n, 3) 或 (n, 4)。若为 (n,3) 且未传入颜色，则所有点同色。
        color_col : int, str 或 array-like, 默认 3
            - int/str：指定从 data 中提取颜色列的索引或列名。
            - array-like（ndarray, DataFrame, Tensor）：直接作为颜色数据，长度必须为 n。
              支持一维数组，或二维但第二维为 1 的数组/DataFrame。
        figsize : tuple, 默认 (8, 6)
        cmap : str, 默认 'viridis'
        xlim, ylim, zlim : tuple 或 None
        fig, ax : matplotlib figure/axes，若提供则绘制于其上。
        返回
        -------
        fig, ax
        """
        # ---------- 数据预处理 ----------
        if hasattr(data, 'numpy'):          # PyTorch Tensor
            data_np = data.detach().cpu().numpy()
        elif hasattr(data, 'values'):       # Pandas DataFrame
            data_np = data.values
        else:
            data_np = np.asarray(data)
    
        if data_np.ndim != 2:
            raise ValueError("数据必须是二维数组，形状 (n, 3) 或 (n, 4)")
        n_rows, n_cols = data_np.shape
        if n_cols < 3:
            raise ValueError("数据至少需要 3 列 (x, y, z)")
    
        x = data_np[:, 0]
        y = data_np[:, 1]
        z = data_np[:, 2]
    
        # ---------- 处理颜色 ----------
        has_color = False
        colors = None
    
        # ===== 新增：判断 color_col 是否直接为颜色数据 =====
        if hasattr(color_col, 'shape') or hasattr(color_col, 'numpy') or hasattr(color_col, 'values'):
            # 转换为 NumPy 数组
            if hasattr(color_col, 'numpy'):          # Tensor
                colors_np = color_col.detach().cpu().numpy()
            elif hasattr(color_col, 'values'):       # DataFrame/Series
                colors_np = color_col.values
            else:                                    # ndarray 或 list
                colors_np = np.asarray(color_col)
    
            # 确保一维（允许二维单列）
            if colors_np.ndim == 2:
                if colors_np.shape[1] == 1:
                    colors_np = colors_np.flatten()
                else:
                    raise ValueError("如果 color_col 是二维数组，必须只有一列 (n,1)")
    
            # 长度校验
            if len(colors_np) != n_rows:
                raise ValueError(f"color_col 长度 ({len(colors_np)}) 必须与数据行数 ({n_rows}) 一致")
    
            colors = colors_np
            has_color = True
        # ===== 否则按原逻辑从 data 中提取颜色列 =====
        else:
            if n_cols >= 4:
                if isinstance(data, (pd.DataFrame)):
                    try:
                        col = data.iloc[:, color_col] if isinstance(color_col, int) else data[color_col]
                        colors = col.values
                        has_color = True
                    except Exception:
                        colors = data_np[:, color_col] if isinstance(color_col, int) else data_np[:, -1]
                        has_color = True
                else:
                    col_idx = color_col if isinstance(color_col, int) else -1
                    if 0 <= col_idx < n_cols:
                        colors = data_np[:, col_idx]
                        has_color = True
    
        # ---------- 创建图形 ----------
        if ax is None:
            plt.close('all')
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
    
        # ---------- 绘制散点 ----------
        if has_color and colors is not None:
            if colors.dtype.kind in 'USO':  # 字符串或对象类型（类别）
                unique_cats = np.unique(colors)
                cat_to_int = {cat: i for i, cat in enumerate(unique_cats)}
                colors_num = np.array([cat_to_int[cat] for cat in colors]).astype(float)
                sc = ax.scatter(x, y, z, c=colors_num, cmap=cmap, s=20, alpha=0.7)
                cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(len(unique_cats)))
                cbar.set_ticklabels(unique_cats)
            else:
                colors_num = colors.astype(float)
                sc = ax.scatter(x, y, z, c=colors_num, cmap=cmap, s=20, alpha=0.7)
                fig.colorbar(sc, ax=ax, label='Color value')
        else:
            ax.scatter(x, y, z, c='blue', s=20, alpha=0.7)
    
        # ---------- 坐标轴与范围 ----------
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        margin = 0.01
        if xlim is None:
            x_min, x_max = x.min(), x.max()
            dx = (x_max - x_min) * margin
            ax.set_xlim(x_min - dx, x_max + dx)
        else:
            ax.set_xlim(xlim)
        if ylim is None:
            y_min, y_max = y.min(), y.max()
            dy = (y_max - y_min) * margin
            ax.set_ylim(y_min - dy, y_max + dy)
        else:
            ax.set_ylim(ylim)
        if zlim is None:
            z_min, z_max = z.min(), z.max()
            dz = (z_max - z_min) * margin
            ax.set_zlim(z_min - dz, z_max + dz)
        else:
            ax.set_zlim(zlim)
        if show:
            from IPython.display import display
            display(fig)

        return fig, ax

    @staticmethod
    def myPlotPointPairs(points, idx, red_duration=0.8, single='Y'):
        """
        单个二维画布上,先显示所有点，然后每 red_duration 秒让一组点（idx指定）显示红色，再等 red_duration 秒变黑。
        参数:
            points：全部需要显示的点，是一个N行2列的ndarray
            idx：是一个dict，key是0，1，2，...，值是一个数组，数组的值为对应点的序号
            red_duration: 红色显示时长（秒）
            single: 控制模式，'Y' 表示无黑色间隔直接切换下一对，'N' 表示红黑交替
        """
        # 强制使用交互后端（脚本中可省略，Jupyter 必须手动切换）
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        # 初始全部黑色
        n = len(points)
        base_colors = np.zeros((n, 4))
        base_colors[:, 3] = 1.0          # alpha = 1
        sc = ax.scatter(points[:, 0], points[:, 1], facecolors=base_colors)
        total_pairs = len(idx)
        if total_pairs == 0:
            plt.show()
            return
        single_mode = single == 'Y' or single == 'y'
        interval = 50                    # 更新间隔(ms)
        frames_per_stage = int(red_duration * 1000 / interval)   # 红色/黑色持续帧数
        if single_mode:
            frames_per_pair = frames_per_stage          # 每对只有红色阶段
            total_frames = total_pairs * frames_per_pair
        else:
            frames_per_pair = 2 * frames_per_stage      # 红色+黑色
            total_frames = total_pairs * frames_per_pair
        # 辅助变量：记录上一对索引（用于 single 模式重置为黑色）
        prev_pair_idx = -1
        def update(frame):
            nonlocal prev_pair_idx
            if single_mode:
                # 当前第几对
                pair_idx = frame // frames_per_stage
                if pair_idx >= total_pairs:
                    anim.event_source.stop()
                    return [sc]
                # 将上一对点恢复为黑色
                if prev_pair_idx != -1 and prev_pair_idx != pair_idx:
                    i_prev = idx[prev_pair_idx]
                    base_colors[i_prev] = [0, 0, 0, 1]
                # 将当前对设为红色
                i = idx[pair_idx]
                base_colors[i] = [1, 0, 0, 1]
                prev_pair_idx = pair_idx
            else:   # 原模式：红黑交替
                pair_idx = frame // frames_per_pair
                if pair_idx >= total_pairs:
                    anim.event_source.stop()
                    return [sc]
                stage = (frame % frames_per_pair) // frames_per_stage
                i = idx[pair_idx]
                if stage == 0:      # 红色
                    base_colors[i] = [1, 0, 0, 1]
                else:               # 黑色
                    base_colors[i] = [0, 0, 0, 1]
            sc.set_facecolor(base_colors)
            return [sc]
        anim = FuncAnimation(fig, update, frames=total_frames,
                             interval=interval, blit=True, repeat=False)
        plt.show()
        return anim

    @staticmethod
    def myPlotPointPairs_ori(points, idx,red_duration=0.8,fsize=(8,8),xlim=(-3.5,3.5),ylim=(-3.5,3.5)):
        """
        单个画布上，每 red_duration 秒让一对点变红，再等 red_duration 秒变黑。
        points：全部需要显示的点，是一个N行2列的ndarray
        idx：是一个字典，key 是0，1，2，...的序号。value是列表，其中是point的行序号
        red_duration：显示为不同颜色的持续时间
        """
        # 强制使用交互后端（脚本中可省略，Jupyter 必须手动切换）
        plt.close('all')
        fig, ax = plt.subplots(figsize=fsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
        # 初始全部黑色
        n = len(points)
        base_colors = np.zeros((n, 4))
        base_colors[:, 3] = 1.0  # alpha = 1
        sc = ax.scatter(points[:, 0], points[:, 1], facecolors=base_colors)
    
        total_pairs = len(idx)
        if total_pairs == 0:
            plt.show()
            return
    
        # 将时间转为帧数：red_duration 秒 → 红色帧数，黑色也持续同样帧数
        interval = 50  # 更新间隔(ms)，越小越流畅 。1秒20帧
        frames_per_stage = int(red_duration * 1000 / interval)  # 红色/黑色持续帧数
        frames_per_pair = 2 * frames_per_stage                 # 显示一组点的总帧数
        total_frames = total_pairs * frames_per_pair
    
        def update(frame):
            # 计算当前是第几对点
            pair_idx = frame // frames_per_pair
            if pair_idx >= total_pairs:
                # 全部播完，动画自动停止
                anim.event_source.stop()
                return [sc]
    
            # 判断当前阶段：0=红色，1=黑色
            stage = (frame % frames_per_pair) // frames_per_stage
    
            i = idx[pair_idx]
            base_colors[:] = [0, 0, 0, 1]
            base_colors[i] = [1, 0, 0, 1]
            
            # if stage == 0:  # 红色
            #     base_colors[i] = [1, 0, 0, 1]
            # else:           # 黑色
            #     base_colors[i] = [0, 0, 0, 1]
    
            sc.set_facecolor(base_colors)
            return [sc]
    
        anim = FuncAnimation(fig, update, frames=total_frames,
                             interval=interval, blit=True, repeat=False)
        plt.show()
        return anim

    @staticmethod
    def myPlotHint(df=None, x=None, y=None,c=None, cm='viridis', hint=None,fsize=None,xlim=None,ylim=None,fig=None,axs = None):
        """
        绘制带有悬停提示的二维散点图。
        
        参数
        ----------
        df : ndarray, list, tensor, DataFrame, 默认 None
            数据源，形状 (m, n)，n>=2。当 x、y、cm、hint 为整数时，从 df 取对应列。
        x : int 或 array-like, 默认 None
            横坐标数据。若为 int，表示 df 的列索引；否则直接作为数组。
            若为 None 且 df 非空，则取 df 的第 0 列。
        y : int 或 array-like, 默认 None
            纵坐标数据，规则同 x。若为 None 且 df 非空，则取 df 的第 1 列。
        c : int 或 array-like, 默认 None
            颜色类别数据，用于映射散点颜色。若为 int，表示 df 的列索引；否则直接作为数组。
            若为 None，所有点使用统一颜色（蓝色）。
        hint : int 或 array-like, 默认 None
            悬停提示内容。若为 int，表示 df 的列索引；否则直接作为数组。
            若为 None，则显示坐标信息 (x, y)。
        
        返回
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        # ---------- 辅助：将输入转为 numpy 数组 ----------
        def to_numpy(data):
            if data is None:
                return None
            if isinstance(data, (list, tuple)):
                return np.array(data)
            if hasattr(data, 'values'):  # pandas DataFrame/Series
                return data.values
            if hasattr(data, 'numpy'):   # PyTorch Tensor
                return data.detach().cpu().numpy()
            if isinstance(data, np.ndarray):
                return data
            raise TypeError(f"不支持的数据类型: {type(data)}")
    
        # ---------- 处理 df ----------
        df_np = to_numpy(df)
        if df_np is not None:
            if df_np.ndim != 2:
                raise ValueError("df 必须是二维数组 (m, n)，n>=2")
            if df_np.shape[1] < 2:
                raise ValueError("df 至少需要 2 列")
    
        # ---------- 处理 x ----------
        if x is None and df_np is not None:
            x = 0   # 默认取第 0 列
        if isinstance(x, int):
            if df_np is None:
                raise ValueError("x 为整数但 df 为空")
            x_data = df_np[:, x]
        else:
            x_data = to_numpy(x)
            if x_data is None:
                raise ValueError("x 不能为空")
        x_data = np.asarray(x_data).squeeze()
        if x_data.ndim != 1:
            raise ValueError("x 必须是一维数组")
    
        # ---------- 处理 y ----------
        if y is None and df_np is not None:
            y = 1   # 默认取第 1 列
        if isinstance(y, int):
            if df_np is None:
                raise ValueError("y 为整数但 df 为空")
            y_data = df_np[:, y]
        else:
            y_data = to_numpy(y)
            if y_data is None:
                raise ValueError("y 不能为空")
        y_data = np.asarray(y_data).squeeze()
        if y_data.ndim != 1:
            raise ValueError("y 必须是一维数组")
    
        # 检查 x 和 y 长度一致
        if len(x_data) != len(y_data):
            raise ValueError("x 和 y 长度必须相同")
        n_points = len(x_data)
    
        # ---------- 处理 cm ----------
        if c is None:
            cm_data = None
            colors = 'blue'
        elif isinstance(c, int):
            if df_np is None:
                raise ValueError("c 为整数但 df 为空")
            cm_data = df_np[:, c]
        else:
            cm_data = to_numpy(c)
            if cm_data is not None:
                cm_data = np.asarray(cm_data).squeeze()
                if cm_data.ndim != 1:
                    raise ValueError("c 必须是一维数组")
                if len(cm_data) != n_points:
                    raise ValueError("c 长度与 x 不一致")
            else:
                raise ValueError("c 无效")
    
        # ---------- 处理 hint ----------
        if hint is None:
            hint_data = None
        elif isinstance(hint, int):
            if df_np is None:
                raise ValueError("hint 为整数但 df 为空")
            hint_data = df_np[:, hint]
        else:
            hint_data = to_numpy(hint)
            if hint_data is not None:
                hint_data = np.asarray(hint_data).squeeze()
                if hint_data.ndim != 1:
                    raise ValueError("hint 必须是一维数组")
                if len(hint_data) != n_points:
                    raise ValueError("hint 长度与 x 不一致")
            else:
                raise ValueError("hint 无效")
    
        # ---------- 绘制散点图 ----------
        if axs is None:
            fig, ax = plt.subplots(figsize = (8, 6) if fsize is None else fsize )
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
        else:
            ax = axs
    
        if cm_data is not None:
            # 如果 cm_data 是数值，使用 colormap
            scatter = ax.scatter(x_data, y_data, c=cm_data, cmap=cm, s=50, alpha=0.8)
            cbar = fig.colorbar(scatter, ax=ax, label='Color')
        else:
            scatter = ax.scatter(x_data, y_data, color=colors, s=50, alpha=0.8)
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Scatter Plot with Hover Hint')
    
        # ---------- 添加悬停提示 ----------
        def show_hint(sel):
            idx = sel.index
            if hint_data is not None:
                # 如果 hint_data 可能是字符串数组，直接显示
                hint_text = str(hint_data[idx])
            else:
                # 默认显示坐标
                hint_text = f"x={x_data[idx]:.3f}, y={y_data[idx]:.3f}"
            sel.annotation.set(text=hint_text, fontsize=10)
    
        # 使用 mplcursors 绑定悬停事件
        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", show_hint)
    
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def myPltGradient(arr,totAngle=np.pi,stepCnt=80,fsize=(6,6),xlim=(-3.5,3.5),ylim=(-3.5,3.5)):
        """用于绘制图形的变换过程，一条条显示arr的内容，观察变化过程
        arr：用于展示的ndarray。
        totAngle：总共旋转的角度。
        stepCnt：要进行多少次变换
        """
        fig, ax = plt.subplots(figsize=fsize)
        
        plt.close(fig) # 关闭初始空白图

        angles = np.linspace(0,totAngle,stepCnt)
        for i,j in enumerate(angles):
            fig, ax = plt.subplots(figsize=fsize)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            t = arr@np.array([[np.cos(j),np.sin(j)],[-np.sin(j),np.cos(j)]])
            
            ax.plot(t[:,0],t[:,1],'ro')
            ax.set_title(f'第{i}张图像')
            plt.show()
            time.sleep(1)
            clear_output(wait=True)
    
    @staticmethod
    def myPlotV(A, c='red', fig=None, ax=None, axis=0, arrow_length_ratio=0.1,
                show=True, show_values=True, value_fontsize=9, value_offset=0.2,
                auto_adjust_range=True):
        """
        绘制3D向量（支持交互式旋转 + 端点显示数值 + 多向量分别设色），
        并支持在已有图形上追加向量时自动调整坐标轴范围。
    
        参数：
        A : 需要绘制的向量，支持 1D (3,) 或 2D (3,n)/(n,3)
        c : 颜色，单个颜色值或颜色列表
        fig, ax : 已有的 Figure 和 Axes3D，若为 None 则自动创建
        axis : 向量存储方向，0 列优先，1 行优先
        arrow_length_ratio : 箭头长度比例
        show : 是否显示图像
        show_values : 是否在端点显示数值
        value_fontsize, value_offset : 数值标签样式
        auto_adjust_range : bool, 默认 True
            若为 True，且传入了 ax，则自动从 ax 中提取已有散点/线条数据，
            并重新计算坐标轴范围，确保所有元素（旧点 + 新向量）完整可见。
            若为 False，则使用原逻辑（基于向量自身范围，最大 15）。
        
        返回：
        fig, ax
        """
        # ---------- 数据预处理 ----------
        A = np.asarray(A)
        if A.ndim not in (1, 2):
            raise ValueError("输入向量A必须是1D或2D数组")
        if A.ndim == 1:
            if len(A) != 3:
                raise ValueError("1D向量长度必须为3")
            A = A.reshape(3, 1)
        else:
            if axis == 1:
                A = A.T
            if A.shape[0] != 3:
                raise ValueError(f"axis={axis}时，向量维度必须为3（当前为{A.shape[0]}）")
        
        n_vectors = A.shape[1]
        if isinstance(c, (list, np.ndarray)):
            if len(c) != n_vectors:
                raise ValueError(f"颜色列表长度({len(c)})必须等于向量数量({n_vectors})")
            color_list = c
        else:
            color_list = [c] * n_vectors
    
        # ---------- 图形创建 ----------
        if ax is None:
            plt.close('all')
            # try:
            #     plt.switch_backend('module://ipympl.backend_nbagg')
            # except:
            #     pass
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.mouse_init()
        else:
            fig = ax.figure
    
        # ---------- 绘制向量 ----------
        for i in range(n_vectors):
            x, y, z = A[0, i], A[1, i], A[2, i]
            curr_color = color_list[i]
            ax.quiver(0, 0, 0, x, y, z,
                      color=curr_color,
                      arrow_length_ratio=arrow_length_ratio,
                      label=f"vector {i}")
            if show_values:
                dx = np.sign(x) * value_offset if x != 0 else value_offset
                dy = np.sign(y) * value_offset if y != 0 else value_offset
                dz = np.sign(z) * value_offset if z != 0 else value_offset
                ax.text(x + dx, y + dy, z + dz,
                        f"({x:.2f}, {y:.2f}, {z:.2f})",
                        fontsize=value_fontsize, color=curr_color,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8))
        ax.legend()
    
        # ---------- 自动调整坐标轴范围（基于当前范围 + 向量端点） ----------
        if auto_adjust_range:
            # 获取当前坐标轴范围（如果未设置，则从数据推断）
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            # 如果当前范围是自动计算的（即没有手动设置），它们已经是数据的范围
            # 但我们仍需要确保范围存在（避免全为0）
            x_min, x_max = xlim
            y_min, y_max = ylim
            z_min, z_max = zlim
    
            # 加入向量端点
            vec_x = A[0, :]
            vec_y = A[1, :]
            vec_z = A[2, :]
            x_min = min(x_min, vec_x.min())
            x_max = max(x_max, vec_x.max())
            y_min = min(y_min, vec_y.min())
            y_max = max(y_max, vec_y.max())
            z_min = min(z_min, vec_z.min())
            z_max = max(z_max, vec_z.max())
    
            # 处理重合情况
            if np.isclose(x_max, x_min):
                x_min, x_max = x_min - 1, x_max + 1
            if np.isclose(y_max, y_min):
                y_min, y_max = y_min - 1, y_max + 1
            if np.isclose(z_max, z_min):
                z_min, z_max = z_min - 1, z_max + 1
    
            margin = 0.05
            dx = (x_max - x_min) * margin
            dy = (y_max - y_min) * margin
            dz = (z_max - z_min) * margin
    
            ax.set_xlim(x_min - dx, x_max + dx)
            ax.set_ylim(y_min - dy, y_max + dy)
            ax.set_zlim(z_min - dz, z_max + dz)
    
            # 设置等比例显示
            try:
                ax.set_box_aspect([1, 1, 1])
            except AttributeError:
                pass
            ax.autoscale_view(scalex=False, scaley=False, scalez=False)
    
        else:
            # 原逻辑：基于向量自身范围，最大15
            m = min(np.max(np.abs(A)), 15)
            ax.set_xlim(-m, m)
            ax.set_ylim(-m, m)
            ax.set_zlim(-m, m)
    
        # ---------- 统一显示（只在此处显示一次） ----------
        if show:
            # try:
            from IPython.display import display
            display(fig)
            # except ImportError:
            #     plt.show(block=False)
    
        return fig, ax

    @staticmethod
    def myPlotVField(A, range_val=(-4, 4), scale_factor=1.0, normalize=False, cmap='coolwarm', fig=None, ax=None):
        """
        绘制二维线性系统的向量场： d/dt [x,y]^T = A [x,y]^T
            返回figure,ax
        参数
        ----------
        A : 二维方阵 (list, ndarray, DataFrame, Tensor)
        range_val : 范围，默认 (-4,4)
        scale_factor : 浮点数，默认 1.0。当 normalize=False 时，用作整体长度微调（乘以自动缩放后的长度）
        normalize : bool, 默认 False。若为 True，所有箭头长度一致，颜色表示速度大小（推荐用于差异大的系统）
        cmap : 颜色映射，默认 'coolwarm'（以0为中心，正负对比鲜明）
        fig, ax : 图形对象，默认 None 自动创建
        """
        # -------------------- 1. 数据预处理 --------------------
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=float)
        if A.shape != (2, 2):
            raise ValueError("A 必须是 2×2 矩阵")
        
        # 解析范围
        if np.isscalar(range_val):
            r = float(range_val)
            xmin, xmax = -r, r
        else:
            xmin, xmax = float(range_val[0]), float(range_val[1])
        ymin, ymax = xmin, xmax
        
        n_pts = 20
        x = np.linspace(xmin, xmax, n_pts)
        y = np.linspace(ymin, ymax, n_pts)
        X, Y = np.meshgrid(x, y)
        
        # 计算速度向量
        points = np.vstack([X.ravel(), Y.ravel()])
        velocities = A @ points
        U = velocities[0, :].reshape(X.shape)
        V = velocities[1, :].reshape(X.shape)
        speed = np.hypot(U, V)
        
        # -------------------- 2. 处理箭头长度 --------------------
        if normalize:
            # 归一化模式：统一长度，颜色表示速度大小
            mask_zero = (speed == 0)
            U_dir = np.divide(U, speed, out=np.zeros_like(U), where=~mask_zero)
            V_dir = np.divide(V, speed, out=np.zeros_like(V), where=~mask_zero)
            # 固定长度取网格间距的 0.6 倍，再乘以 scale_factor
            grid_step = x[1] - x[0]
            arrow_len = grid_step * 0.6 * scale_factor
            U_plot = U_dir * arrow_len
            V_plot = V_dir * arrow_len
        else:
            # 非归一化模式：速度向量直接用于绘图，但让 quiver 自动缩放（不设 scale=1 和 units）
            # 我们仍可乘以 scale_factor 来微调
            U_plot = U * scale_factor
            V_plot = V * scale_factor
            # 注意：这里不设 scale 和 units，quiver 会使用 autoscale
            # 但如果用户想手动控制缩放，可设 scale 参数，但本例不添加该参数
        
        # -------------------- 3. 图形管理 --------------------
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        elif ax is not None:
            fig = ax.figure
        else:
            ax = fig.add_subplot(111)
        
        # -------------------- 4. 绘图 --------------------
        # 关键：当 normalize=False 时，不设置 scale=1, units='xy'，让 matplotlib 自动缩放
        # 但若 normalize=True，我们已手动计算了长度，仍用 scale=1, units='xy' 保证严格长度
        if normalize:
            q = ax.quiver(X, Y, U_plot, V_plot, speed,
                          cmap=cmap, alpha=0.85,
                          scale=1, units='xy', width=0.008)
        else:
            # 自动缩放，只需把 U_plot, V_plot 传进去，quiver 会自己调整长度
            q = ax.quiver(X, Y, U_plot, V_plot, speed,
                          cmap=cmap, alpha=0.85,
                          width=0.01)  # 调大箭头杆宽度
            
        # 绘制流线（灰色辅助）
        ax.streamplot(X, Y, U, V, color='lightgray', linewidth=0.8, density=1.2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Vector Field')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis('equal')
        
        cbar = fig.colorbar(q, ax=ax, label='Speed')
        return fig, ax
    
    @staticmethod
    def myPltNumNum(df, cols, mdl='model_name', ctheme='plasma', figsize=None, alpha=0.6, s=50):
        """
        绘制属性两两配对散点图（Pair Plot），不同型号用不同颜色区分
        
        参数：
        df：数据DataFrame
        cols：要绘制的属性列名列表（长度需≥2）
        mdl：分类列名（用于颜色区分，默认'model_name'）
        ctheme：颜色映射主题（默认'plasma'）
        figsize：画布大小，默认(5*len(cols), 5*len(cols))
        alpha：散点透明度（默认0.6）
        s：散点大小（默认50）
        
        返回：
        fig：Figure对象
        ax_arr：Axes二维数组
        """
        # 输入验证
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df必须是pandas DataFrame类型")
        if len(cols) < 2:
            raise ValueError("cols列表长度必须≥2")
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"属性列 '{col}' 不存在于df中")
        if mdl not in df.columns:
            raise ValueError(f"分类列 '{mdl}' 不存在于df中")
        
        # 预处理数据：确保cols为数值型，去除缺失值
        plot_df = df[cols + [mdl]].copy()
        for col in cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        plot_df = plot_df.dropna(subset=cols)
        
        # 初始化颜色映射
        models = plot_df[mdl].unique()
        cmap = plt.get_cmap(ctheme, len(models))
        model_color_map = dict(zip(models, cmap(np.linspace(0, 1, len(models)))))
        point_colors = plot_df[mdl].map(model_color_map)
        
        # 初始化画布和轴
        m = len(cols)
        if figsize is None:
            figsize = (5 * m, 5 * m)
        fig, ax_arr = plt.subplots(m, m, figsize=figsize)
        
        # 遍历绘制配对图
        for i in range(m):  # 行：Y轴
            for j in range(m):  # 列：X轴
                ax = ax_arr[i, j]
                if i > j:
                    # 下三角：绘制散点图
                    ax.scatter(plot_df[cols[j]], plot_df[cols[i]], 
                               c=point_colors, alpha=alpha, s=s, edgecolors='white', linewidth=0.5)
                    # 设置轴标签
                    ax.set_xlabel(cols[j], fontsize=10)
                    ax.set_ylabel(cols[i], fontsize=10)
                elif i == j:
                    # 对角线：绘制直方图（按型号分色）
                    for model in models:
                        data = plot_df[plot_df[mdl] == model][cols[i]]
                        ax.hist(data, bins=20, alpha=0.5, color=model_color_map[model], label=model)
                    ax.set_xlabel(cols[i], fontsize=10)
                    ax.set_ylabel('频数', fontsize=10)
                    if i == 0:  # 只在第一个对角线图显示图例
                        ax.legend(fontsize=8, loc='upper right')
                else:
                    # 上三角：隐藏轴
                    ax.axis('off')
                # 设置标题
                ax.set_title(f"{cols[i]} vs {cols[j]}", fontsize=11, pad=10)
        
        plt.tight_layout()
        return fig, ax_arr

    @staticmethod
    def myPltTypeNum(df, atts, mdl='model_name', figsize=None, jitter=0.8, legend_outside=True):
        """
        绘制不同型号下各属性的散点分布图（X轴为型号，Y轴为属性值，同型号点轻微错开避免重叠）
        
        参数：
        df：数据DataFrame
        atts：要绘制的Y轴属性列名列表
        mdl：X轴分类列名（默认'model_name'）
        figsize：画布大小，默认(10, 6*len(atts))
        jitter：同型号点的错开宽度（0-1之间，避免不同型号点重叠）
        legend_outside：是否将图例放在图外（避免遮挡数据）
        
        返回：
        fig：Figure对象
        ax：Axes数组（或单个Axes对象）
        """
        # 输入验证
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df必须是pandas DataFrame类型")
        if mdl not in df.columns:
            raise ValueError(f"列 '{mdl}' 不存在于df中")
        for col in atts:
            if col not in df.columns:
                raise ValueError(f"属性列 '{col}' 不存在于df中")
        
        # 初始化画布和轴
        attscn = len(atts)
        if figsize is None:
            figsize = (10, 6 * attscn)
        fig, ax_arr = plt.subplots(attscn, 1, figsize=figsize)
        ax_arr = np.atleast_1d(ax_arr)  # 兼容attscn=1的情况
        
        # 获取唯一型号并预生成颜色
        models = df[mdl].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        model_color_map = dict(zip(models, colors))
        
        # 遍历每个属性绘图
        for i, col in enumerate(atts):
            ax = ax_arr[i]
            # 确保属性列为数值型
            plot_data = pd.to_numeric(df[col], errors='coerce').dropna()
            plot_df = df.loc[plot_data.index, [mdl, col]].copy()
            plot_df[col] = plot_data
            
            # 遍历每个型号绘制散点
            for i2, model in enumerate(models):
                t = plot_df[plot_df[mdl] == model]
                if len(t) == 0:
                    continue
                # 生成错开的X轴位置（同型号点在[i2, i2+jitter]区间均匀分布）
                x_pos = i2 + np.linspace(0, jitter, len(t), endpoint=False)
                ax.scatter(x_pos, t[col], color=model_color_map[model], 
                           label=model, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # 设置轴属性（放在循环外，避免重复设置）
            ax.set_xlabel(mdl, fontsize=10)
            ax.set_ylabel(col, fontsize=10)
            ax.set_title(f"属性 {col} vs 类别 {mdl}", fontsize=12, pad=15)
            # 设置X轴刻度和标签
            ax.set_xticks(np.arange(len(models)) + jitter/2)
            ax.set_xticklabels(models, rotation=45, ha='right')
            # 添加图例
            if legend_outside:
                ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            else:
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        return fig, ax_arr

    @staticmethod
    def myCompare_ori(df1,df2,retflag=1,ecols=None):
        """ 
        功能：比较两个dataframe：df1和df2 的值：是df1中的每一行和df2中的每一行进行比较，根据retflag 返回不同的数据
        retflag：1表示只返回不同的数据，相同的全部为空，默认为1
                     2表示只返回相同的数据，不同的全部为空，
                     3表示既返回相同的数据也返回不同的数据
        ecols: 不参与比较的列
        """
        assert isinstance(df1,pd.DataFrame) and isinstance(df2,pd.DataFrame)
    
        if isinstance(ecols,list):
            edf = df2[ecols]
            edf.columns = [f"{i}_" for i in edf.columns]
            result = pd.concat((edf,df2),axis=1).iloc[0:0,:]
        else:
            result = df2.iloc[0:0,:]
        for i in range(len(df1)):
            tdf = df1.iloc[i:i+1,:].sample(len(df2),replace=True)
            tdf.index = df2.index
    
            _1 = tdf.isna().astype(np.int8)
            _2 = df2.isna().astype(np.int8)
            _3 = _1 + _2
            _1 = (_3 == 2)
            _2 = (tdf == df2) + _1
            _3 = tdf.astype(str) + "~" + df2.astype(str)
            if retflag == 1:
                _3 = _3.where(~_2)
            elif retflag == 2:
                _3 = df2.where(tdf == df2)
            else:
                _3 = df2.where(_2,other=_3)
            # _3 = (df1.astype(str) + "-" + df2.astype(str)).where(~_2)
            if isinstance(ecols,list):
                result = pd.concat((result,pd.concat((edf,_3),axis=1)) )
            else:
                result = pd.concat((result,_3))
        return result

    @staticmethod
    def myCompare(df1, df2, retflag=1, ecols=None):
        """
        功能：比较两个DataFrame的每一行组合（df1的每一行 vs df2的每一行）
        参数：
        df1, df2：待比较的两个DataFrame，必须有相同的列
        retflag：返回模式
            1：只显示不同值（格式："df1值~df2值"），相同值为空
            2：只显示相同值（取df2的值），不同值为空
            3：同时显示，相同取df2值，不同显示"df1值~df2值"
        ecols：需在结果中额外复制展示的列（列名加"_"后缀），注意：这些列仍会参与比较
        返回：
        比较结果DataFrame，行数为len(df1)*len(df2)
        """
        # 输入验证
        if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
            raise TypeError("df1和df2必须是pandas DataFrame类型")
        if not df1.columns.equals(df2.columns):
            raise ValueError("df1和df2的列必须一致")
        if retflag not in (1, 2, 3):
            raise ValueError("retflag必须是1、2或3")
        if ecols is not None and not isinstance(ecols, list):
            raise TypeError("ecols必须是列表或None")
    
        # 重置索引避免索引对齐问题
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        n1, n2 = len(df1), len(df2)
    
        # 向量化扩展DataFrame（替代原循环，大幅提升效率）
        # df1_exp：df1的每一行重复n2次（形状：n1*n2, m）
        df1_exp = df1.loc[df1.index.repeat(n2)].reset_index(drop=True)
        # df2_exp：df2整体重复n1次（形状：n1*n2, m）
        df2_exp = df2.iloc[np.tile(np.arange(n2), n1)].reset_index(drop=True)
    
        # 处理ecols：复制df2的指定列并添加后缀
        ecols_exp = None
        if ecols:
            ecols_exp = df2_exp[ecols].copy()
            ecols_exp.columns = [f"{col}_" for col in ecols_exp.columns]
    
        # 计算相等掩码（同时处理"两个都是NaN视为相等"的逻辑）
        eq_mask = (df1_exp == df2_exp) | (df1_exp.isna() & df2_exp.isna())
    
        # 生成不同值的字符串表示
        diff_str = df1_exp.astype(str) + "~" + df2_exp.astype(str)
    
        # 根据retflag生成结果（修复原代码retflag==2时未处理双NaN的逻辑不一致问题）
        if retflag == 1:
            comp_result = diff_str.where(~eq_mask)
        elif retflag == 2:
            comp_result = df2_exp.where(eq_mask)  # 这里改为eq_mask，与其他模式保持一致
        else:  # retflag == 3
            comp_result = df2_exp.where(eq_mask, other=diff_str)
    
        # 合并ecols和比较结果
        if ecols_exp is not None:
            result = pd.concat([ecols_exp, comp_result], axis=1)
        else:
            result = comp_result
    
        return result

    @staticmethod
    def myDfReplace(df, oldval, newval, cols='all', rows='all', inplace=False):
        """
        功能：在 DataFrame 的指定行列中查找并替换值
        
        参数：
        df：需要处理的 DataFrame
        oldval：被替换的旧值（支持 np.nan）
        newval：用于替换的新值
        cols：指定列，默认为 'all'（所有列），也可传入列名列表
        rows：指定行，默认为 'all'（所有行），也可传入行索引列表或布尔数组
        inplace：是否直接修改原 DataFrame，默认为 False（返回副本）
        
        返回：
        替换后的 DataFrame（如果 inplace=True 则返回 None）
        """
        # 输入验证
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas DataFrame 类型")
        
        # 1. 确定是否原地修改
        result_df = df if inplace else df.copy()
        
        # 2. 确定目标列
        if cols == 'all':
            target_cols = result_df.columns
        else:
            # 确保 cols 是可迭代的且列存在
            if isinstance(cols, str):
                cols = [cols]
            target_cols = [c for c in cols if c in result_df.columns]
            if not target_cols:
                raise ValueError("指定的 cols 中没有有效的列名")
        
        # 3. 确定目标行
        if rows == 'all':
            row_mask = slice(None)  # 表示所有行
        else:
            # 支持行索引列表或布尔数组
            row_mask = rows
        
        # 4. 执行替换（利用 Pandas 内置方法，更高效且易读）
        if pd.isna(oldval):
            # 替换 NaN
            result_df.loc[row_mask, target_cols] = result_df.loc[row_mask, target_cols].fillna(newval)
        else:
            # 替换普通值
            result_df.loc[row_mask, target_cols] = result_df.loc[row_mask, target_cols].replace(oldval, newval)
        
        return None if inplace else result_df

    @staticmethod
    def myVectorAngle(arr, B=None, radian='N'):
        """
        计算两个向量之间的夹角。默认返回角度
        参数
        ----------
        arr : ndarray, DataFrame, Tensor, list, tuple
            输入数据。
            - 当 B 为 None 时，arr 必须是形状 (N, 2) 的二维数组，两列分别作为两个向量。
            - 当 B 不为 None 时，arr 可以是形状 (N,) 或 (N,1) 的一维向量，与 B 形状一致。
        B : 可选，默认 None
            第二个向量。若提供，则与 arr 共同计算夹角；arr 和 B 必须形状相同且为一维。
        radian : str, 默认 'N'
            控制输出单位：'Y' 表示弧度，'N' 表示角度（度）。
    
        返回
        -------
        float
            夹角（单位由 radian 决定）。
        """
        # ---------- 辅助：将输入转换为 numpy 数组 ----------
        def to_numpy(data):
            if isinstance(data, (list, tuple)):
                return np.array(data)
            elif isinstance(data, pd.DataFrame):
                return data.values
            elif hasattr(data, 'numpy'):  # PyTorch Tensor
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                raise TypeError(f"不支持的数据类型: {type(data)}")
    
        # ---------- 处理 arr ----------
        arr_np = to_numpy(arr)
        # 确保是浮点型
        arr_np = arr_np.astype(np.float64)
    
        # ---------- 处理 B ----------
        if B is not None:
            B_np = to_numpy(B)
            B_np = B_np.astype(np.float64)
            # 确保 arr 和 B 都是一维（或可 squeeze 为一维）
            arr_np = arr_np.squeeze()
            B_np = B_np.squeeze()
            if arr_np.ndim != 1 or B_np.ndim != 1:
                raise ValueError("当 B 不为 None 时，arr 和 B 必须为一维向量或可压扁为一维")
            if arr_np.shape != B_np.shape:
                raise ValueError("arr 和 B 的形状必须相同")
            v1 = arr_np
            v2 = B_np
        else:
            # B 为 None，计算 arr 的两个列向量
            if arr_np.ndim != 2:
                raise ValueError("当 B 为 None 时，arr 必须是二维数组")
            if arr_np.shape[1] != 2:
                raise ValueError("当 B 为 None 时，arr 必须有两列（形状 (N, 2)）")
            v1 = arr_np[:, 0]
            v2 = arr_np[:, 1]
    
        # ---------- 检查向量是否非零 ----------
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            raise ValueError("零向量无法计算夹角")
    
        # ---------- 计算夹角 ----------
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        # 防止数值误差导致超出 [-1, 1]
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
    
        # ---------- 返回结果 ----------
        if radian == 'Y':
            return angle_rad,v1,v2
        elif radian == 'N':
            return np.degrees(angle_rad)
        else:
            raise ValueError("radian 参数必须是 'Y' 或 'N'")

    @staticmethod
    def myVAngleRef(arr,B=None,refv=None,radian='N'):
        """输出两个向量的角度，除了refv参数，其它同myVectorAngle
        核心在于体现哪个方向的角度，如果这两个向量的夹角大于180°，会导致
        refv：是和v1，v2同平面的一个向量。
        """
        theta,v1,v2 = MyTools.myVectorAngle(arr,B,radian='Y')
        if B is None:
            return np.degrees(theta) if radian=='N' else theta
        else:
            vr1 = MyTools.myVRectV(refv,v2)
            vr2 = MyTools.myVRectV(v1,v2)
            if MyTools.myVectorAngle(vr1,vr2) < 1:
                return np.degrees(theta) if radian=='N' else theta
            else:
                angle = np.pi * 2 - theta
                return np.degrees(angle) if radian=='N' else angle
    
    @staticmethod
    def myVectorProject(arr, B=None):
        """
        计算A向量投影到B向量,如果B为空，那么就是arr的第一列向量投影到第二列向量
        参数
        ----------
        arr : ndarray, DataFrame, Tensor, list, tuple
            输入数据。
            - 当 B 为 None 时，arr 必须是形状 (N, 2) 的二维数组，两列分别作为两个向量。
            - 当 B 不为 None 时，arr 可以是形状 (N,) 或 (N,1) 的一维向量，与 B 形状一致。
        B : 可选，默认 None
            第二个向量。若提供，则arr投影到B；arr 和 B 必须形状相同且为一维。
    
        返回
        -------
        向量 ：和B向量方向完全一致
        """
        # ---------- 辅助：将输入转换为 numpy 数组 ----------
        def to_numpy(data):
            if isinstance(data, (list, tuple)):
                return np.array(data)
            elif isinstance(data, pd.DataFrame):
                return data.values
            elif hasattr(data, 'numpy'):  # PyTorch Tensor
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                raise TypeError(f"不支持的数据类型: {type(data)}")
    
        # ---------- 处理 arr ----------
        arr_np = to_numpy(arr)
        # 确保是浮点型
        arr_np = arr_np.astype(np.float64)
    
        # ---------- 处理 B ----------
        if B is not None:
            B_np = to_numpy(B)
            B_np = B_np.astype(np.float64)
            # 确保 arr 和 B 都是一维（或可 squeeze 为一维）
            arr_np = arr_np.squeeze()
            B_np = B_np.squeeze()
            if arr_np.ndim != 1 or B_np.ndim != 1:
                raise ValueError("当 B 不为 None 时，arr 和 B 必须为一维向量或可压扁为一维")
            if arr_np.shape != B_np.shape:
                raise ValueError("arr 和 B 的形状必须相同")
            v1 = arr_np
            v2 = B_np
        else:
            # B 为 None，计算 arr 的两个列向量
            if arr_np.ndim != 2:
                raise ValueError("当 B 为 None 时，arr 必须是二维数组")
            if arr_np.shape[1] != 2:
                raise ValueError("当 B 为 None 时，arr 必须有两列（形状 (N, 2)）")
            v1 = arr_np[:, 0]
            v2 = arr_np[:, 1]
    
        r = np.dot(v1,v2)
        if np.abs(r) < 1e-9:
            return np.zeros(len(v1))
        else:
            return (r * v2)/np.dot(v2,v2)
    
    @staticmethod
    def myVectorArea(arr, B=None):
        """
        计算两个向量之间的面积。
    
        参数
        ----------
        arr : ndarray, DataFrame, Tensor, list, tuple
            输入数据。
            - 当 B 为 None 时，arr 必须是形状 (N, 2) 的二维数组，两列分别作为两个向量。
            - 当 B 不为 None 时，arr 可以是形状 (N,) 或 (N,1) 的一维向量，与 B 形状一致。
        B : 可选，默认 None
            第二个向量。若提供，则与 arr 共同计算夹角；arr 和 B 必须形状相同且为一维。
        返回
        -------
        float
            返回两个向量组成的平行四边形面积。
        """
        # ---------- 辅助：将输入转换为 numpy 数组 ----------
        def to_numpy(data):
            if isinstance(data, (list, tuple)):
                return np.array(data)
            elif isinstance(data, pd.DataFrame):
                return data.values
            elif hasattr(data, 'numpy'):  # PyTorch Tensor
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                raise TypeError(f"不支持的数据类型: {type(data)}")
    
        # ---------- 处理 arr ----------
        arr1 = to_numpy(arr)
        # 确保是浮点型
        arr1 = arr1.astype(np.float64)
        if B is not None:
            arr2 = to_numpy(B)
            arr = np.column_stack((arr1,arr2))
        else:
            arr = arr1
        return np.sqrt(np.linalg.det(arr.T@arr))

    @staticmethod
    def myVRectV(A,B,d='N'):
        """求垂直于B的向量，此向量是A向量的投影
        d：表示是否返回 A@B的值
        """
        if B is None:
            B = A[:,1]
            A = A[:,0]
        r = np.dot(A,B)
        if d == 'Y':
            return A - r*B/np.dot(B,B),r
        else:
            return A - r*B/np.dot(B,B)
        
    @staticmethod
    def myVrotaion(A,B=None,theta=np.pi/2):
        """A向量向B向量旋转theta角（弧度）"""
        if B is None:
            B = A[:,1]
            A = A[:,0]
        ay,r = MyTools.myVRectV(A,B,d='Y')
        r = -1 if r < 0 else 1
        ax = r * np.linalg.norm(A - ay)  # A 投影到B向量的长度
        A = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        rst = A@np.array([r*np.linalg.norm(ax),np.linalg.norm(ay)])  # A分解 到(B,B垂直)的矩阵。然后进行旋转对应角度
        rst = B*rst[0]/np.linalg.norm(B) + ay*rst[1]/np.linalg.norm(ay) #针对 (B,B垂直)这个坐标，然后还原多维坐标
        return rst

    @staticmethod
    def myVRotation3D(axs, point, angle=np.pi/2):
        """
        在三维空间中绕指定轴向量旋转一个点/向量。
    
        参数
        ----------
        axs : array_like, shape (3,)
            旋转轴向量（不必为单位向量）。
        point : array_like, shape (3,)
            待旋转的点/向量。
        angle : float, 默认 np.pi/2
            旋转角度（弧度）。正方向遵循右手定则。
    
        返回
        -------
        np.ndarray, shape (3,)
            旋转后的向量。
        """
        # 转换为 NumPy 数组
        axs = np.asarray(axs, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)
        
        if axs.shape != (3,) or point.shape != (3,):
            raise ValueError("轴向量和点向量必须为长度为 3 的一维数组")
        
        # 归一化旋转轴
        norm = np.linalg.norm(axs)
        if norm == 0:
            raise ValueError("旋转轴不能为零向量")
        k = axs / norm
        
        # 计算三角函数
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 罗德里格旋转公式
        term1 = point * cos_a
        term2 = np.cross(k, point) * sin_a
        term3 = k * (np.dot(k, point)) * (1 - cos_a)
        
        return term1 + term2 + term3
    
    @staticmethod
    def myVQuatRotaion(ax,angle,point,rad=None):
        """使用四元素进行旋转：围绕指定轴ax旋转对应的点point 。旋转角度为angle
        ax:旋转的轴；
        angle：旋转的角度；
        point：旋转的点；
        rad：是否使用弧度，默认为None,表示直接根据angle值判断，如果小于pi那么就是弧度，否则是角度"""
        def quat_mult(q1, q2):
            """两个四元数的相乘"""
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return np.array([w, x, y, z])
        def quat_conj(q):
            """求四元素的共轭，也用于求逆"""
            w, x, y, z = q
            return np.array([w, -x, -y, -z])
        def axis_angle_to_quat(axis, angle):
            """把旋转轴和旋转角度 转换为四元数"""
            axis = np.array(axis)
            axis = axis / np.linalg.norm(axis)
            half = angle / 2
            s = np.sin(half)
            w = np.cos(half)
            x = axis[0] * s
            y = axis[1] * s
            z = axis[2] * s
            return np.array([w, x, y, z])
        def rotate_point(q, p):
            """将需要旋转的点转换为四元数P，乘以 旋转轴和角度形成四元素Q，然后乘以Q的共轭四元数"""
            q_p = np.array([0, p[0], p[1], p[2]])
            q_conj = quat_conj(q)
            res = quat_mult(quat_mult(q, q_p), q_conj)
            return res[1:]
        if rad is not None and rad not in ('R','D'):
            rad = None
        if angle >= np.pi and rad is None:
            q = axis_angle_to_quat(ax, angle*np.pi/180)
        elif angle < np.pi and rad is None:
            q = axis_angle_to_quat(ax, angle)
        elif rad == 'D':
            q = axis_angle_to_quat(ax, angle*np.pi/180)
        else:
            q = axis_angle_to_quat(ax, angle)
        # q_conj = quat_conj(q)
        p_rot = rotate_point(q, point)
        return p_rot
    
    @staticmethod
    def myCumCnt(df, calCol, typecol, cumflag='Y'):
        """
        按类别分组，对数值列进行频数统计，默认是每个值对应种类的个数（非累计）。
    
        参数
        ----------
        df : ndarray, DataFrame, torch.Tensor 或 None
            数据源。当 calCol 或 typecol 为整数时，df 不能为 None。
        calCol : int, str 或 可迭代对象
            - 若为 int：从 df 中取第 calCol 列（0-based）。
            - 若为 str：仅当 df 是 DataFrame 且列名存在时有效。
            - 否则：视为可迭代对象，直接作为数值数据（此时 df 可忽略）。
        typecol : int, str 或 可迭代对象
            类别列，规则同 calCol。
        cumflag : str, 默认 'Y'
            - 'Y'：返回累计频数（小于等于该数值的样本个数）。
            - 'N'：返回每个 (category, value) 组合的原始频数。
    
        返回
        -------
        pd.DataFrame
            宽表格式，行索引为类别，列名为数值（升序），值为频数（原始或累计）。
            若 cumflag='Y'，每行值单调非减，最后一列等于该类别的总样本数。
            若 cumflag='N'，每行值表示该类别下该数值的频数（可能为0）。
        """
        def is_iterable_not_str(obj):
            return hasattr(obj, '__iter__') and not isinstance(obj, str)
    
        # ---------- 1. 参数合法性检查 ----------
        if df is None:
            if isinstance(calCol, int) or isinstance(typecol, int):
                raise ValueError("当 df 为 None 时，calCol 和 typecol 不能为整数，必须是可迭代对象")
    
        # 辅助：判断参数是否应从 df 提取
        def should_extract_from_df(param):
            if isinstance(param, int):
                return True
            if isinstance(df, pd.DataFrame) and isinstance(param, str):
                return True
            return False
    
        # 从数据源提取列（支持 ndarray, DataFrame, Tensor）
        def extract_column(data_source, col_spec):
            if isinstance(data_source, np.ndarray):
                if not isinstance(col_spec, int):
                    raise TypeError("对于 ndarray 数据源，列指定必须为整数索引")
                return data_source[:, col_spec]
            elif isinstance(data_source, pd.DataFrame):
                if isinstance(col_spec, int):
                    return data_source.iloc[:, col_spec]
                elif isinstance(col_spec, str):
                    return data_source[col_spec]
                else:
                    raise TypeError("对于 DataFrame，列指定必须为整数或字符串")
            elif hasattr(data_source, 'numpy') and hasattr(data_source, 'detach'):
                import torch
                if isinstance(data_source, torch.Tensor):
                    if not isinstance(col_spec, int):
                        raise TypeError("对于 Tensor 数据源，列指定必须为整数索引")
                    return data_source.detach().cpu().numpy()[:, col_spec]
                else:
                    raise TypeError("不支持的数据类型")
            else:
                raise TypeError("df 类型不支持，必须是 ndarray, DataFrame 或 torch.Tensor")
    
        # ---------- 2. 提取 calCol ----------
        if should_extract_from_df(calCol):
            if df is None:
                raise ValueError("calCol 指定从 df 提取列，但 df 为 None")
            value_data = extract_column(df, calCol)
        else:
            if not is_iterable_not_str(calCol):
                raise ValueError("calCol 必须为可迭代对象（非字符串）")
            value_data = np.asarray(calCol)
    
        # ---------- 3. 提取 typecol ----------
        if should_extract_from_df(typecol):
            if df is None:
                raise ValueError("typecol 指定从 df 提取列，但 df 为 None")
            category_data = extract_column(df, typecol)
        else:
            if not is_iterable_not_str(typecol):
                raise ValueError("typecol 必须为可迭代对象（非字符串）")
            category_data = np.asarray(typecol)
    
        if len(value_data) != len(category_data):
            raise ValueError("数值数据和类别数据长度必须相等")
    
        data_df = pd.DataFrame({'value': value_data, 'category': category_data})
    
        # ---------- 4. 分组统计 ----------
        counts = data_df.groupby(['category', 'value']).size().reset_index(name='count')
        counts = counts.sort_values(['category', 'value'])
    
        # ---------- 5. 根据 cumflag 决定返回的数值 ----------
        if cumflag == 'Y':
            # 累计模式：计算累计频数
            counts['value_to_show'] = counts.groupby('category')['count'].cumsum()
        else:  # 'N'
            # 原始频数
            counts['value_to_show'] = counts['count']
    
        # ---------- 6. 转换为宽表 ----------
        pivot = counts.pivot(index='category', columns='value', values='value_to_show')
        all_values = sorted(counts['value'].unique())
        pivot = pivot.reindex(columns=all_values)
    
        if cumflag == 'Y':
            # 累计模式需要向前填充并置0
            pivot = pivot.ffill(axis=1)
            pivot = pivot.fillna(0)
        else:
            # 原始频数模式：缺失值（即该类别下没有出现该数值）填0
            pivot = pivot.fillna(0)
    
        pivot = pivot.astype(int)
        pivot.index.name = 'category'
        return pivot
    
    @staticmethod
    def mySearchDf(df, target, regex=False):
        """
        在整个 DataFrame 中查找 target。
        参数:
            target : 查找内容
            regex  : False → 精确匹配（两边都转成 str 再比较）
                     True  → 把 target 当成正则表达式（re.search 语义），
                             也就是你直接传 r'string' 里的那个 string
        返回: list，每项是单键 dict：
            列名命中 → {列名: [前3行的值]}          # 不足3行返回全部
            值命中   → {列名: [行索引, 值]}
        """
        results = []

        def col_hit(name):
            s = str(name)
            return re.search(target, s) is not None if regex else s == str(target)

        # 1) 列名命中
        for col in df.columns:
            if col_hit(col):
                results.append({col: df[col].head(3).tolist()})

        # 2) 值命中
        for col in df.columns:
            if regex:
                mask = df[col].astype(str).str.contains(target, regex=True, na=False)
            else:
                mask = df[col].astype(str) == str(target)
            for idx in df.index[mask]:
                results.append({col: [idx, df.at[idx, col]]})

        return results
    
# -------------------------- 测试示例 --------------------------
if __name__ == "__main__":
    #下面测试myCopare

    #下面测试myplt1，myPlt2
    # 示例数据（需替换为你的实际数据）
    np.random.seed(42)
    df_example = pd.DataFrame({
        'model_name': np.random.choice(['Model_A', 'Model_B', 'Model_C'], 100),
        'CTE': np.random.randn(100)*10 + 50,
        'DF': np.random.randn(100)*0.5 + 3,
        'DK': np.random.randn(100)*2 + 10,
        'Density': np.random.randn(100)*0.1 + 1.5
    })
    
    # 调用 myPlt1
    fig1, ax1 = myPlt1(df_example, atts=['CTE', 'DF', 'DK'], mdl='model_name')
    
    # 调用 myPlt2
    fig2, ax2 = myPlt2(df_example, cols=['CTE', 'DF', 'DK', 'Density'], mdl='model_name')
    
    #下面测试myPlotP
    # 示例1：绘制单个向量
    v = np.array([1, 2, 3])
    myPlotP(v)
    
    # 示例2：绘制多个行向量（需设置axis=1）
    vs = np.array([[1,0,0], [0,1,0], [0,0,1]])  # 每行一个向量
    myPlotP(vs, axis=1, c=['r', 'g', 'b'])  # 不同颜色
    
    # 示例3：在已有坐标轴上叠加绘制
    fig, ax = myPlotP([1,1,1], show=False)  # 先不显示
    myPlotP([2,2,2], ax=ax, c='blue')  # 叠加蓝色向量
    
    #下面测试myFindSimilar
    # 构造测试数据（混合文本、数值、空值）
    df1 = pd.DataFrame({
        'id': [1, 2, 3, 4],  # 后续会排除
        'text_col': ['apple', 'banana', '', 'test'],  # 空字符串
        'num_col': ['10.5', '20.3', '30.0', '0'],  # object类型的数值
        'mixed_col': ['x', np.nan, 'z', '']
    }, dtype='object')
    
    df2 = pd.DataFrame({
        'id': [101, 102, 103, 104, 105],
        'text_col': ['apples', 'banana', 'orange', '', 'apple'],
        'num_col': ['10.6', '20.3', '30.0', '40.0', '10.5'],
        'mixed_col': ['x', 'y', 'z', np.nan, 'x']
    }, dtype='object')
    
    # 测试1：默认带相似度的比较模式（Y）
    print("=== 测试1：enable_similarity='Y'（默认带相似度） ===")
    top3_similar = MyFindSimilar.myFindSimilar(df1, df2, exclude_cols=['id'])
    print(top3_similar)
    
    # 测试2：精确相等比较模式（N）
    print("\n=== 测试2：enable_similarity='N'（精确相等） ===")
    top3_exact = MyFindSimilar.myFindSimilar(df1, df2, exclude_cols=['id'], enable_similarity='N')
    print(top3_exact)