import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.fft import fft, fftfreq, fftshift
class MyFT():
    @staticmethod
    def myFFTAnalysis(x,y,T=None):
        """
        对周期信号进行傅里叶分析,使用numpy的FFT
        参数
        ----------
        x : 1D ndarray
            等间距采样时间点，应覆盖一个完整周期 [0, T)（不含右端点）
        y : 1D ndarray
            与 x 对应的信号值
        T : float, optional
            信号周期。若为 None，则根据 x 自动计算（假设采样均匀且区间结束于 T 之前）
        返回
        -------
        result : dict，结构如下：
          {
              'freq': ndarray,          # 正频率分量（包含直流和 Nyquist）
              'amplitude': ndarray,     # 单边幅度谱
              'phase': ndarray          # 相位（弧度，对应余弦形式 cos(2πft + φ)）
          }
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        N = len(x)
        if N != len(y):
            raise ValueError("x 和 y 长度必须相同")
        # 自动确定周期 T（假设采样均匀且覆盖 [0, T)）
        if T is None:
            dx = x[1] - x[0] if N > 1 else 1.0
            T = x[-1] + dx          # 最后一个采样点 + 采样间隔

        if N % 2 == 1:          # 奇数个点，最高谐波 (N-1)/2
            K = (N - 1) // 2
            nyquist = False
        else:                   # 偶数个点，最高谐波 N/2 (Nyquist)
            K = N // 2
            nyquist = True
        # ========== 方法1：NumPy FFT（单边谱）==========
        fft_vals = np.fft.fft(y) / N          # 双边谱（已归一化）
        freqs_fft = np.fft.fftfreq(N, d=T/N)  # 双边频率
        # 取单边正频率（0 到 Nyquist）
        pos_idx = np.arange(0, N//2 + 1)
        freqs_single = freqs_fft[pos_idx]
        fft_single = fft_vals[pos_idx]
        # 幅度：单边谱时，除直流外乘以 2
        amp_single = np.abs(fft_single)
        amp_single[1:-1] *= 2   # 对非直流且非 Nyquist 的频率的幅度加倍
        # 相位（直接来自 FFT 的辐角）
        phase_single = np.angle(fft_single)

        # 从 FFT 单边谱转换为余弦/正弦系数 a, b
        # 关系：A cos(2πft + φ) = A cosφ cos(2πft) - A sinφ sin(2πft)
        # 因此 a = A cosφ, b = -A sinφ
        a_fft = amp_single * np.cos(phase_single)
        b_fft = -amp_single * np.sin(phase_single)
        # 直流分量和 Nyquist 分量（如果存在）的 b 系数应为 0（理论上），但保留计算值
        # 若需要严格为 0，可强制设定：
        # b_fft[0] = 0.0
        # if N % 2 == 0:  # 偶数个点，存在 Nyquist 频率
        #     b_fft[-1] = 0.0
    
        result = {'freq': freqs_single,  #频率
            'amplitude': amp_single,     #幅度
            'phase': phase_single,       #相位
            'cos_coef': a_fft,
            'sin_coef': b_fft,
            'cycle': T,
            'X': x
        }
        return result

    @staticmethod
    def myFFTAlgebra(x, y, T=None):
        """
        对周期信号使用代数的方式进行傅里叶计算
        参数
        ----------
        x : 1D ndarray
            等间距采样时间点，应覆盖一个完整周期 [0, T)（不含右端点）
        y : 1D ndarray
            与 x 对应的信号值
        T : float, optional
            信号周期。若为 None，则根据 x 自动计算（假设采样均匀且区间结束于 T 之前）
    
        返回
        -------
        result : dict。结构如下：
        {
          'freq': ndarray,          # 频率（k/T）
          'amplitude': ndarray,     # 幅度 sqrt(a_k² + b_k²)
          'phase': ndarray,         # 相位 arctan2(-b_k, a_k)
          'cos_coef': ndarray,      # 余弦系数 a_k（k=0,1,...,Nyquist）
          'sin_coef': ndarray,      # 正弦系数 b_k（b_0 = 0, b_Nyquist = 0）
          'cycle':float             # 周期
          'K':int                   #单边参数个数
        }
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        N = len(x)
        if N != len(y):
            raise ValueError("x 和 y 长度必须相同")
    
        # 自动确定周期 T（假设采样均匀且覆盖 [0, T)）
        if T is None:
            dx = x[1] - x[0] if N > 1 else 1.0
            T = x[-1] + dx          # 最后一个采样点 + 采样间隔        
    
        # ========== 方法2：手动傅里叶级数系数 ==========
        # 确定最高谐波次数
        if N % 2 == 1:          # 奇数个点，最高谐波 (N-1)/2
            K = (N - 1) // 2
            nyquist = False
        else:                   # 偶数个点，最高谐波 N/2 (Nyquist)
            K = N // 2
            nyquist = True
    
        a = np.zeros(K + 1)     # cos 系数
        b = np.zeros(K + 1)     # sin 系数
    
        for k in range(K + 1):
            if k == 0:
                a[k] = np.mean(y)          # 直流分量 = 平均值
                b[k] = 0.0
            elif nyquist and k == K:       # Nyquist 频率（只有余弦项）
                a[k] = (1.0 / N) * np.sum(y * np.cos(2 * np.pi * k * x / T))
                b[k] = 0.0
            else:
                a[k] = (2.0 / N) * np.sum(y * np.cos(2 * np.pi * k * x / T))
                b[k] = (2.0 / N) * np.sum(y * np.sin(2 * np.pi * k * x / T))
    
        freqs_manual = np.arange(K + 1) / T
        # 幅度与相位（余弦形式：a cos + b sin = A cos(2πft + φ)）
        amp_manual = np.sqrt(a**2 + b**2)
        phase_manual = np.arctan2(-b, a)   # 注意负号，使得 cos(2πft + φ) 展开为 a cos - b sin
        # ========== 输出结果 ==========
        result = {
            'freq': freqs_manual,
            'amplitude': amp_manual,
            'phase': phase_manual,
            'cos_coef': a,
            'sin_coef': b,
            'cycle':T,
            'X':x
        }
        return result
    
    @staticmethod
    def myFFTMatrix(y,start_time=0, T=None,return_mat=False):
        """
        使用傅里叶矩阵（最小二乘法）对周期信号进行傅里叶级数分解。
        返回频率、幅度、相位，以及余弦系数 a 和正弦系数 b。
    
        参数
        ----------
        y : np.ndarray
            一维信号数组，假设在区间 [0, T) 上等间距采样（不含右端点）。
        T : float, optional
            信号周期。若为 None，则 T = len(y)（采样间隔为 1）。
    
        返回
        -------
        result : dict
            {
                'freq': ndarray,          # 频率数组 [0, 1/T, 2/T, ..., K/T]
                'amplitude': ndarray,     # 幅度谱（单边，对应余弦形式 A*cos(2πft + φ)）
                'phase': ndarray,         # 相位谱（弧度，范围 -π 到 π）
                'cos_coef': ndarray,      # 余弦系数 a_k（k=0,1,...,K）
                'sin_coef': ndarray,      # 正弦系数 b_k（k=0,1,...,K，其中 b_0=0，Nyquist 处 b=0）
                'reconstructed': ndarray, # 使用系数重建的信号
            }
        """
        y = np.asarray(y).flatten()
        N = len(y)
        if T is None:
            T = float(N)          # 默认周期等于采样点数（采样间隔为1）
        # 等间距时间点（不包含右端点，保证周期性）
        x = np.linspace(start_time, start_time + T, N, endpoint=False)
        w = 2 * np.pi / T         # 基频
    
        # 确定傅里叶级数的最高次数 K
        if N % 2 == 1:            # 奇数个点
            K = (N - 1) // 2
            cos_freqs = np.arange(0, K + 1)          # 0,1,...,K
            sin_freqs = np.arange(1, K + 1)          # 1,2,...,K
        else:                     # 偶数个点
            K = N // 2
            cos_freqs = np.arange(0, K + 1)          # 0,1,...,K
            sin_freqs = np.arange(1, K)              # 1,2,...,K-1
    
        # 构造傅里叶矩阵 mat
        mat = np.empty((N, 0))
        for k in cos_freqs:
            mat = np.column_stack((mat, np.cos(k * w * x)))
        for k in sin_freqs:
            mat = np.column_stack((mat, np.sin(k * w * x)))
    
        # 最小二乘法求解系数（mat 是方阵时等价于线性方程组求解）
        # 使用 lstsq 更稳定，但原代码用 inv，这里保留原风格也可，改为 lstsq 更鲁棒
        c, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)   # 系数向量
    
        # 分离余弦系数 a 和正弦系数 b（注意长度可能不同）
        a_raw = c[:len(cos_freqs)]        # 对应 cos_freqs 顺序
        b_raw = c[len(cos_freqs):]        # 对应 sin_freqs 顺序
    
        # 构造完整的 a_k, b_k 数组（k = 0 .. K）
        a_full = np.zeros(K + 1)
        b_full = np.zeros(K + 1)
    
        # 填充余弦系数
        for idx, k in enumerate(cos_freqs):
            a_full[k] = a_raw[idx]
        # 填充正弦系数（sin_freqs 中的 k 从 1 开始）
        for idx, k in enumerate(sin_freqs):
            b_full[k] = b_raw[idx]
    
        # 频率数组
        freqs = np.arange(K + 1) / T
    
        # 幅度和相位（余弦形式：A cos(2πft + φ) = a cos(2πft) + b sin(2πft)）
        # 关系：a = A cos φ, b = -A sin φ  => A = sqrt(a²+b²), φ = atan2(-b, a)
        amplitude = np.sqrt(a_full**2 + b_full**2)
        phase = np.arctan2(-b_full, a_full)   # 范围 [-π, π]
    
        # 重建信号（使用完整级数）
        # y_recon = mat @ c
    
        # 结果字典
        result = {
            'freq': freqs,
            'amplitude': amplitude,
            'phase': phase,
            'cos_coef': a_full,
            'sin_coef': b_full,
            # 'reconstructed': y_recon,
            'cycle': T,
            'X': x
        }
        if return_mat == True:
            result['mat'] = mat
            result['C'] = c
        return result

    @staticmethod
    def myFFTNonperiodic(func, t_range, N, return_signal=False):
        """
        数值计算连续非周期信号 func(t) 的傅里叶变换 F(f) = ∫ func(t) e^{-j2πft} dt
        参数
        ----------
        func : callable 或 ndarray
            若为可调用函数，将在 t 上求值；若为一维数组，则直接作为信号。
        t_range : tuple (t_start, t_end)
            时间采样范围，应足够大以包含信号的主要能量。
        N : int
            采样点数（推荐 2 的幂，以提高 FFT 效率）。
        return_signal : bool
            是否同时返回采样的时域信号和时间轴。
    
        返回
        -------
        result : dict
            {
                'freq': ndarray,          # 频率数组（双边对称）
                'amplitude': ndarray,     # 幅度谱 |F(f)|
                'phase': ndarray,         # 相位谱 arg(F(f))  [-π, π]
                'real': ndarray,          # 实部
                'imag': ndarray,          # 虚部
                'freq_shifted': ndarray,  # fftshift 后的频率（便于绘图）
                'amplitude_shifted': ndarray,  # fftshift 后的幅度
                'phase_shifted': ndarray,      # fftshift 后的相位
            }
            若 return_signal 为 True，则额外返回 't', 'y'。
        """
        t_start, t_end = t_range
        t = np.linspace(t_start, t_end, N, endpoint=False)
        dt = t[1] - t[0]   # 采样间隔
    
        # 生成信号
        if callable(func):
            y = func(t)
        else:
            y = np.asarray(func).flatten()
            if len(y) != N:
                raise ValueError("如果直接传入信号数组，其长度必须等于 N")
    
        # 执行 FFT
        Y = fft(y) * dt          # 乘以 dt 近似连续傅里叶变换积分
        freqs = fftfreq(N, dt)   # 双边频率（负频在前，正频在后）
    
        # 计算幅度、相位、实部、虚部
        amplitude = np.abs(Y)
        phase = np.angle(Y)
        real = np.real(Y)
        imag = np.imag(Y)
    
        # fftshift：将零频移到中心
        freqs_shifted = fftshift(freqs)
        amplitude_shifted = fftshift(amplitude)
        phase_shifted = fftshift(phase)
    
        result = {
            'freq': freqs,
            'amplitude': amplitude,
            'phase': phase,
            'real': real,
            'imag': imag,
            'freq_shifted': freqs_shifted,
            'amplitude_shifted': amplitude_shifted,
            'phase_shifted': phase_shifted,
            'cycle': t_end - t_start, 
            'X': t
        }
        if return_signal:
            result['y'] = y
    
        return result
    
    @staticmethod
    def myPredictFromAB(x,a,b,T=None,K=None):
        """ 使用cos sin的级数进行时间序列还原函数
        x：需要计算的时间序列的时间
        a：cos的级数值
        b：sin的级数值
        T：# 周期
        K：单边参数个数"""
        y_recon = np.zeros_like(x)
        for i, t in enumerate(x):
            for k in range(K + 1):
                y_recon[i] += a[k] * np.cos(2 * np.pi * k * t / T) + b[k] * np.sin(2 * np.pi * k * t / T)
        return y_recon

    @staticmethod
    def myPredictFromSeries(x, freq, amp, phase,x0=0.0):
        """调用时 x需要减去原信号的x0。根据傅里叶分解的频率，幅度，相位，计算时间t对应的值
        x:时间t，求此时间 对应的振幅值，可以是标量，也可以是数组
        freq：傅里叶的频率
        amp：傅里叶的幅度
        phase：傅里叶相度
        x0: 表示训练数据的第0位的数值
        """
        x = x-x0
        if np.isscalar(x):
            return np.cos(2 * np.pi * x * freq + phase) @ amp
        else:
            # x 为数组，形状 (M,)
            return np.cos(2 * np.pi * x[:, None] * freq + phase) @ amp
    
    @staticmethod
    def myPredictFromSeries_ori(x):
        rst = np.cos(2*np.pi* x * r['freq'] + r['phase']) @ r['amplitude']
        return rst

    @staticmethod
    def myPredictFromComplex(ft_result, t_points):
        """
        基于傅里叶变换结果（双边频谱）预测时域信号在任意时间点的值。
        
        参数
        ----------
        ft_result : dict
            由 fourier_transform_nonperiodic 返回的字典，必须包含以下键：
            'freq', 'real', 'imag' (或 'amplitude', 'phase' 之一)。
        t_points : float or array_like
            需要预测的时间点（标量或一维数组）。
        
        返回
        -------
        y_pred : ndarray
            预测得到的时域信号值（形状与 t_points 相同）。
        """
        # 获取频率数组（双边，可能包含负频）
        f = ft_result['freq']
        # 复频谱 F(f)
        if 'real' in ft_result and 'imag' in ft_result:
            F = ft_result['real'] + 1j * ft_result['imag']
        elif 'amplitude' in ft_result and 'phase' in ft_result:
            F = ft_result['amplitude'] * np.exp(1j * ft_result['phase'])
        else:
            raise KeyError("ft_result 必须包含 ('real','imag') 或 ('amplitude','phase') 字段")
        
        # 频率分辨率 Δf
        df = f[1] - f[0]   # 假设频率等间隔
        
        # 将 t_points 转换为数组
        t_vals = np.atleast_1d(t_points)
        # y_vals = np.zeros_like(t_vals, dtype=float)
        y_vals = np.zeros_like(t_vals, dtype=complex)
        t0 = ft_result['t'][0]
        
        # 对每个时间点求和：y(t) ≈ Σ F(f_k) * exp(j 2π f_k t) * Δf
        for i, t in enumerate(t_vals):
            # 注意：由于 F 是双边谱，直接使用全部频率积分即可
            y_vals[i] = np.sum(F * np.exp(1j * 2 * np.pi * f * (t-t0) )) * df
        
        # 理论上结果应为实数（因为原信号是实函数），取实部
        y_vals = np.real(y_vals)
        
        # 如果输入是标量，返回标量
        return y_vals.item() if np.isscalar(t_points) else y_vals
    
    @staticmethod
    def myPredictFromSpectrum(ft_result, t_points):
        """
        利用 IFFT 恢复时域采样点，再通过插值预测任意时间点的值。
        参数
        ----------
        ft_result : dict
            必须包含 'freq', 'real', 'imag' (或 'amplitude', 'phase') 以及 't', 'y'（由 fourier_transform_nonperiodic 且 return_signal=True 返回）。
        t_points : float or array_like
            需要预测的时间点。
        返回
        -------
        y_pred : ndarray
            预测值。
        """
        # 从 ft_result 中提取原始采样时间和信号值
        if 't' not in ft_result or 'y' not in ft_result:
            raise KeyError("ft_result 必须包含 't' 和 'y' 字段（调用 fourier_transform_nonperiodic 时请设置 return_signal=True）")
        t_orig = ft_result['t']
        y_orig = ft_result['y']
        
        # 确保时间点是单调递增的（默认 linspace 生成的就是）
        # 创建三次样条插值器（可换为其他插值方法）
        spline = CubicSpline(t_orig, y_orig, bc_type='natural')
        
        t_vals = np.atleast_1d(t_points)
        y_pred = spline(t_vals)
        
        return y_pred.item() if np.isscalar(t_points) else y_pred
    
    @staticmethod
    def gaussian_pdf(t, mu=0, sigma=1):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    
    @staticmethod
    def gaussian_theoretical_ft(f, mu=0, sigma=1):
        return np.exp(-2 * np.pi**2 * sigma**2 * f**2) * np.exp(-1j * 2 * np.pi * mu * f)
    