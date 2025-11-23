import numpy as np
import scipy.io.wavfile
from scipy import signal
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

'''自己写的特征提取 - 修改为与librosa输出格式一致'''
def extract_features_from_valid_segment(valid_signal, sample_rate, include_delta=True, include_delta_delta=True):
    """
    在有效语音区间内提取特征
    修改为与librosa的MFCC输出格式保持一致
    
    参数:
    valid_signal: 有效区间的语音信号
    sample_rate: 采样率
    include_delta: 是否包含一阶差分特征
    include_delta_delta: 是否包含二阶差分特征
    
    返回:
    mfcc_features: MFCC特征矩阵，形状为(帧数, 特征维度)
    """
    
    # 参数设置 - 调整为与librosa更接近的参数
    frame_length = 512  # 32ms for 16kHz
    frame_shift = 256   # 16ms for 16kHz, 50% overlap
    n_fft = 512
    nfilt = 40
    num_ceps = 13
    
    # 1. 预加重
    pre_emphasis = 0.97
    emphasized_signal = np.append(valid_signal[0], valid_signal[1:] - pre_emphasis * valid_signal[:-1])
    
    # 2. 分帧
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_shift)) + 1
    pad_length = (num_frames - 1) * frame_shift + frame_length
    
    if pad_length > signal_length:
        padding = np.zeros(pad_length - signal_length)
        padded_signal = np.concatenate((emphasized_signal, padding))
    else:
        padded_signal = emphasized_signal
    
    frames = []
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = padded_signal[start:end]
        frames.append(frame)
    frames = np.array(frames)
    
    # 3. 加窗
    hamming_window = np.hamming(frame_length)
    windowed_frames = frames * hamming_window
    
    # 4. 计算频谱特征
    magnitude_frames = np.zeros((num_frames, n_fft//2 + 1))
    energy_frames = np.zeros((num_frames, n_fft//2 + 1))
    
    for i in range(num_frames):
        windowed_frame = windowed_frames[i, :]
        wf_fft = np.abs(np.fft.fft(windowed_frame, n_fft))
        magnitude_frames[i, :] = wf_fft[:n_fft//2 + 1]
        energy_frames[i, :] = magnitude_frames[i, :] ** 2
    
    # 5. 计算梅尔滤波器组和MFCC
    def get_filter_banks(nfilt=40, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        if highfreq is None:
            highfreq = samplerate / 2
        lowmel = 2595.0 * np.log10(1 + lowfreq / 700.0)
        highmel = 2595.0 * np.log10(1 + highfreq / 700.0)
        melpoints = np.linspace(lowmel, highmel, nfilt + 2)
        hzpoints = 700 * (10**(melpoints / 2595.0) - 1.0)
        bin = np.floor((nfft + 1) * hzpoints / samplerate)
        fbank = np.zeros([nfilt, nfft//2 + 1])
        for j in range(nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j, i] = (i - bin[j]) / (bin[j+1] - bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j, i] = (bin[j+2] - i) / (bin[j+2] - bin[j+1])
        return fbank, hzpoints
    
    filter_banks, hz_points = get_filter_banks(nfilt=nfilt, nfft=n_fft, samplerate=sample_rate)
    filter_bank_energies = np.dot(energy_frames, filter_banks.T)
    filter_bank_energies = np.where(filter_bank_energies == 0, np.finfo(float).eps, filter_bank_energies)
    log_mel_spectrogram = 10 * np.log10(filter_bank_energies)
    
    # 6. 计算MFCC
    mfcc_features = dct(log_mel_spectrogram, type=2, axis=1, norm='ortho')[:, :num_ceps]
    
    # 7. 计算一阶差分（Delta特征）
    def compute_delta(features, N=2):
        """计算特征的一阶差分"""
        delta = np.zeros_like(features)
        for t in range(len(features)):
            if t < N:
                delta[t] = features[t+1] - features[t]
            elif t >= len(features) - N:
                delta[t] = features[t] - features[t-1]
            else:
                numerator = sum(i * (features[t+i] - features[t-i]) for i in range(1, N+1))
                denominator = 2 * sum(i**2 for i in range(1, N+1))
                delta[t] = numerator / denominator
        return delta
    
    # 8. 组合特征 - 与librosa格式保持一致
    if include_delta and include_delta_delta:
        # 与librosa相同：MFCC + Delta + Delta-Delta
        mfcc_delta = np.array([compute_delta(mfcc_features[:, i]) for i in range(mfcc_features.shape[1])]).T
        mfcc_delta_delta = np.array([compute_delta(mfcc_delta[:, i]) for i in range(mfcc_delta.shape[1])]).T
        combined_features = np.hstack([mfcc_features, mfcc_delta, mfcc_delta_delta])
    elif include_delta:
        # 只包含MFCC + Delta
        mfcc_delta = np.array([compute_delta(mfcc_features[:, i]) for i in range(mfcc_features.shape[1])]).T
        combined_features = np.hstack([mfcc_features, mfcc_delta])
    else:
        # 只包含MFCC
        combined_features = mfcc_features
    
    return combined_features


'''github的特征提取 - 保持原样'''
def mfcc(wav_path, delta=2):
    """
    Read .wav files and calculate MFCC
    返回格式: (帧数, 特征维度)
    """
    y, sr = librosa.load(wav_path)
    # MEL frequency cepstrum coefficient
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    ans = [mfcc_feat]
    # Calculate the 1st derivative
    if delta >= 1:
        mfcc_delta1 = librosa.feature.delta(mfcc_feat, order=1, mode='nearest')
        ans.append(mfcc_delta1)
    # Calculate the 2nd derivative
    if delta >= 2:
        mfcc_delta2 = librosa.feature.delta(mfcc_feat, order=2, mode='nearest')
        ans.append(mfcc_delta2)
    
    # 转置为 (帧数, 特征维度) 格式
    return np.transpose(np.concatenate(ans, axis=0), [1, 0])


'''统一接口函数'''
def extract_features_unified(input_data, sample_rate=None, method='custom', **kwargs):
    """
    统一的特征提取接口
    
    参数:
    input_data: 可以是文件路径或语音信号数组
    sample_rate: 采样率（当input_data为数组时需要）
    method: 'custom' 或 'librosa'
    **kwargs: 其他参数
    
    返回:
    features: 形状为 (帧数, 特征维度) 的numpy数组
    """
    
    if method == 'librosa':
        # 使用librosa方法
        if isinstance(input_data, str):
            # 输入是文件路径
            return mfcc(input_data, **kwargs)
        else:
            # 输入是信号，需要先保存为临时文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                scipy.io.wavfile.write(tmp_file.name, sample_rate, input_data.astype(np.int16))
                features = mfcc(tmp_file.name, **kwargs)
                os.unlink(tmp_file.name)
            return features
    
    elif method == 'custom':
        # 使用自定义方法
        if isinstance(input_data, str):
            # 输入是文件路径，需要先读取
            sample_rate, signal_data = scipy.io.wavfile.read(input_data)
            return extract_features_from_valid_segment(signal_data, sample_rate, **kwargs)
        else:
            # 输入是信号
            if sample_rate is None:
                raise ValueError("使用自定义方法时，当输入为信号数组时必须提供sample_rate")
            return extract_features_from_valid_segment(input_data, sample_rate, **kwargs)
    
    else:
        raise ValueError("method参数必须是 'custom' 或 'librosa'")


# 测试函数
def test_feature_extraction():
    """测试两种特征提取方法的输出格式"""
    
    # 示例文件路径 - 请替换为实际文件路径
    test_file = "C:\\Users\\24821\\Desktop\\dsp实验\\voice_set\\0\\recording_01_20251121_220811.wav"
    
    try:
        # 使用librosa方法
        print("=== Librosa方法 ===")
        features_librosa = extract_features_unified(test_file, method='librosa', delta=2)
        print(f"Librosa特征形状: {features_librosa.shape}")
        print(f"Librosa特征范围: [{features_librosa.min():.3f}, {features_librosa.max():.3f}]")
        
        # 使用自定义方法
        print("\n=== 自定义方法 ===")
        features_custom = extract_features_unified(test_file, method='custom', 
                                                  include_delta=True, include_delta_delta=True)
        print(f"自定义特征形状: {features_custom.shape}")
        print(f"自定义特征范围: [{features_custom.min():.3f}, {features_custom.max():.3f}]")
        
        # 比较特征维度
        if features_librosa.shape[1] == features_custom.shape[1]:
            print(f"\n✓ 两种方法输出维度一致: {features_librosa.shape[1]} 维")
        else:
            print(f"\n✗ 维度不一致: Librosa={features_librosa.shape[1]}, 自定义={features_custom.shape[1]}")
            
        # 比较帧数（可能会有差异，因为参数不同）
        print(f"帧数: Librosa={features_librosa.shape[0]}, 自定义={features_custom.shape[0]}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        # 使用模拟数据进行测试
        print("\n使用模拟数据进行测试...")
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_signal = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz正弦波
        
        # 自定义方法
        features_custom = extract_features_unified(test_signal, sample_rate, method='custom',
                                                  include_delta=True, include_delta_delta=True)
        print(f"自定义特征形状: {features_custom.shape}")
        
        # Librosa方法
        features_librosa = extract_features_unified(test_signal, sample_rate, method='librosa', delta=2)
        print(f"Librosa特征形状: {features_librosa.shape}")


# if __name__ == '__main__':
#     test_feature_extraction()