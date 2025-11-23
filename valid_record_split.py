import os
import wave
import numpy as np
import scipy.io.wavfile
from scipy import signal
from scipy.fftpack import dct
import matplotlib.pyplot as plt

'''不论是自己写的还是github的，都统一为输入语音地址，输出有效语音段的格式'''

'''统一的有效语音段提取接口'''
def extract_valid_segment(wav_file, method='custom'):
    """
    统一的有效语音段提取接口
    
    参数:
    wav_file: 音频文件路径
    method: 'custom' 或 'github'，选择使用哪种方法
    
    返回:
    valid_segments: 有效语音段列表，每个元素为 (start_index, end_index)
    sample_rate: 采样率
    signal_wave: 原始信号（用于后续保存）
    """
    
    if method == 'custom':
        return custom_endpoint_detection(wav_file)
    elif method == 'github':
        return github_endpoint_detection(wav_file)
    else:
        raise ValueError("method参数必须是 'custom' 或 'github'")

'''自己写的有效语音段提取函数'''
def custom_endpoint_detection(wav_file):
    """
    改进的双门限法端点检测，严格按照三级判决步骤
    返回格式: (valid_segments, sample_rate, signal_wave)
    """
    
    # 读取音频，设置参数
    sample_rate, signal_wave = scipy.io.wavfile.read(wav_file)
    frame_length = int(0.025 * sample_rate)  # 25ms
    frame_shift = int(0.01 * sample_rate)    # 10ms

    # 1. 分帧
    signal_length = len(signal_wave)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_shift)) + 1
    pad_length = (num_frames - 1) * frame_shift + frame_length
    
    if pad_length > signal_length:
        padding = np.zeros(pad_length - signal_length)
        padded_signal = np.concatenate((signal_wave, padding))
    else:
        padded_signal = signal_wave
    
    frames = []
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = padded_signal[start:end]
        frames.append(frame)
    frames = np.array(frames)
    
    # 2. 加窗
    hamming_window = np.hamming(frame_length)
    windowed_frames = frames * hamming_window
    
    # 3. 计算短时能量
    frame_energies = np.sum(windowed_frames ** 2, axis=1)
    
    # 4. 计算短时过零率
    def zero_crossing_rate(frame):
        """计算一帧信号的过零率"""
        signs = np.sign(frame)
        crossings = np.abs(np.diff(signs)) / 2
        return np.sum(crossings)
    
    zcr = np.array([zero_crossing_rate(frame) for frame in windowed_frames])
    
    # 5. 计算统计特征用于设置门限
    energy_mean = np.mean(frame_energies)
    energy_std = np.std(frame_energies)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # 设置门限
    T2 = energy_mean + 0.8 * energy_std   # 高能量门限
    T1 = energy_mean + 0.2 * energy_std   # 低能量门限
    T3 = zcr_mean + 0.8 * zcr_std         # 过零率门限
    
    print(f"门限设置: T1(低能量)={T1:.2f}, T2(高能量)={T2:.2f}, T3(过零率)={T3:.2f}")
    
    # 第一级判决：根据高能量门限T2确定核心语音段
    high_energy_segments = []
    in_segment = False
    segment_start = -1
    
    for i in range(len(frame_energies)):
        if not in_segment and frame_energies[i] > T2:
            # 开始新的高能量段
            in_segment = True
            segment_start = i
        elif in_segment and frame_energies[i] <= T2:
            # 结束当前高能量段
            in_segment = False
            if segment_start != -1:
                high_energy_segments.append((segment_start, i-1))
                segment_start = -1
    
    # 处理最后一个段
    if in_segment and segment_start != -1:
        high_energy_segments.append((segment_start, len(frame_energies)-1))
    
    print(f"第一级判决: 找到 {len(high_energy_segments)} 个高能量核心段")
    
    # 第二级判决：用低能量门限T1扩展语音段边界
    expanded_segments = []
    
    for seg_start, seg_end in high_energy_segments:
        # 向左扩展 (B点)
        left_bound = seg_start
        while left_bound > 0 and frame_energies[left_bound - 1] > T1:
            left_bound -= 1
        
        # 向右扩展 (E点)
        right_bound = seg_end
        while right_bound < len(frame_energies) - 1 and frame_energies[right_bound + 1] > T1:
            right_bound += 1
        
        expanded_segments.append((left_bound, right_bound))
    
    print(f"第二级判决: 扩展后的语音段 {expanded_segments}")
    
    # 第三级判决：用过零率门限T3精确定位起止点
    final_segments = []
    min_silence_length = 5  # 最小静音长度（帧）
    
    for seg_start, seg_end in expanded_segments:
        # 向左搜索过零率低于T3的点 (A点)
        left_zcr_bound = seg_start
        silence_count = 0
        
        while left_zcr_bound > 0:
            if zcr[left_zcr_bound - 1] < T3:
                silence_count += 1
                if silence_count >= min_silence_length:
                    # 找到连续的静音段，确定左边界
                    left_zcr_bound = left_zcr_bound - min_silence_length
                    break
            else:
                silence_count = 0
            left_zcr_bound -= 1
        
        # 确保左边界不小于0
        left_zcr_bound = max(0, left_zcr_bound)
        
        # 向右搜索过零率低于T3的点 (F点)
        right_zcr_bound = seg_end
        silence_count = 0
        
        while right_zcr_bound < len(zcr) - 1:
            if zcr[right_zcr_bound + 1] < T3:
                silence_count += 1
                if silence_count >= min_silence_length:
                    # 找到连续的静音段，确定右边界
                    right_zcr_bound = right_zcr_bound + min_silence_length
                    break
            else:
                silence_count = 0
            right_zcr_bound += 1
        
        # 确保右边界不超过最大值
        right_zcr_bound = min(len(zcr) - 1, right_zcr_bound)
        
        final_segments.append((left_zcr_bound, right_zcr_bound))
    
    print(f"第三级判决: 最终语音段 {final_segments}")
    
    # 合并相邻的语音段
    merged_segments = []
    if final_segments:
        merged_segments.append(final_segments[0])
        for i in range(1, len(final_segments)):
            last_segment = merged_segments[-1]
            current_segment = final_segments[i]
            
            # 如果两个段之间的距离小于阈值，则合并
            if current_segment[0] - last_segment[1] < 10:  # 10帧的间隔阈值
                merged_segments[-1] = (last_segment[0], current_segment[1])
            else:
                merged_segments.append(current_segment)
    
    # 转换为原始信号索引
    valid_segments = []
    for seg_start, seg_end in merged_segments:
        start_index = seg_start * frame_shift
        end_index = min(seg_end * frame_shift + frame_length, len(signal_wave))
        valid_segments.append((start_index, end_index))
        
        print(f"有效语音段: 起始索引 {start_index}, 结束索引 {end_index}")
        print(f"有效语音长度: {end_index - start_index} 采样点 ({(end_index - start_index)/sample_rate:.2f} 秒)")
    
    return valid_segments, sample_rate, signal_wave


'''github上的有效语音段提取 - 修改为统一格式'''
def github_endpoint_detection(wave_file):
    """
    GitHub版本端点检测，修改为统一输出格式
    返回格式: (valid_segments, sample_rate, signal_wave)
    """
    # 音频读取
    f = wave.open(wave_file, 'rb')
    # get the channels, sample_width, frame_rate and frames num of wav file
    channels, sample_width, frame_rate, frames = f.getparams()[:4]

    # convert data to binary array
    wave_data = np.frombuffer(f.readframes(frames), dtype=np.short)
    f.close()

    # 计算能量
    energy = []
    sum_val = 0
    for i in range(len(wave_data)):
        sum_val = sum_val + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0:  # 分帧，帧长256，间隔256
            energy.append(sum_val)
            sum_val = 0
        elif i == len(wave_data) - 1:
            energy.append(sum_val)

    # 计算过零率
    zeroCrossingRate = []
    sum_val = 0
    for i in range(len(wave_data)):
        sum_val = sum_val + np.abs(int(wave_data[i] >= 0) - int(wave_data[i - 1] >= 0))
        if (i + 1) % 256 == 0:
            zeroCrossingRate.append(float(sum_val) / 255)
            sum_val = 0
        elif i == len(wave_data) - 1:
            zeroCrossingRate.append(float(sum_val) / 255)

    # 计算平均能量
    sum_val = 0
    for en in energy:
        sum_val = sum_val + en
    avg_energy = sum_val / len(energy)

    # 计算能量阈值
    sum_val = 0
    for en in energy[:5]:
        sum_val = sum_val + en
    ML = sum_val / 5
    MH = avg_energy / 4  # high energy threshold
    ML = (ML + MH) / 4   # low energy threshold

    # 计算过零率阈值
    sum_val = 0
    for zcr in zeroCrossingRate[:5]:
        sum_val = float(sum_val) + zcr
    Zs = sum_val / 5  # zero crossing rate threshold

    A = []
    B = []
    C = []

    # MH is used for preliminary detection
    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and (len(A) == 0 or i - 21 > A[len(A) - 1]):
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and len(A) > 0 and i - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[i] < MH:
            # if frame is too short, remove it
            if len(A) > 0 and i - A[len(A) - 1] <= 2:
                A = A[:len(A) - 1]
            else:
                A.append(i)
            flag = 0

    # ML is used for second detection
    for j in range(len(A)):
        i = A[j]
        if j % 2 == 1:
            while i < len(energy) and energy[i] > ML:
                i = i + 1
            B.append(i)
        else:
            while i >= 0 and energy[i] > ML:
                i = i - 1
            B.append(max(0, i))

    # zero crossing rate threshold is for the last step
    for j in range(len(B)):
        i = B[j]
        if j % 2 == 1:
            while i < len(zeroCrossingRate) and zeroCrossingRate[i] >= 3 * Zs:
                i = i + 1
            C.append(i)
        else:
            while i >= 0 and zeroCrossingRate[i] >= 3 * Zs:
                i = i - 1
            C.append(max(0, i))
    
    # 转换为统一格式: [(start_index, end_index), ...]
    valid_segments = []
    m = 0
    while m < len(C):
        if m + 1 < len(C):
            start_index = C[m] * 256
            end_index = C[m + 1] * 256
            valid_segments.append((start_index, end_index))
            print(f"GitHub方法 - 有效语音段: 起始索引 {start_index}, 结束索引 {end_index}")
            print(f"GitHub方法 - 有效语音长度: {end_index - start_index} 采样点 ({(end_index - start_index)/frame_rate:.2f} 秒)")
        m = m + 2
    
    return valid_segments, frame_rate, wave_data


if __name__ == '__main__':
    # 创建目录保存处理后的记录，按4:1划分训练集和测试集
    if not os.path.exists("./processed_test_records/"):
        os.makedirs("./processed_test_records/")
    if not os.path.exists("./processed_train_records/"):
        os.makedirs("./processed_train_records/")
    for i in range(10):
        if not os.path.exists("./processed_test_records/" + 'digit_' + str(i)):
            os.makedirs("./processed_test_records/" + 'digit_' + str(i))
        if not os.path.exists("./processed_train_records/" + 'digit_' + str(i)):
            os.makedirs("./processed_train_records/" + 'digit_' + str(i))
    
    # records_path = "./records/" # 对应于github的录音
    records_path = "./voice_set_en/" # 对应于自己的录音
    
    # 选择使用哪种端点检测方法
    method = 'custom'  # 使用自定义方法，更快
    # method = 'github'    # 使用GitHub方法
    
    for i in range(10):
        # 如果使用自己的录音，因为录音没命名好，需要下面这几行
        digit_dir = os.path.join(records_path, str(i))
        # 检查目录是否存在
        if not os.path.exists(digit_dir):
            print(f"警告: 目录 {digit_dir} 不存在，跳过")
            continue
        files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
        print(f"处理数字 {i}, 找到 {len(files)} 个文件")   

        for j in range(20):
            # file_path = records_path + 'digit_' + str(i) + "/" + str(j + 1) + '_' + str(i) + ".wav" # 对应于github的录音
            file_path = os.path.join(digit_dir, files[j]) # 对应于自己的录音
            # 端点检测
            try:
                valid_segments, sample_rate, wave_data = extract_valid_segment(file_path, method=method)
                
                # 保存每个有效语音段
                for seg_idx, (start_idx, end_idx) in enumerate(valid_segments):
                    # 确定保存路径
                    if j >= 16:  # 测试集
                        save_path = "./processed_test_records/" + 'digit_' + str(i) + "/" + str(j + 1) + '_' + str(i)
                    else:  # 训练集
                        save_path = "./processed_train_records/" + 'digit_' + str(i) + "/" + str(j + 1) + '_' + str(i)
                    
                    # 如果有多段，添加段索引
                    if len(valid_segments) > 1:
                        save_path += f"_seg{seg_idx}"
                    
                    save_path += ".wav"
                    
                    # 保存有效语音段
                    wf = wave.open(save_path, 'wb')
                    wf.setnchannels(1)  # 单声道
                    wf.setsampwidth(2)  # 16位
                    wf.setframerate(sample_rate)
                    
                    # 提取有效段并保存
                    segment_data = wave_data[start_idx:end_idx]
                    wf.writeframes(segment_data.tobytes())
                    wf.close()
                    
                    print(f"保存文件: {save_path}, 长度: {len(segment_data)} 采样点")
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")