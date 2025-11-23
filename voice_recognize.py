# -*- coding: utf-8 -*-
from math import sqrt
from utils import *
from feature_extract import *
from valid_record_split import *
import os
import numpy as np
import warnings


def calEuclidDist(A, B):
    """
    :param A, B: two vectors
    :return: the Euclidean distance of A and B
    """
    return sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def dtw(M1, M2):
    """
    Compute Dynamic Time Warping(DTW) of two mfcc sequences.
    :param M1, M2: two mfcc sequences
    :return: the minimum distance and the wrap path
    """
    # length of two sequences
    M1_len = len(M1)
    M2_len = len(M2)
    cost_0 = np.zeros((M1_len + 1, M2_len + 1))
    cost_0[0, 1:] = np.inf
    cost_0[1:, 0] = np.inf
    # Initialize the array size to M1_len * M2_len
    cost = cost_0[1:, 1:]
    for i in range(M1_len):
        for j in range(M2_len):
            cost[i, j] = calEuclidDist(M1[i], M2[j])
    # dynamic programming to calculate cost matrix
    for i in range(M1_len):
        for j in range(M2_len):
            cost[i, j] += min([cost_0[i, j], \
                               cost_0[min(i + 1, M1_len - 1), j], \
                               cost_0[i, min(j + 1, M2_len - 1)]])

    # calculate the warp path
    if len(M1) == 1:
        path = np.zeros(len(M2)), range(len(M2))
    elif len(M2) == 1:
        path = range(len(M1)), np.zeros(len(M1))
    else:
        i, j = np.array(cost_0.shape) - 2
        path_1, path_2 = [i], [j]
        # path_1, path_2 with the minimum cost is what we want
        while (i > 0) or (j > 0):
            arg_min = np.argmin((cost_0[i, j], cost_0[i, j + 1], cost_0[i + 1, j]))
            if arg_min == 0:
                i -= 1
                j -= 1
            elif arg_min == 1:
                i -= 1
            else:
                j -= 1
            path_1.insert(0, i)
            path_2.insert(0, j)
        # convert to array
        path = np.array(path_1), np.array(path_2)
    # the minimum distance is the normalized distance
    return cost[-1, -1] / sum(cost.shape), path


def train_model_dtw_unified(train_dir, feature_method='custom', **feature_kwargs):
    """
    使用统一特征提取接口的训练函数
    
    :param train_dir: directory of train audios
    :param feature_method: 'custom' 或 'librosa'，选择特征提取方法
    :param feature_kwargs: 特征提取的其他参数
    :return: the trained mfcc model
    """
    model = []
    for digit in range(10):
        digit_dir = os.path.join(train_dir, "digit_" + str(digit))
        file_list = os.listdir(digit_dir)
        mfcc_list = []
        
        print(f"训练数字 {digit}，处理 {len(file_list)} 个文件")
        
        # 使用统一的特征提取方法
        for j in range(len(file_list)):
            file_path = os.path.join(digit_dir, file_list[j])
            if file_path[-4:] == ".wav":
                try:
                    # 使用统一的特征提取接口
                    wav_mfcc = extract_features_unified(
                        file_path, 
                        method=feature_method,
                        **feature_kwargs
                    )
                    mfcc_list.append(wav_mfcc)
                    print(f"  文件 {file_list[j]}: 特征形状 {wav_mfcc.shape}")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
                    continue

        if not mfcc_list:
            print(f"数字 {digit} 没有有效的训练数据")
            # 添加一个空模型占位
            model.append(None)
            continue

        # 后续的DTW模板生成逻辑保持不变
        mfcc_count = np.zeros(len(mfcc_list[0]))
        mfcc_all = np.zeros(mfcc_list[0].shape)
        
        for i in range(len(mfcc_list)):
            _, path = dtw(mfcc_list[0], mfcc_list[i])
            for j in range(len(path[0])):
                mfcc_count[int(path[0][j])] += 1
                mfcc_all[int(path[0][j])] += mfcc_list[i][path[1][j]]

        # 生成模板
        model_digit = np.zeros(mfcc_all.shape)
        for i in range(len(mfcc_count)):
            for j in range(len(mfcc_all[i])):
                if mfcc_count[i] > 0:  # 避免除零
                    model_digit[i][j] = mfcc_all[i][j] / mfcc_count[i]
        
        model.append(model_digit)
        print(f"数字 {digit} 模型形状: {model_digit.shape}")
    
    return model


def predict_dtw_unified(model, file_path, feature_method='custom', **feature_kwargs):
    """
    使用统一特征提取接口的预测函数
    
    :param model: trained model
    :param file_path: path of .wav file
    :param feature_method: 'custom' 或 'librosa'，选择特征提取方法
    :param feature_kwargs: 特征提取的其他参数
    :return: digit
    """
    try:
        # 使用统一的特征提取接口
        mfcc_feat = extract_features_unified(
            file_path, 
            method=feature_method,
            **feature_kwargs
        )
        
        # 后续的DTW距离计算逻辑保持不变
        digit = 0
        min_dist = float('inf')
        
        # 找到第一个有效的模型
        valid_model_index = -1
        for i in range(len(model)):
            if model[i] is not None:
                valid_model_index = i
                min_dist, _ = dtw(model[i], mfcc_feat)
                digit = i
                break
        
        if valid_model_index == -1:
            print("没有有效的模型可用于预测")
            return "0"
        
        # 比较所有有效模型
        for i in range(valid_model_index + 1, len(model)):
            if model[i] is not None:
                dist, _ = dtw(model[i], mfcc_feat)
                if dist < min_dist:
                    digit = i
                    min_dist = dist
        
        return str(digit)
    except Exception as e:
        print(f"预测文件 {file_path} 时出错: {e}")
        return "0"  # 返回默认值


def evaluate_model(test_dir, model, predict_func, **predict_kwargs):
    """
    评估模型准确率
    
    :param test_dir: 测试集目录
    :param model: 训练好的模型
    :param predict_func: 预测函数
    :param predict_kwargs: 预测函数的其他参数
    :return: 准确率
    """
    count = 0
    pred_true = 0
    
    for i in range(10):
        digit_dir = os.path.join(test_dir, "digit_" + str(i))
        if not os.path.exists(digit_dir):
            continue
            
        file_list = os.listdir(digit_dir)
        for j in range(len(file_list)):
            file_path = os.path.join(digit_dir, file_list[j])
            if file_path[-4:] == ".wav":
                count += 1
                pred = predict_func(model, file_path, **predict_kwargs)
                if file_path[-5] == pred:
                    pred_true += 1
                else:
                    print(f"预测错误: 文件 {file_path} 预测为 {pred}, 实际为 {file_path[-5]}")
    
    accuracy = pred_true / count if count > 0 else 0
    print(f"测试样本数: {count}, 正确识别数: {pred_true}")
    print(f"准确率: {accuracy:.2%}")
    
    return accuracy


if __name__ == '__main__':
    train_dir = "./processed_train_records_e_c"
    test_dir = "./processed_test_records_e_c"
    
    print("=" * 50)
    print(f'语音识别系统 - 多种特征提取方法比较 - 数据集：{train_dir[-3:]}')
    print("=" * 50)
    
    # 方法1: 使用自定义特征提取方法
    print("\n1. 使用自定义特征提取方法")
    print("-" * 30)
    model_custom = train_model_dtw_unified(
        train_dir, 
        feature_method='custom',
        include_delta=True,
        include_delta_delta=True
    )
    accuracy_custom = evaluate_model(
        test_dir, 
        model_custom, 
        predict_dtw_unified,
        feature_method='custom',
        include_delta=True,
        include_delta_delta=True
    )
    
    # 方法2: 使用Librosa特征提取方法
    print("\n2. 使用Librosa特征提取方法")
    print("-" * 30)
    model_librosa = train_model_dtw_unified(
        train_dir, 
        feature_method='librosa',
        delta=2
    )
    accuracy_librosa = evaluate_model(
        test_dir, 
        model_librosa, 
        predict_dtw_unified,
        feature_method='librosa',
        delta=2
    )

    
    # 结果汇总
    print("\n" + "=" * 50)
    print(f'结果汇总 - 数据集：{train_dir[-3:]}')
    print("=" * 50)
    print(f"自定义特征提取方法准确率: {accuracy_custom:.2%}")
    print(f"Librosa特征提取方法准确率: {accuracy_librosa:.2%}")
    
    # 找出最佳方法
    methods = [
        ("自定义方法", accuracy_custom),
        ("Librosa方法", accuracy_librosa),
    ]
    best_method = max(methods, key=lambda x: x[1])
    print(f"\n最佳方法: {best_method[0]}, 准确率: {best_method[1]:.2%}")