## 说明
> 下面说的github方法指的是[@Sherry-XLL:https://github.com/Sherry-XLL/Digital-Recognition-DTW_HMM_GMM](https://github.com/Sherry-XLL/Digital-Recognition-DTW_HMM_GMM)的数据和方法。


### 代码
1. **音频录制.py：** 批量录制语音
2. **voice_record_split.py：** 可以用自己写的双门限法或者github的双门限法提取有效语音段，并将所有有效语音段分为训练集和测试集
3. **feature_extract.py：** 可以用自己写的频域分析特征提取方法或者github的调用librosa计算mfcc的方法来提取语音段特征
4. **voice_recignize.py：** 对2的训练集和数据集，分别使用3中两种特征提取方法，统一用dtw进行预测，比较两种特征提取方法的准确率

### 文件
1. **records：** github上下载的英文录音集，每个数字有20段录音，不同说话人
2. **voice_set：** 自己创建的普通话/粤语（_can）/英语（_en）录音集，每个数字有20段录音，同一说话人
3. **processed_test_records_g_g：** test/train表示测试集（4）/训练集（16）；第一个后缀_g/_c/_y/_e表示github录音/个人中文录音/个人粤语录音/个人英文录音；第二个后缀_g/_c表示特征提取方法上用github的/自己写的
   
> 在提取有效语音段以及特征提取的方法上感觉不用对比，因为自己写的和github上的原理其实是一样的。准确率上虽然github的方法更高，但差距感觉在合理范围内。特征提取上，github直接调用librosa计算mfcc的函数。自己写的包括计算mfcc的所有流程，感觉放在报告中更合适，所以还是想用自己写的特征提取方法。
> 
> 之前以为是中英文的差异导致准确率不同，但是发现github的英文录音集是不同的说话人，因此准确率比单人录制的低很多，这感觉也可以作为一个对比的点（中文/英文，单一说话人/多个说话人）。
