


# list = [[1,1,1],[2,2,2]]
# Matrix = np.array(list)
# print(Matrix)
# print(Matrix.T)
# print(Matrix.shape[0])
# print(Matrix.shape[1])
# print(np.reshape(Matrix.T, Matrix.shape[0] * Matrix.shape[1]))
"""

[[1 1 1]
 [2 2 2]]
[[1 2]
 [1 2]
 [1 2]]
2
3
[1 2 1 2 1 2]
"""

# v = np.zeros(3)
# v1 = np.ones(2)
# v2 = np.concatenate((v, v1))
# print(v2)
# print(v2.shape)
# p = np.outer(v1,v1)
# print(p)
# Phi = np.identity(n = 3)
# a = Phi.T
# c = []
# a = np.array([1,1,1,1])
# b = np.array([2,2,2,2])
# c.append(a)
#
# c.append(b)
# print(c)
# d = np.array(c).T
#
# print(d)
# print(d.shape)
# a = np.identity(3)
# print(a.shape[0])
#
# list1 = [1, 2, 3]
# list2 = ['a', 'b', 'c']
# list3 = [4.5, 5.6, 6.7]
# zipped = zip(list1, list2, list3)
# print(list(zipped))  # 输出 [(1, 'a', 4.5), (2, 'b', 5.6), (3, 'c', 6.7)]
# import result
# print(result.str)
# result.str = "2"
# print(result.str)



import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import RUN
import hypernet
import  config
"""
热力图

"""
# #读入user feature
# UserFeatureVectors, ItemFeatureVectors = RUN.read_in_observed_feature("foursquare")
# #加载模型
# with open('model_trained12.0.pkl', 'rb') as f:
#     mlp = pickle.load(f)
# #onehot
# categories_week = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
# categories_day = np.array(['rest', 'morning', 'noon', 'afternoon', 'night', 'rest'])
# user_feature = UserFeatureVectors[32]
# sns.set_theme()
# device = torch.device("cuda")
# for i in range(7):
#     for j in range(5):
#         time_week = categories_week[i]
#         time_day = categories_day[j]
#         one_hot_week, one_hot_day = hypernet.onehot_encode(time_week, time_day)
#         time_line = np.concatenate([one_hot_week, one_hot_day], axis=0)
#         input = np.concatenate([time_line, user_feature[:config.FactorUCB_setting.user_observed_dimension]], axis=0)
#         input = torch.Tensor(input).to(device)
#         output = mlp(input)
#         Theta_Matrix = output.reshape(config.FactorUCB_setting.item_dimension, config.FactorUCB_setting.user_observed_dimension)
#         Theta_hypernet = Theta_Matrix.detach().cpu().numpy()

#         df = pd.DataFrame(Theta_hypernet, columns=range(config.FactorUCB_setting.user_observed_dimension))
#         long_df = df.reset_index().melt(id_vars=['index'], var_name='column', value_name='Value')
#         print(long_df)
#         flights = long_df.pivot("index", "column", "Value")
# # # Load the example flights dataset and convert to long-form
# # flights_long = sns.load_dataset("flights")
# # print(flights_long)
# # flights = flights_long.pivot("month", "year", "passengers")
# # print(flights)


# # Draw a heatmap with the numeric values in each cell
#         f, ax = plt.subplots(figsize=(9, 6))
#         sns.heatmap(flights, annot=True, fmt="f", linewidths=.5, ax=ax, vmin=-1.5, vmax=1.5)
#         plt.savefig('./image/{}_{}.png'.format(time_week, time_day))
#         plt.show()

# a = np.array([[1,2,3],
#          [4,5,6]])# 2 3
# batch_a = np.tile(a, (3, 1, 1))
# sqrt_a = np.sqrt(batch_a)


# label = np.zeros(25)
# label[0] = 1
# label_shuffle_lists =  np.tile(label, (10000, 1, 1))

# print(label_shuffle_lists)

# print(batch_a)

# b = np.array([[1,1],
#              [2,2],
#              [3,3]])# 3,2
# print(b)

# c = np.matmul(a, b)
# print("c:", c)
# batch_c = np.matmul(b, batch_a)

# print("batch_c:", batch_c)

# import random

# # 定义两个list
# list1 = [1, 2, 3, 4]
# list2 = ['a', 'b', 'c', 'd']

# # 使用zip()函数将两个list打包成一个元组列表
# zipped = list(zip(list1, list2))

# # 使用shuffle()函数对元组列表进行随机排序
# random.shuffle(zipped)
# a = list1[0]
# # 使用*运算符将元组列表解压成两个list
# shuffled_list1, shuffled_list2 = zip(*zipped)

# # 打印结果
# print(shuffled_list1)
# print(shuffled_list2)
# print(a)
# import torch
# a=torch.tensor([[[1.],[2.],[3.]],[[1.],[2.],[3.]]])
# b = torch.softmax(a,dim=0)
# print(b)
# c = torch.softmax(a,dim=1)
# print(c)
# import matplotlib.pyplot as plt
plt.plot(range(10), range(10))
plt.xlabel('batch()')
plt.ylabel('normalized cumulative gain')
# plt.legend()
plt.show()
# list = [1, 2, 3, 5, 4]
# position = list.index(5)
# print(position)
# list[position] = 0

# with open("resultdata.pkl", "rb") as f:
#         algresult = pickle.load(f)

# for alg_name, alg in algresult.algorithms.items():
#     plt.plot(range(len(algresult.AlgRewardRatio_vsRandom[alg_name])), algresult.AlgRewardRatio_vsRandom[alg_name], label=alg_name)
# plt.xlabel('batch({})'.format(str(10000)))
# plt.ylabel('normalized cumulative gain')
# plt.legend()
# plt.savefig("test1.png")
# plt.show()

# import torch

# logits = np.ndarray([1,2,3,4,5,6,7]) #0.6721, -0.0308,  0.3088, -0.2480, -1.8096,  0.5981, -0.6040,  1.7112, 0.6591, -0.5506

# # torch.randn(1, 10)
# print(logits)
# sample = gumbel_softmax(logits, tau=0.1, hard=False)

# print(sample)
# import numpy as np

# arr = np.array([0.6721, -0.0308,  0.3088, -0.2480, -1.8096,  0.5981, -0.6040,  1.7112, 0.6591, -0.5506])
# arr2 = np.array([0.6721, -0.0308,  0.3088, -0.2480, -1.8096,  0.5981, -0.6040,  1.7112, 0.6591, -0.5506])
# print(np.dot(arr,arr2))

# print(arr)

# idx = np.argsort(arr)
# print(idx)
"""
D:\SoftwareFamily\Anaconda\envs\nlplab\python.exe D:\IIR\Server\PycharmProject\HyperBandit\RUN.py 
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
Batch: 1.0 LinUCB Normalized Accumulated Payoff: 2.6875
Batch: 1.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.03125
----------------------------
Batch: 2.0 LinUCB Normalized Accumulated Payoff: 2.903225806451613
Batch: 2.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.0483870967741935
----------------------------
Batch: 3.0 LinUCB Normalized Accumulated Payoff: 2.339622641509434
Batch: 3.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.0471698113207548
----------------------------
Batch: 4.0 LinUCB Normalized Accumulated Payoff: 2.0516129032258066
Batch: 4.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.0903225806451613
----------------------------
Batch: 5.0 LinUCB Normalized Accumulated Payoff: 1.9417989417989419
Batch: 5.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.2804232804232805
----------------------------
Batch: 6.0 LinUCB Normalized Accumulated Payoff: 1.9523809523809523
Batch: 6.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.4502164502164503
----------------------------
Batch: 7.0 LinUCB Normalized Accumulated Payoff: 1.9586466165413534
Batch: 7.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.5714285714285714
----------------------------
Batch: 8.0 LinUCB Normalized Accumulated Payoff: 1.9163879598662208
Batch: 8.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.859531772575251
----------------------------
Batch: 9.0 LinUCB Normalized Accumulated Payoff: 1.8075801749271136
Batch: 9.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.064139941690962
----------------------------
Batch: 10.0 LinUCB Normalized Accumulated Payoff: 1.753968253968254
Batch: 10.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.4021164021164023
----------------------------
Batch: 11.0 LinUCB Normalized Accumulated Payoff: 1.6761229314420805
Batch: 11.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.548463356973995
----------------------------
Batch: 12.0 LinUCB Normalized Accumulated Payoff: 1.6846652267818574
Batch: 12.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.7710583153347734
----------------------------
Batch: 14.0 LinUCB Normalized Accumulated Payoff: 1.6925925925925926
Batch: 14.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.274074074074074
----------------------------
Batch: 15.0 LinUCB Normalized Accumulated Payoff: 1.7382198952879582
Batch: 15.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.5479930191972078
----------------------------
Batch: 16.0 LinUCB Normalized Accumulated Payoff: 1.7863105175292153
Batch: 16.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.7228714524207014
----------------------------
Batch: 17.0 LinUCB Normalized Accumulated Payoff: 1.7635239567233385
Batch: 17.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.848531684698609
----------------------------
Batch: 18.0 LinUCB Normalized Accumulated Payoff: 1.7580174927113703
Batch: 18.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.989795918367347
----------------------------
Batch: 19.0 LinUCB Normalized Accumulated Payoff: 1.7363387978142077
Batch: 19.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.133879781420765
----------------------------
Batch: 20.0 LinUCB Normalized Accumulated Payoff: 1.7238219895287958
Batch: 20.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.239528795811518
----------------------------
Batch: 21.0 LinUCB Normalized Accumulated Payoff: 1.7202007528230865
Batch: 21.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.347553324968632
----------------------------
Batch: 22.0 LinUCB Normalized Accumulated Payoff: 1.7019230769230769
Batch: 22.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.413461538461538
----------------------------
Batch: 23.0 LinUCB Normalized Accumulated Payoff: 1.6985040276179517
Batch: 23.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.493670886075949
----------------------------
Batch: 24.0 LinUCB Normalized Accumulated Payoff: 1.7388392857142858
Batch: 24.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.611607142857143
----------------------------
Batch: 25.0 LinUCB Normalized Accumulated Payoff: 1.761290322580645
Batch: 25.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.744086021505376
----------------------------
Batch: 26.0 LinUCB Normalized Accumulated Payoff: 1.7736625514403292
Batch: 26.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.811728395061729
----------------------------
Batch: 27.0 LinUCB Normalized Accumulated Payoff: 1.766798418972332
Batch: 27.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.9061264822134385
----------------------------
Batch: 28.0 LinUCB Normalized Accumulated Payoff: 1.7838095238095237
Batch: 28.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.958095238095238
----------------------------
Batch: 29.0 LinUCB Normalized Accumulated Payoff: 1.7740164684354987
Batch: 29.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.998170173833485
----------------------------
Batch: 30.0 LinUCB Normalized Accumulated Payoff: 1.7804444444444445
Batch: 30.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.109333333333334
----------------------------
Batch: 31.0 LinUCB Normalized Accumulated Payoff: 1.783132530120482
Batch: 31.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.253012048192771
----------------------------
Batch: 32.0 LinUCB Normalized Accumulated Payoff: 1.7663628831814415
Batch: 32.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.256006628003314
----------------------------
Batch: 33.0 LinUCB Normalized Accumulated Payoff: 1.7483922829581993
Batch: 33.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.296623794212219
----------------------------
Batch: 34.0 LinUCB Normalized Accumulated Payoff: 1.749607535321821
Batch: 34.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.356357927786499
----------------------------
Batch: 35.0 LinUCB Normalized Accumulated Payoff: 1.7196048632218845
Batch: 35.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.350303951367781
----------------------------
Batch: 36.0 LinUCB Normalized Accumulated Payoff: 1.7253731343283583
Batch: 36.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.423880597014925
----------------------------
Batch: 37.0 LinUCB Normalized Accumulated Payoff: 1.7181159420289855
Batch: 37.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.452173913043478
----------------------------
Batch: 38.0 LinUCB Normalized Accumulated Payoff: 1.72965322009908
Batch: 38.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.559094125973107
----------------------------
Batch: 39.0 LinUCB Normalized Accumulated Payoff: 1.7371705963938973
Batch: 39.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.631067961165049
----------------------------
Batch: 40.0 LinUCB Normalized Accumulated Payoff: 1.7349560513860716
Batch: 40.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.647734956051386
----------------------------
Batch: 41.0 LinUCB Normalized Accumulated Payoff: 1.729551451187335
Batch: 41.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.7097625329815305
----------------------------
Batch: 42.0 LinUCB Normalized Accumulated Payoff: 1.717586649550706
Batch: 42.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.7041078305519894
----------------------------
Batch: 43.0 LinUCB Normalized Accumulated Payoff: 1.709717868338558
Batch: 43.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.747962382445141
----------------------------
Batch: 44.0 LinUCB Normalized Accumulated Payoff: 1.6977886977886978
Batch: 44.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.745085995085995
----------------------------
Batch: 45.0 LinUCB Normalized Accumulated Payoff: 1.6913357400722022
Batch: 45.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.763537906137184
----------------------------
Batch: 47.0 LinUCB Normalized Accumulated Payoff: 1.673187571921749
Batch: 47.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.823935558112773
----------------------------
Batch: 48.0 LinUCB Normalized Accumulated Payoff: 1.6923512747875353
Batch: 48.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.896883852691218
----------------------------
Batch: 49.0 LinUCB Normalized Accumulated Payoff: 1.6947835738068813
Batch: 49.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.9184239733629305
----------------------------
Batch: 50.0 LinUCB Normalized Accumulated Payoff: 1.6983695652173914
Batch: 50.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.966847826086957
----------------------------
Batch: 52.0 LinUCB Normalized Accumulated Payoff: 1.6924281984334204
Batch: 52.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.982767624020887
----------------------------
Batch: 53.0 LinUCB Normalized Accumulated Payoff: 1.7071538857436954
Batch: 53.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.05609881626351
----------------------------
Batch: 54.0 LinUCB Normalized Accumulated Payoff: 1.7163048965169108
Batch: 54.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.065118626956083
----------------------------
Batch: 55.0 LinUCB Normalized Accumulated Payoff: 1.7035573122529644
Batch: 55.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.076086956521739
----------------------------
Batch: 56.0 LinUCB Normalized Accumulated Payoff: 1.6941747572815533
Batch: 56.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.0830097087378645
----------------------------
Batch: 57.0 LinUCB Normalized Accumulated Payoff: 1.683484055211804
Batch: 57.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.086625416468348
----------------------------
Batch: 58.0 LinUCB Normalized Accumulated Payoff: 1.678688524590164
Batch: 58.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.089461358313818
----------------------------
Batch: 59.0 LinUCB Normalized Accumulated Payoff: 1.677685950413223
Batch: 59.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.116161616161616
----------------------------
Batch: 60.0 LinUCB Normalized Accumulated Payoff: 1.6869328493647913
Batch: 60.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.171506352087114
----------------------------
Batch: 61.0 LinUCB Normalized Accumulated Payoff: 1.7024128686327078
Batch: 61.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.202859696157283
----------------------------
Batch: 62.0 LinUCB Normalized Accumulated Payoff: 1.7082415160863818
Batch: 62.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.246804759806082
----------------------------
Batch: 63.0 LinUCB Normalized Accumulated Payoff: 1.7140997830802602
Batch: 63.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.282863340563991
----------------------------
Batch: 64.0 LinUCB Normalized Accumulated Payoff: 1.7199828473413379
Batch: 64.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.342195540308748
----------------------------
Batch: 65.0 LinUCB Normalized Accumulated Payoff: 1.7146464646464648
Batch: 65.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.3367003367003365
----------------------------
Batch: 66.0 LinUCB Normalized Accumulated Payoff: 1.712567399419328
Batch: 66.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.375777685607631
----------------------------
Batch: 67.0 LinUCB Normalized Accumulated Payoff: 1.7117814920505503
Batch: 67.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.3889115368936
----------------------------
Batch: 68.0 LinUCB Normalized Accumulated Payoff: 1.7037630104083266
Batch: 68.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.381104883907126
----------------------------
Batch: 69.0 LinUCB Normalized Accumulated Payoff: 1.7042309213127718
Batch: 69.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.412020561486754
----------------------------
Batch: 70.0 LinUCB Normalized Accumulated Payoff: 1.7076320939334637
Batch: 70.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.439138943248532
----------------------------
Batch: 71.0 LinUCB Normalized Accumulated Payoff: 1.7098145285935085
Batch: 71.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.4857032457496135
----------------------------
Batch: 72.0 LinUCB Normalized Accumulated Payoff: 1.7150476190476192
Batch: 72.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.514666666666667
----------------------------
Batch: 73.0 LinUCB Normalized Accumulated Payoff: 1.7172853393325835
Batch: 73.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.525309336332958
----------------------------
Batch: 75.0 LinUCB Normalized Accumulated Payoff: 1.7226491035492133
Batch: 75.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.5744603000365895
----------------------------
Batch: 76.0 LinUCB Normalized Accumulated Payoff: 1.7254261878853827
Batch: 76.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.595937613347842
----------------------------
Batch: 77.0 LinUCB Normalized Accumulated Payoff: 1.7223419540229885
Batch: 77.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.629310344827586
----------------------------
Batch: 78.0 LinUCB Normalized Accumulated Payoff: 1.729382363441628
Batch: 78.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.671545876472688
----------------------------
Batch: 80.0 LinUCB Normalized Accumulated Payoff: 1.7166724678036895
Batch: 80.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.686042464323007
----------------------------
Batch: 81.0 LinUCB Normalized Accumulated Payoff: 1.7268646408839778
Batch: 81.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.758632596685083
----------------------------
Batch: 84.0 LinUCB Normalized Accumulated Payoff: 1.7210997018880423
Batch: 84.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.784365684001325
----------------------------
Batch: 85.0 LinUCB Normalized Accumulated Payoff: 1.7110091743119267
Batch: 85.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.777850589777195
----------------------------
Batch: 86.0 LinUCB Normalized Accumulated Payoff: 1.7054439403758912
Batch: 86.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.785806869734284
----------------------------
Batch: 87.0 LinUCB Normalized Accumulated Payoff: 1.7028846153846153
Batch: 87.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.794871794871795
----------------------------
Batch: 88.0 LinUCB Normalized Accumulated Payoff: 1.7123809523809523
Batch: 88.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.833015873015873
----------------------------
Batch: 89.0 LinUCB Normalized Accumulated Payoff: 1.711327267022278
Batch: 89.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.876686539064951
----------------------------
Batch: 90.0 LinUCB Normalized Accumulated Payoff: 1.7066666666666668
Batch: 90.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.88031007751938
----------------------------
Batch: 91.0 LinUCB Normalized Accumulated Payoff: 1.7132330365366901
Batch: 91.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.909118821000921
----------------------------
Batch: 92.0 LinUCB Normalized Accumulated Payoff: 1.7120291616038883
Batch: 92.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.925273390036452
----------------------------
Batch: 93.0 LinUCB Normalized Accumulated Payoff: 1.710257948410318
Batch: 93.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.934013197360528
----------------------------
Batch: 94.0 LinUCB Normalized Accumulated Payoff: 1.7158176943699732
Batch: 94.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.9502532022639265
----------------------------
Batch: 96.0 LinUCB Normalized Accumulated Payoff: 1.703325554259043
Batch: 96.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.944282380396733
----------------------------
Batch: 97.0 LinUCB Normalized Accumulated Payoff: 1.70213996529786
Batch: 97.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.9534412955465585
----------------------------
Batch: 98.0 LinUCB Normalized Accumulated Payoff: 1.700256922637739
Batch: 98.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.968027405081359
----------------------------
Batch: 99.0 LinUCB Normalized Accumulated Payoff: 1.7054153671675645
Batch: 99.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.999149418769493
----------------------------
Batch: 100.0 LinUCB Normalized Accumulated Payoff: 1.703869882220976
Batch: 100.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.01402131239484
----------------------------
Batch: 101.0 LinUCB Normalized Accumulated Payoff: 1.7006651884700665
Batch: 101.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.03270509977827
----------------------------
Batch: 102.0 LinUCB Normalized Accumulated Payoff: 1.7091710272652163
Batch: 102.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.063894244009915
----------------------------
Batch: 103.0 LinUCB Normalized Accumulated Payoff: 1.7047930283224402
Batch: 103.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.07380174291939
----------------------------
Batch: 105.0 LinUCB Normalized Accumulated Payoff: 1.6972452527413746
Batch: 105.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.079967905857181
----------------------------
Batch: 106.0 LinUCB Normalized Accumulated Payoff: 1.702365134201435
Batch: 106.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.125697581716715
----------------------------
Batch: 107.0 LinUCB Normalized Accumulated Payoff: 1.705897840968931
Batch: 107.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.132701421800948
----------------------------
Batch: 108.0 LinUCB Normalized Accumulated Payoff: 1.7084967320261437
Batch: 108.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.168366013071895
----------------------------
Batch: 109.0 LinUCB Normalized Accumulated Payoff: 1.7139890994030627
Batch: 109.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.1933558266286015
----------------------------
Batch: 110.0 LinUCB Normalized Accumulated Payoff: 1.7080442045746596
Batch: 110.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.185299408892315
----------------------------
Batch: 111.0 LinUCB Normalized Accumulated Payoff: 1.7080887981627966
Batch: 111.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.21204388874713
----------------------------
Batch: 114.0 LinUCB Normalized Accumulated Payoff: 1.711615154536391
Batch: 114.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.250997008973081
----------------------------
Batch: 115.0 LinUCB Normalized Accumulated Payoff: 1.708888888888889
Batch: 115.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.254074074074074
----------------------------
Batch: 117.0 LinUCB Normalized Accumulated Payoff: 1.7035957240038873
Batch: 117.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.258503401360544
----------------------------
Batch: 119.0 LinUCB Normalized Accumulated Payoff: 1.6972301814708692
Batch: 119.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.234001910219675
----------------------------
Batch: 120.0 LinUCB Normalized Accumulated Payoff: 1.7008547008547008
Batch: 120.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.260446343779677
----------------------------
Batch: 121.0 LinUCB Normalized Accumulated Payoff: 1.7062293534686173
Batch: 121.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.273949976403964
----------------------------
Batch: 122.0 LinUCB Normalized Accumulated Payoff: 1.7087242026266416
Batch: 122.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.299484052532833
----------------------------
Batch: 123.0 LinUCB Normalized Accumulated Payoff: 1.715585627624825
Batch: 123.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.331077928138124
----------------------------
Batch: 124.0 LinUCB Normalized Accumulated Payoff: 1.713855421686747
Batch: 124.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.338044485634847
----------------------------
Batch: 126.0 LinUCB Normalized Accumulated Payoff: 1.715036563071298
Batch: 126.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.365173674588665
----------------------------
Batch: 127.0 LinUCB Normalized Accumulated Payoff: 1.7133772427890075
Batch: 127.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.372700431523961
----------------------------
Batch: 131.0 LinUCB Normalized Accumulated Payoff: 1.707177454865698
Batch: 131.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.371642448260678
----------------------------
Batch: 132.0 LinUCB Normalized Accumulated Payoff: 1.7062937062937062
Batch: 132.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.376748251748252
----------------------------
Batch: 133.0 LinUCB Normalized Accumulated Payoff: 1.7037518976360877
Batch: 133.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.377792235957493
----------------------------
Batch: 134.0 LinUCB Normalized Accumulated Payoff: 1.7023706896551725
Batch: 134.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.376939655172414
----------------------------
Batch: 136.0 LinUCB Normalized Accumulated Payoff: 1.7000212901852245
Batch: 136.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.392591015541835
----------------------------
Batch: 137.0 LinUCB Normalized Accumulated Payoff: 1.7007607776838547
Batch: 137.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.4061707523245985
----------------------------
Batch: 139.0 LinUCB Normalized Accumulated Payoff: 1.697936210131332
Batch: 139.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.407337919533042
----------------------------
Batch: 140.0 LinUCB Normalized Accumulated Payoff: 1.6932312150693438
Batch: 140.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.39557027530532
----------------------------
Batch: 142.0 LinUCB Normalized Accumulated Payoff: 1.6973549313102316
Batch: 142.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.427927004305926
----------------------------
Batch: 143.0 LinUCB Normalized Accumulated Payoff: 1.6962102689486553
Batch: 143.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.425835370823146
----------------------------
Batch: 144.0 LinUCB Normalized Accumulated Payoff: 1.6972644376899697
Batch: 144.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.434447821681864
----------------------------
Batch: 145.0 LinUCB Normalized Accumulated Payoff: 1.6958887545344619
Batch: 145.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.4379282547359935
----------------------------
Batch: 147.0 LinUCB Normalized Accumulated Payoff: 1.6965709728867624
Batch: 147.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.443979266347688
----------------------------
Batch: 148.0 LinUCB Normalized Accumulated Payoff: 1.697582243361078
Batch: 148.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.449861276258423
----------------------------
Batch: 150.0 LinUCB Normalized Accumulated Payoff: 1.6900802191351987
Batch: 150.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.444531402856584
----------------------------
Batch: 151.0 LinUCB Normalized Accumulated Payoff: 1.6875608331711116
Batch: 151.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.442670819544481
----------------------------
Batch: 152.0 LinUCB Normalized Accumulated Payoff: 1.6865440464666022
Batch: 152.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.438334946757019
----------------------------
Batch: 153.0 LinUCB Normalized Accumulated Payoff: 1.6847679568650107
Batch: 153.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.437512035432313
----------------------------
Batch: 154.0 LinUCB Normalized Accumulated Payoff: 1.6822268987947198
Batch: 154.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.44040558637842
----------------------------
Batch: 155.0 LinUCB Normalized Accumulated Payoff: 1.680373120121835
Batch: 155.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.441842756520084
----------------------------
Batch: 156.0 LinUCB Normalized Accumulated Payoff: 1.6810606060606061
Batch: 156.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.445833333333334
----------------------------
Batch: 157.0 LinUCB Normalized Accumulated Payoff: 1.685153744576495
Batch: 157.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.46085644218072
----------------------------
Batch: 159.0 LinUCB Normalized Accumulated Payoff: 1.6792101341281669
Batch: 159.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.456780923994039
----------------------------
Batch: 163.0 LinUCB Normalized Accumulated Payoff: 1.6817598533455544
Batch: 163.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.479926672777268
----------------------------
Batch: 164.0 LinUCB Normalized Accumulated Payoff: 1.6801021338683202
Batch: 164.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.486230165967536
----------------------------
Batch: 165.0 LinUCB Normalized Accumulated Payoff: 1.681727454182544
Batch: 165.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.486844492832517
----------------------------
Batch: 166.0 LinUCB Normalized Accumulated Payoff: 1.6800721370604148
Batch: 166.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.484941388638413
----------------------------
Batch: 167.0 LinUCB Normalized Accumulated Payoff: 1.6774367259019924
Batch: 167.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.4828576557171065
----------------------------
Batch: 168.0 LinUCB Normalized Accumulated Payoff: 1.6743978590544157
Batch: 168.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.477074041034791
----------------------------
Batch: 169.0 LinUCB Normalized Accumulated Payoff: 1.675133214920071
Batch: 169.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.483836589698046
----------------------------
Batch: 170.0 LinUCB Normalized Accumulated Payoff: 1.6748053786270347
Batch: 170.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.485668789808917
----------------------------
Batch: 171.0 LinUCB Normalized Accumulated Payoff: 1.6721889319703913
Batch: 171.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.487663024321466
----------------------------
Batch: 172.0 LinUCB Normalized Accumulated Payoff: 1.6712280701754385
Batch: 172.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.490701754385965
----------------------------
Batch: 174.0 LinUCB Normalized Accumulated Payoff: 1.6722878998609179
Batch: 174.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.496870653685675
----------------------------
Batch: 175.0 LinUCB Normalized Accumulated Payoff: 1.669837426496022
Batch: 175.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.49066067104808
----------------------------
Batch: 176.0 LinUCB Normalized Accumulated Payoff: 1.6685044796691937
Batch: 176.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.495864920744315
----------------------------
Batch: 177.0 LinUCB Normalized Accumulated Payoff: 1.667238421955403
Batch: 177.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.500686106346484
----------------------------
Batch: 179.0 LinUCB Normalized Accumulated Payoff: 1.6652484683458135
Batch: 179.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.5125936010891765
----------------------------
Batch: 180.0 LinUCB Normalized Accumulated Payoff: 1.6657627118644067
Batch: 180.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.52135593220339
----------------------------
Batch: 181.0 LinUCB Normalized Accumulated Payoff: 1.6652613827993255
Batch: 181.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.521922428330523
----------------------------
Batch: 182.0 LinUCB Normalized Accumulated Payoff: 1.6680679334118043
Batch: 182.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.534387085925677
----------------------------
Batch: 183.0 LinUCB Normalized Accumulated Payoff: 1.6681193429433456
Batch: 183.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.550620181025813
----------------------------
Batch: 184.0 LinUCB Normalized Accumulated Payoff: 1.6656104069379587
Batch: 184.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.545530353569046
----------------------------
Batch: 185.0 LinUCB Normalized Accumulated Payoff: 1.6618549858967977
Batch: 185.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.536253525800564
----------------------------
Batch: 186.0 LinUCB Normalized Accumulated Payoff: 1.6639669421487604
Batch: 186.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.540661157024793
----------------------------
Batch: 188.0 LinUCB Normalized Accumulated Payoff: 1.6629526462395543
Batch: 188.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.5413730951990825
----------------------------
Batch: 190.0 LinUCB Normalized Accumulated Payoff: 1.658703071672355
Batch: 190.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.537623923289452
----------------------------
Batch: 192.0 LinUCB Normalized Accumulated Payoff: 1.6555000805282654
Batch: 192.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.545176356901273
----------------------------
Batch: 195.0 LinUCB Normalized Accumulated Payoff: 1.652623211446741
Batch: 195.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.537837837837838
----------------------------
Batch: 197.0 LinUCB Normalized Accumulated Payoff: 1.6499526963103122
Batch: 197.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.536581520025229
----------------------------
Batch: 198.0 LinUCB Normalized Accumulated Payoff: 1.6519007225887528
Batch: 198.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.545083254791078
----------------------------
Batch: 199.0 LinUCB Normalized Accumulated Payoff: 1.6521195057093696
Batch: 199.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.5488815892382295
----------------------------
Batch: 200.0 LinUCB Normalized Accumulated Payoff: 1.6510359869138496
Batch: 200.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.548527808069792
----------------------------
Batch: 201.0 LinUCB Normalized Accumulated Payoff: 1.6509931719428925
Batch: 201.0 Basline_FactorUCB Normalized Accumulated Payoff: 7.551986343885785
----------------------------

"""


"""

固定用户特征，W采用单位矩阵
D:\SoftwareFamily\Anaconda\envs\nlplab\python.exe D:\IIR\Server\PycharmProject\HyperBandit\RUN.py 
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
Batch: 1.0 LinUCB Normalized Accumulated Payoff: 1.5769230769230769
Batch: 1.0 Basline_FactorUCB Normalized Accumulated Payoff: 0.5576923076923077
----------------------------
Batch: 2.0 LinUCB Normalized Accumulated Payoff: 1.8105263157894738
Batch: 2.0 Basline_FactorUCB Normalized Accumulated Payoff: 0.631578947368421
----------------------------
Batch: 3.0 LinUCB Normalized Accumulated Payoff: 1.7956204379562044
Batch: 3.0 Basline_FactorUCB Normalized Accumulated Payoff: 0.8832116788321168
----------------------------
Batch: 4.0 LinUCB Normalized Accumulated Payoff: 1.8313953488372092
Batch: 4.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.1569767441860466
----------------------------
Batch: 5.0 LinUCB Normalized Accumulated Payoff: 1.6545454545454545
Batch: 5.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.3227272727272728
----------------------------
Batch: 6.0 LinUCB Normalized Accumulated Payoff: 1.7049808429118773
Batch: 6.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.4406130268199233
----------------------------
Batch: 7.0 LinUCB Normalized Accumulated Payoff: 1.7466216216216217
Batch: 7.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.5439189189189189
----------------------------
Batch: 8.0 LinUCB Normalized Accumulated Payoff: 1.6627565982404693
Batch: 8.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.5366568914956011
----------------------------
Batch: 9.0 LinUCB Normalized Accumulated Payoff: 1.6380697050938338
Batch: 9.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.5361930294906165
----------------------------
Batch: 10.0 LinUCB Normalized Accumulated Payoff: 1.5645933014354068
Batch: 10.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.5717703349282297
----------------------------
Batch: 11.0 LinUCB Normalized Accumulated Payoff: 1.4893162393162394
Batch: 11.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.5534188034188035
----------------------------
Batch: 12.0 LinUCB Normalized Accumulated Payoff: 1.5481927710843373
Batch: 12.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.7248995983935742
----------------------------
Batch: 14.0 LinUCB Normalized Accumulated Payoff: 1.5631399317406143
Batch: 14.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.901023890784983
----------------------------
Batch: 15.0 LinUCB Normalized Accumulated Payoff: 1.5968
Batch: 15.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0176
----------------------------
Batch: 16.0 LinUCB Normalized Accumulated Payoff: 1.6246200607902737
Batch: 16.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.1306990881458967
----------------------------
Batch: 17.0 LinUCB Normalized Accumulated Payoff: 1.6230440967283073
Batch: 17.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.194879089615932
----------------------------
Batch: 18.0 LinUCB Normalized Accumulated Payoff: 1.6355013550135502
Batch: 18.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.2913279132791327
----------------------------
Batch: 19.0 LinUCB Normalized Accumulated Payoff: 1.6541450777202074
Batch: 19.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.378238341968912
----------------------------
Batch: 20.0 LinUCB Normalized Accumulated Payoff: 1.6537982565379825
Batch: 20.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.476961394769614
----------------------------
Batch: 21.0 LinUCB Normalized Accumulated Payoff: 1.6511350059737155
Batch: 21.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.6057347670250897
----------------------------
Batch: 22.0 LinUCB Normalized Accumulated Payoff: 1.6224489795918366
Batch: 22.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.6485260770975056
----------------------------
Batch: 23.0 LinUCB Normalized Accumulated Payoff: 1.622004357298475
Batch: 23.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.791938997821351
----------------------------
Batch: 24.0 LinUCB Normalized Accumulated Payoff: 1.6429319371727749
Batch: 24.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.918324607329843
----------------------------
Batch: 25.0 LinUCB Normalized Accumulated Payoff: 1.6676829268292683
Batch: 25.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.111788617886179
----------------------------
Batch: 26.0 LinUCB Normalized Accumulated Payoff: 1.6948818897637796
Batch: 26.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.295275590551181
----------------------------
Batch: 27.0 LinUCB Normalized Accumulated Payoff: 1.7045889101338432
Batch: 27.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.482791586998088
----------------------------
Batch: 28.0 LinUCB Normalized Accumulated Payoff: 1.726598702502317
Batch: 28.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.624652455977757
----------------------------
Batch: 29.0 LinUCB Normalized Accumulated Payoff: 1.738973897389739
Batch: 29.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.7695769576957696
----------------------------
Batch: 30.0 LinUCB Normalized Accumulated Payoff: 1.7378048780487805
Batch: 30.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.8858885017421603
----------------------------
Batch: 31.0 LinUCB Normalized Accumulated Payoff: 1.7313182199832073
Batch: 31.0 Basline_FactorUCB Normalized Accumulated Payoff: 3.9916036943744753
----------------------------
Batch: 32.0 LinUCB Normalized Accumulated Payoff: 1.7211382113821139
Batch: 32.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.047967479674797
----------------------------
Batch: 33.0 LinUCB Normalized Accumulated Payoff: 1.706067769897557
Batch: 33.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.127659574468085
----------------------------
Batch: 34.0 LinUCB Normalized Accumulated Payoff: 1.6916030534351145
Batch: 34.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.165648854961832
----------------------------
Batch: 35.0 LinUCB Normalized Accumulated Payoff: 1.659040590405904
Batch: 35.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.199261992619927
----------------------------
Batch: 36.0 LinUCB Normalized Accumulated Payoff: 1.6518305814788228
Batch: 36.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.2426417803302225
----------------------------
Batch: 37.0 LinUCB Normalized Accumulated Payoff: 1.6427076064200976
Batch: 37.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.296580600139567
----------------------------
Batch: 38.0 LinUCB Normalized Accumulated Payoff: 1.64135593220339
Batch: 38.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.381694915254237
----------------------------
Batch: 39.0 LinUCB Normalized Accumulated Payoff: 1.6509308510638299
Batch: 39.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.474734042553192
----------------------------
Batch: 40.0 LinUCB Normalized Accumulated Payoff: 1.6588388780169603
Batch: 40.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.553163731245923
----------------------------
Batch: 41.0 LinUCB Normalized Accumulated Payoff: 1.650571791613723
Batch: 41.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.613087674714104
----------------------------
Batch: 42.0 LinUCB Normalized Accumulated Payoff: 1.6552582451773492
Batch: 42.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.655258245177349
----------------------------
Batch: 43.0 LinUCB Normalized Accumulated Payoff: 1.6554673182651192
Batch: 43.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.7348808796579105
----------------------------
Batch: 44.0 LinUCB Normalized Accumulated Payoff: 1.6398330351818724
Batch: 44.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.7412045319022065
----------------------------
Batch: 45.0 LinUCB Normalized Accumulated Payoff: 1.6335476329631795
Batch: 45.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.77323202805377
----------------------------
Batch: 47.0 LinUCB Normalized Accumulated Payoff: 1.626336522228475
Batch: 47.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.899831176139561
----------------------------
Batch: 48.0 LinUCB Normalized Accumulated Payoff: 1.637816979051819
Batch: 48.0 Basline_FactorUCB Normalized Accumulated Payoff: 4.949283351708931
----------------------------
Batch: 49.0 LinUCB Normalized Accumulated Payoff: 1.6489419424850786
Batch: 49.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.010309278350515
----------------------------
Batch: 50.0 LinUCB Normalized Accumulated Payoff: 1.6551357104843
Batch: 50.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.072911122937733
----------------------------
Batch: 52.0 LinUCB Normalized Accumulated Payoff: 1.6447166921898928
Batch: 52.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.104645227156713
----------------------------
Batch: 53.0 LinUCB Normalized Accumulated Payoff: 1.6547559134373426
Batch: 53.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.180674383492702
----------------------------
Batch: 54.0 LinUCB Normalized Accumulated Payoff: 1.6631996037642398
Batch: 54.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.20901436354631
----------------------------
Batch: 55.0 LinUCB Normalized Accumulated Payoff: 1.66455078125
Batch: 55.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.26513671875
----------------------------
Batch: 56.0 LinUCB Normalized Accumulated Payoff: 1.6624939817043813
Batch: 56.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.2961001444390945
----------------------------
Batch: 57.0 LinUCB Normalized Accumulated Payoff: 1.6540463795551348
Batch: 57.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.326076668244203
----------------------------
Batch: 58.0 LinUCB Normalized Accumulated Payoff: 1.6629160806375995
Batch: 58.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.383028598218472
----------------------------
Batch: 59.0 LinUCB Normalized Accumulated Payoff: 1.6619069553201289
Batch: 59.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.41731920773837
----------------------------
Batch: 60.0 LinUCB Normalized Accumulated Payoff: 1.6601721794290893
Batch: 60.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.442682374263707
----------------------------
Batch: 61.0 LinUCB Normalized Accumulated Payoff: 1.6731967943009796
Batch: 61.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.463045414069457
----------------------------
Batch: 62.0 LinUCB Normalized Accumulated Payoff: 1.6844424856765094
Batch: 62.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.527985896870868
----------------------------
Batch: 63.0 LinUCB Normalized Accumulated Payoff: 1.6883398352839185
Batch: 63.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.556133506718682
----------------------------
Batch: 64.0 LinUCB Normalized Accumulated Payoff: 1.6936589545844045
Batch: 64.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.6139674378748925
----------------------------
Batch: 65.0 LinUCB Normalized Accumulated Payoff: 1.6921452702702702
Batch: 65.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.6456925675675675
----------------------------
Batch: 66.0 LinUCB Normalized Accumulated Payoff: 1.6849315068493151
Batch: 66.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.660855126608551
----------------------------
Batch: 67.0 LinUCB Normalized Accumulated Payoff: 1.6861463015937883
Batch: 67.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.682468328565591
----------------------------
Batch: 68.0 LinUCB Normalized Accumulated Payoff: 1.6791314837153197
Batch: 68.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.688379573783675
----------------------------
Batch: 69.0 LinUCB Normalized Accumulated Payoff: 1.676984126984127
Batch: 69.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.718253968253968
----------------------------
Batch: 70.0 LinUCB Normalized Accumulated Payoff: 1.6726632772780603
Batch: 70.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.720375439968714
----------------------------
Batch: 71.0 LinUCB Normalized Accumulated Payoff: 1.6641074856046065
Batch: 71.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.727831094049904
----------------------------
Batch: 72.0 LinUCB Normalized Accumulated Payoff: 1.6680545041635124
Batch: 72.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.758516275548827
----------------------------
Batch: 73.0 LinUCB Normalized Accumulated Payoff: 1.669776119402985
Batch: 73.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.773134328358209
----------------------------
Batch: 75.0 LinUCB Normalized Accumulated Payoff: 1.6721431179262505
Batch: 75.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.828404527199708
----------------------------
Batch: 76.0 LinUCB Normalized Accumulated Payoff: 1.6700361010830325
Batch: 76.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.848736462093862
----------------------------
Batch: 77.0 LinUCB Normalized Accumulated Payoff: 1.6527137282724371
Batch: 77.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.839304717985101
----------------------------
Batch: 78.0 LinUCB Normalized Accumulated Payoff: 1.6454545454545455
Batch: 78.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.834265734265735
----------------------------
Batch: 80.0 LinUCB Normalized Accumulated Payoff: 1.6304273504273503
Batch: 80.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.877606837606837
----------------------------
Batch: 81.0 LinUCB Normalized Accumulated Payoff: 1.6308679500168861
Batch: 81.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.912191827085444
----------------------------
Batch: 84.0 LinUCB Normalized Accumulated Payoff: 1.6341383812010444
Batch: 84.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.9657310704960835
----------------------------
Batch: 85.0 LinUCB Normalized Accumulated Payoff: 1.6260897642880208
Batch: 85.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.9696480464966095
----------------------------
Batch: 86.0 LinUCB Normalized Accumulated Payoff: 1.6232023010546501
Batch: 86.0 Basline_FactorUCB Normalized Accumulated Payoff: 5.981783317353787
----------------------------
Batch: 87.0 LinUCB Normalized Accumulated Payoff: 1.6234177215189873
Batch: 87.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.003164556962025
----------------------------
Batch: 88.0 LinUCB Normalized Accumulated Payoff: 1.6234779893849516
Batch: 88.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.014361536059944
----------------------------
Batch: 89.0 LinUCB Normalized Accumulated Payoff: 1.6278207109737248
Batch: 89.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.045440494590418
----------------------------
Batch: 90.0 LinUCB Normalized Accumulated Payoff: 1.6283700980392157
Batch: 90.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.0661764705882355
----------------------------
Batch: 91.0 LinUCB Normalized Accumulated Payoff: 1.6298760205624434
Batch: 91.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.071666162685213
----------------------------
Batch: 92.0 LinUCB Normalized Accumulated Payoff: 1.632983508245877
Batch: 92.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.095052473763118
----------------------------
Batch: 93.0 LinUCB Normalized Accumulated Payoff: 1.6365527488855869
Batch: 93.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.121545319465081
----------------------------
Batch: 94.0 LinUCB Normalized Accumulated Payoff: 1.6376214306741241
Batch: 94.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.1324698263173385
----------------------------
Batch: 96.0 LinUCB Normalized Accumulated Payoff: 1.63249348392702
Batch: 96.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.1711555169417895
----------------------------
Batch: 97.0 LinUCB Normalized Accumulated Payoff: 1.6319862424763543
Batch: 97.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.172828890799656
----------------------------
Batch: 98.0 LinUCB Normalized Accumulated Payoff: 1.6394654535115154
Batch: 98.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.211259596246801
----------------------------
Batch: 99.0 LinUCB Normalized Accumulated Payoff: 1.6369766788423714
Batch: 99.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.209328463051419
----------------------------
Batch: 100.0 LinUCB Normalized Accumulated Payoff: 1.6367426347971095
Batch: 100.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.214285714285714
----------------------------
Batch: 101.0 LinUCB Normalized Accumulated Payoff: 1.641733370135247
Batch: 101.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.255313276290367
----------------------------
Batch: 102.0 LinUCB Normalized Accumulated Payoff: 1.6479452054794521
Batch: 102.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.277260273972603
----------------------------
Batch: 103.0 LinUCB Normalized Accumulated Payoff: 1.6468512486427795
Batch: 103.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.292888165038002
----------------------------
Batch: 105.0 LinUCB Normalized Accumulated Payoff: 1.641559829059829
Batch: 105.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.323985042735043
----------------------------
Batch: 106.0 LinUCB Normalized Accumulated Payoff: 1.6411749139984122
Batch: 106.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.33633236305901
----------------------------
Batch: 107.0 LinUCB Normalized Accumulated Payoff: 1.6463638750328171
Batch: 107.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.351535836177474
----------------------------
Batch: 108.0 LinUCB Normalized Accumulated Payoff: 1.6496483459234175
Batch: 108.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.374837197186768
----------------------------
Batch: 109.0 LinUCB Normalized Accumulated Payoff: 1.6514447884416925
Batch: 109.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.381836945304437
----------------------------
Batch: 110.0 LinUCB Normalized Accumulated Payoff: 1.6462010744435918
Batch: 110.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.383218214377078
----------------------------
Batch: 111.0 LinUCB Normalized Accumulated Payoff: 1.6435091277890466
Batch: 111.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.397565922920893
----------------------------
Batch: 114.0 LinUCB Normalized Accumulated Payoff: 1.6461271962385549
Batch: 114.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.432813659985152
----------------------------
Batch: 115.0 LinUCB Normalized Accumulated Payoff: 1.6466110019646365
Batch: 115.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.446709233791749
----------------------------
Batch: 117.0 LinUCB Normalized Accumulated Payoff: 1.6405307599517491
Batch: 117.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.444390832328106
----------------------------
Batch: 119.0 LinUCB Normalized Accumulated Payoff: 1.6383079847908746
Batch: 119.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.449144486692015
----------------------------
Batch: 120.0 LinUCB Normalized Accumulated Payoff: 1.6391509433962264
Batch: 120.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.459905660377358
----------------------------
Batch: 121.0 LinUCB Normalized Accumulated Payoff: 1.6443298969072164
Batch: 121.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.471883786316776
----------------------------
Batch: 122.0 LinUCB Normalized Accumulated Payoff: 1.6432558139534883
Batch: 122.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.482558139534884
----------------------------
Batch: 123.0 LinUCB Normalized Accumulated Payoff: 1.6462710690371738
Batch: 123.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.496190256291849
----------------------------
Batch: 124.0 LinUCB Normalized Accumulated Payoff: 1.645938503900872
Batch: 124.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.507572280862781
----------------------------
Batch: 126.0 LinUCB Normalized Accumulated Payoff: 1.6466591166477915
Batch: 126.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.537712344280861
----------------------------
Batch: 127.0 LinUCB Normalized Accumulated Payoff: 1.643579941533618
Batch: 127.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.539239937036204
----------------------------
Batch: 131.0 LinUCB Normalized Accumulated Payoff: 1.637652705061082
Batch: 131.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.543630017452007
----------------------------
Batch: 132.0 LinUCB Normalized Accumulated Payoff: 1.6354392038078753
Batch: 132.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.541324102120294
----------------------------
Batch: 133.0 LinUCB Normalized Accumulated Payoff: 1.6351525569402665
Batch: 133.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.546841426729695
----------------------------
Batch: 134.0 LinUCB Normalized Accumulated Payoff: 1.633119658119658
Batch: 134.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.560042735042735
----------------------------
Batch: 136.0 LinUCB Normalized Accumulated Payoff: 1.6311233108108107
Batch: 136.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.578758445945946
----------------------------
Batch: 137.0 LinUCB Normalized Accumulated Payoff: 1.63325644789264
Batch: 137.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.590060809394003
----------------------------
Batch: 139.0 LinUCB Normalized Accumulated Payoff: 1.6320364238410596
Batch: 139.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.602235099337748
----------------------------
Batch: 140.0 LinUCB Normalized Accumulated Payoff: 1.6274147143444306
Batch: 140.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.595561035758323
----------------------------
Batch: 142.0 LinUCB Normalized Accumulated Payoff: 1.6268535445866341
Batch: 142.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.603290676416819
----------------------------
Batch: 143.0 LinUCB Normalized Accumulated Payoff: 1.6255799878959047
Batch: 143.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.59834577365342
----------------------------
Batch: 144.0 LinUCB Normalized Accumulated Payoff: 1.6266051364365972
Batch: 144.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.604735152487962
----------------------------
Batch: 145.0 LinUCB Normalized Accumulated Payoff: 1.6245264207377867
Batch: 145.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.6009970089730805
----------------------------
Batch: 147.0 LinUCB Normalized Accumulated Payoff: 1.6261829652996846
Batch: 147.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.601932176656152
----------------------------
Batch: 148.0 LinUCB Normalized Accumulated Payoff: 1.6276699980403684
Batch: 148.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.604938271604938
----------------------------
Batch: 150.0 LinUCB Normalized Accumulated Payoff: 1.622187742435997
Batch: 150.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.617339022498061
----------------------------
Batch: 151.0 LinUCB Normalized Accumulated Payoff: 1.6185885075202469
Batch: 151.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.61106826070189
----------------------------
Batch: 152.0 LinUCB Normalized Accumulated Payoff: 1.6202191043628675
Batch: 152.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.622909859696329
----------------------------
Batch: 153.0 LinUCB Normalized Accumulated Payoff: 1.6196671130667688
Batch: 153.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.631719915821695
----------------------------
Batch: 154.0 LinUCB Normalized Accumulated Payoff: 1.6193192622171515
Batch: 154.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.637193382772391
----------------------------
Batch: 155.0 LinUCB Normalized Accumulated Payoff: 1.6188221927665214
Batch: 155.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.641166445748911
----------------------------
Batch: 156.0 LinUCB Normalized Accumulated Payoff: 1.6185993975903614
Batch: 156.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.639495481927711
----------------------------
Batch: 157.0 LinUCB Normalized Accumulated Payoff: 1.622726420401275
Batch: 157.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.650665666604163
----------------------------
Batch: 159.0 LinUCB Normalized Accumulated Payoff: 1.6279329608938546
Batch: 159.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.682495344506518
----------------------------
Batch: 163.0 LinUCB Normalized Accumulated Payoff: 1.6349525200876551
Batch: 163.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.694119795471146
----------------------------
Batch: 164.0 LinUCB Normalized Accumulated Payoff: 1.6331274972756993
Batch: 164.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.695604794769342
----------------------------
Batch: 165.0 LinUCB Normalized Accumulated Payoff: 1.6353111432706222
Batch: 165.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.69952966714906
----------------------------
Batch: 166.0 LinUCB Normalized Accumulated Payoff: 1.638803819131688
Batch: 166.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.710682759863087
----------------------------
Batch: 167.0 LinUCB Normalized Accumulated Payoff: 1.6368364418938306
Batch: 167.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.714311334289813
----------------------------
Batch: 168.0 LinUCB Normalized Accumulated Payoff: 1.6377432601321193
Batch: 168.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.715943581503303
----------------------------
Batch: 169.0 LinUCB Normalized Accumulated Payoff: 1.6375266524520256
Batch: 169.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.716595593461265
----------------------------
Batch: 170.0 LinUCB Normalized Accumulated Payoff: 1.6362349610757254
Batch: 170.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.716737438075017
----------------------------
Batch: 171.0 LinUCB Normalized Accumulated Payoff: 1.6326818101778482
Batch: 171.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.717027645712274
----------------------------
Batch: 172.0 LinUCB Normalized Accumulated Payoff: 1.6324741364194284
Batch: 172.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.721199368753288
----------------------------
Batch: 174.0 LinUCB Normalized Accumulated Payoff: 1.635099913119027
Batch: 174.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.73223284100782
----------------------------
Batch: 175.0 LinUCB Normalized Accumulated Payoff: 1.6357031655422938
Batch: 175.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.732053277979588
----------------------------
Batch: 176.0 LinUCB Normalized Accumulated Payoff: 1.6307824591573516
Batch: 176.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.721066208082545
----------------------------
Batch: 177.0 LinUCB Normalized Accumulated Payoff: 1.630803494946034
Batch: 177.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.729141682371081
----------------------------
Batch: 179.0 LinUCB Normalized Accumulated Payoff: 1.6252967107494065
Batch: 179.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.7268565615462865
----------------------------
Batch: 180.0 LinUCB Normalized Accumulated Payoff: 1.6263717710619618
Batch: 180.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.73543812257302
----------------------------
Batch: 181.0 LinUCB Normalized Accumulated Payoff: 1.6251259657373194
Batch: 181.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.734128317097749
----------------------------
Batch: 182.0 LinUCB Normalized Accumulated Payoff: 1.6269881131759585
Batch: 182.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.744684413192701
----------------------------
Batch: 183.0 LinUCB Normalized Accumulated Payoff: 1.6270450751252086
Batch: 183.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.758597662771286
----------------------------
Batch: 184.0 LinUCB Normalized Accumulated Payoff: 1.6245847176079735
Batch: 184.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.756810631229236
----------------------------
Batch: 185.0 LinUCB Normalized Accumulated Payoff: 1.6223949718822361
Batch: 185.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.758021832616606
----------------------------
Batch: 186.0 LinUCB Normalized Accumulated Payoff: 1.6220576131687243
Batch: 186.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.757201646090535
----------------------------
Batch: 188.0 LinUCB Normalized Accumulated Payoff: 1.6204927394354707
Batch: 188.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.759014521129059
----------------------------
Batch: 190.0 LinUCB Normalized Accumulated Payoff: 1.617028164454516
Batch: 190.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.758983489802525
----------------------------
Batch: 192.0 LinUCB Normalized Accumulated Payoff: 1.61343594676928
Batch: 192.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.757575757575758
----------------------------
Batch: 195.0 LinUCB Normalized Accumulated Payoff: 1.6119639183415098
Batch: 195.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.753758506092736
----------------------------
Batch: 197.0 LinUCB Normalized Accumulated Payoff: 1.6114689709347998
Batch: 197.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.7643362136685
----------------------------
Batch: 198.0 LinUCB Normalized Accumulated Payoff: 1.6154087065455685
Batch: 198.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.775289696210461
----------------------------
Batch: 199.0 LinUCB Normalized Accumulated Payoff: 1.6161411177021543
Batch: 199.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.783015922572588
----------------------------
Batch: 200.0 LinUCB Normalized Accumulated Payoff: 1.616210329807094
Batch: 200.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.78562538892346
----------------------------
Batch: 201.0 LinUCB Normalized Accumulated Payoff: 1.6168629882207068
Batch: 201.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.791537507749535
----------------------------
Batch: 202.0 LinUCB Normalized Accumulated Payoff: 1.617687914801667
Batch: 202.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.7941040283994445
----------------------------
Batch: 203.0 LinUCB Normalized Accumulated Payoff: 1.6174888581527587
Batch: 203.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.797141539880129
----------------------------
Batch: 205.0 LinUCB Normalized Accumulated Payoff: 1.61221071863581
Batch: 205.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.792174177831912
----------------------------
Batch: 207.0 LinUCB Normalized Accumulated Payoff: 1.615349750717631
Batch: 207.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.80586191267563
----------------------------
Batch: 209.0 LinUCB Normalized Accumulated Payoff: 1.6154883163571
Batch: 209.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.815008987417615
----------------------------
Batch: 211.0 LinUCB Normalized Accumulated Payoff: 1.6156705322628606
Batch: 211.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.822777282188522
----------------------------
Batch: 213.0 LinUCB Normalized Accumulated Payoff: 1.6144667059516795
Batch: 213.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.820565704183854
----------------------------
Batch: 214.0 LinUCB Normalized Accumulated Payoff: 1.6115266168059832
Batch: 214.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.818741750989881
----------------------------
Batch: 216.0 LinUCB Normalized Accumulated Payoff: 1.6144104803493449
Batch: 216.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.828966521106259
----------------------------
Batch: 217.0 LinUCB Normalized Accumulated Payoff: 1.6115858073859521
Batch: 217.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.823606082548878
----------------------------
Batch: 218.0 LinUCB Normalized Accumulated Payoff: 1.6107585809056821
Batch: 218.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.824632246899337
----------------------------
Batch: 219.0 LinUCB Normalized Accumulated Payoff: 1.609707064905227
Batch: 219.0 Basline_FactorUCB Normalized Accumulated Payoff: 6.822946582423894
----------------------------
Traceback (most recent call last):
  File "D:\IIR\Server\PycharmProject\HyperBandit\RUN.py", line 222, in <module>
    with open("./result_log/resultdata_{}.pkl".format(util.get_time()), "wb") as f:
OSError: [Errno 22] Invalid argument: './result_log/resultdata_2023-04-04 09:57:37.pkl'

进程已结束,退出代码
"""



"""

D:\SoftwareFamily\Anaconda\envs\nlplab\python.exe D:\IIR\Server\PycharmProject\HyperBandit\RUN.py 
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
新商品
Batch: 1.0 LinUCB Normalized Accumulated Payoff: 2.1025641025641026
Batch: 1.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.7179487179487178
----------------------------
Batch: 2.0 LinUCB Normalized Accumulated Payoff: 2.058139534883721
Batch: 2.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.802325581395349
----------------------------
Batch: 3.0 LinUCB Normalized Accumulated Payoff: 1.968
Batch: 3.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.784
----------------------------
Batch: 4.0 LinUCB Normalized Accumulated Payoff: 1.8304093567251463
Batch: 4.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.7894736842105263
----------------------------
Batch: 5.0 LinUCB Normalized Accumulated Payoff: 1.665137614678899
Batch: 5.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.665137614678899
----------------------------
Batch: 6.0 LinUCB Normalized Accumulated Payoff: 1.6802973977695168
Batch: 6.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.691449814126394
----------------------------
Batch: 7.0 LinUCB Normalized Accumulated Payoff: 1.684887459807074
Batch: 7.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.7009646302250805
----------------------------
Batch: 8.0 LinUCB Normalized Accumulated Payoff: 1.6580459770114941
Batch: 8.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.6867816091954022
----------------------------
Batch: 9.0 LinUCB Normalized Accumulated Payoff: 1.646437994722955
Batch: 9.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.6807387862796834
----------------------------
Batch: 10.0 LinUCB Normalized Accumulated Payoff: 1.6062052505966586
Batch: 10.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.639618138424821
----------------------------
Batch: 11.0 LinUCB Normalized Accumulated Payoff: 1.5871964679911699
Batch: 11.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.6335540838852096
----------------------------
Batch: 12.0 LinUCB Normalized Accumulated Payoff: 1.5983935742971886
Batch: 12.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.6465863453815262
----------------------------
Batch: 14.0 LinUCB Normalized Accumulated Payoff: 1.6573426573426573
Batch: 14.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.6783216783216783
----------------------------
Batch: 15.0 LinUCB Normalized Accumulated Payoff: 1.7107438016528926
Batch: 15.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.7388429752066115
----------------------------
Batch: 16.0 LinUCB Normalized Accumulated Payoff: 1.7563291139240507
Batch: 16.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.7674050632911393
----------------------------
Batch: 17.0 LinUCB Normalized Accumulated Payoff: 1.816793893129771
Batch: 17.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.83206106870229
----------------------------
Batch: 18.0 LinUCB Normalized Accumulated Payoff: 1.793991416309013
Batch: 18.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8197424892703862
----------------------------
Batch: 19.0 LinUCB Normalized Accumulated Payoff: 1.8038147138964578
Batch: 19.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8215258855585832
----------------------------
Batch: 20.0 LinUCB Normalized Accumulated Payoff: 1.82010582010582
Batch: 20.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8452380952380953
----------------------------
Batch: 21.0 LinUCB Normalized Accumulated Payoff: 1.8134517766497462
Batch: 21.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8451776649746192
----------------------------
Batch: 22.0 LinUCB Normalized Accumulated Payoff: 1.7927272727272727
Batch: 22.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8315151515151515
----------------------------
Batch: 23.0 LinUCB Normalized Accumulated Payoff: 1.7932636469221834
Batch: 23.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.829268292682927
----------------------------
Batch: 24.0 LinUCB Normalized Accumulated Payoff: 1.8363028953229399
Batch: 24.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.856347438752784
----------------------------
Batch: 25.0 LinUCB Normalized Accumulated Payoff: 1.8622174381054897
Batch: 25.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8869752421959096
----------------------------
Batch: 26.0 LinUCB Normalized Accumulated Payoff: 1.867420349434738
Batch: 26.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8848920863309353
----------------------------
Batch: 27.0 LinUCB Normalized Accumulated Payoff: 1.845927379784102
Batch: 27.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8714425907752699
----------------------------
Batch: 28.0 LinUCB Normalized Accumulated Payoff: 1.8691943127962085
Batch: 28.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8938388625592417
----------------------------
Batch: 29.0 LinUCB Normalized Accumulated Payoff: 1.8767249310027598
Batch: 29.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.906163753449862
----------------------------
Batch: 30.0 LinUCB Normalized Accumulated Payoff: 1.873114463176575
Batch: 30.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9006211180124224
----------------------------
Batch: 31.0 LinUCB Normalized Accumulated Payoff: 1.8811369509043927
Batch: 31.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9155900086132644
----------------------------
Batch: 32.0 LinUCB Normalized Accumulated Payoff: 1.8760469011725294
Batch: 32.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.912897822445561
----------------------------
Batch: 33.0 LinUCB Normalized Accumulated Payoff: 1.8631578947368421
Batch: 33.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8987854251012146
----------------------------
Batch: 34.0 LinUCB Normalized Accumulated Payoff: 1.8639240506329113
Batch: 34.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9066455696202531
----------------------------
Batch: 35.0 LinUCB Normalized Accumulated Payoff: 1.8457979953739398
Batch: 35.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8928296067848882
----------------------------
Batch: 36.0 LinUCB Normalized Accumulated Payoff: 1.8278688524590163
Batch: 36.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8725782414307004
----------------------------
Batch: 37.0 LinUCB Normalized Accumulated Payoff: 1.8116883116883118
Batch: 37.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8556998556998556
----------------------------
Batch: 38.0 LinUCB Normalized Accumulated Payoff: 1.8170988086895585
Batch: 38.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.861948142957253
----------------------------
Batch: 39.0 LinUCB Normalized Accumulated Payoff: 1.8199863107460643
Batch: 39.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8685831622176592
----------------------------
Batch: 40.0 LinUCB Normalized Accumulated Payoff: 1.8227424749163879
Batch: 40.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8722408026755852
----------------------------
Batch: 41.0 LinUCB Normalized Accumulated Payoff: 1.8264571054354943
Batch: 41.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8788474132285526
----------------------------
Batch: 42.0 LinUCB Normalized Accumulated Payoff: 1.8273427471116817
Batch: 42.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8812580231065468
----------------------------
Batch: 43.0 LinUCB Normalized Accumulated Payoff: 1.8167604752970608
Batch: 43.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8699186991869918
----------------------------
Batch: 44.0 LinUCB Normalized Accumulated Payoff: 1.8063725490196079
Batch: 44.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.860906862745098
----------------------------
Batch: 45.0 LinUCB Normalized Accumulated Payoff: 1.8040865384615385
Batch: 45.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8563701923076923
----------------------------
Batch: 47.0 LinUCB Normalized Accumulated Payoff: 1.804524361948956
Batch: 47.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8584686774941996
----------------------------
Batch: 48.0 LinUCB Normalized Accumulated Payoff: 1.8121798520204895
Batch: 48.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8696642003414912
----------------------------
Batch: 49.0 LinUCB Normalized Accumulated Payoff: 1.8160535117056855
Batch: 49.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8729096989966556
----------------------------
Batch: 50.0 LinUCB Normalized Accumulated Payoff: 1.8306849315068494
Batch: 50.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.8865753424657534
----------------------------
Batch: 52.0 LinUCB Normalized Accumulated Payoff: 1.8350079323109465
Batch: 52.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.889476467477525
----------------------------
Batch: 53.0 LinUCB Normalized Accumulated Payoff: 1.8495575221238938
Batch: 53.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9083810515356585
----------------------------
Batch: 54.0 LinUCB Normalized Accumulated Payoff: 1.8631471040492056
Batch: 54.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9215786776012302
----------------------------
Batch: 55.0 LinUCB Normalized Accumulated Payoff: 1.859808371154816
Batch: 55.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9213313161875945
----------------------------
Batch: 56.0 LinUCB Normalized Accumulated Payoff: 1.8549428713363139
Batch: 56.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9175360158966717
----------------------------
Batch: 57.0 LinUCB Normalized Accumulated Payoff: 1.8543403629230015
Batch: 57.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9156449239823443
----------------------------
Batch: 58.0 LinUCB Normalized Accumulated Payoff: 1.8537644787644787
Batch: 58.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9165057915057915
----------------------------
Batch: 59.0 LinUCB Normalized Accumulated Payoff: 1.8599905078310395
Batch: 59.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9231134314190792
----------------------------
Batch: 60.0 LinUCB Normalized Accumulated Payoff: 1.8718911309244486
Batch: 60.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9310183012670108
----------------------------
Batch: 61.0 LinUCB Normalized Accumulated Payoff: 1.896103896103896
Batch: 61.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.955009276437848
----------------------------
Batch: 62.0 LinUCB Normalized Accumulated Payoff: 1.8977635782747604
Batch: 62.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9584664536741214
----------------------------
Batch: 63.0 LinUCB Normalized Accumulated Payoff: 1.9064327485380117
Batch: 63.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9689608636977058
----------------------------
Batch: 64.0 LinUCB Normalized Accumulated Payoff: 1.9069561364643333
Batch: 64.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9663269827204253
----------------------------
Batch: 65.0 LinUCB Normalized Accumulated Payoff: 1.90239651416122
Batch: 65.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9607843137254901
----------------------------
Batch: 66.0 LinUCB Normalized Accumulated Payoff: 1.8996138996138996
Batch: 66.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9588159588159588
----------------------------
Batch: 67.0 LinUCB Normalized Accumulated Payoff: 1.9035532994923858
Batch: 67.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9686971235194586
----------------------------
Batch: 68.0 LinUCB Normalized Accumulated Payoff: 1.8964656964656965
Batch: 68.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9613305613305614
----------------------------
Batch: 69.0 LinUCB Normalized Accumulated Payoff: 1.8911620294599019
Batch: 69.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9541734860883797
----------------------------
Batch: 70.0 LinUCB Normalized Accumulated Payoff: 1.8994322789943228
Batch: 70.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.964720194647202
----------------------------
Batch: 71.0 LinUCB Normalized Accumulated Payoff: 1.9016
Batch: 71.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.972
----------------------------
Batch: 72.0 LinUCB Normalized Accumulated Payoff: 1.9160728424386382
Batch: 72.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9861441013460015
----------------------------
Batch: 73.0 LinUCB Normalized Accumulated Payoff: 1.9276495893625343
Batch: 73.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9972624168947986
----------------------------
Batch: 75.0 LinUCB Normalized Accumulated Payoff: 1.928598701794578
Batch: 75.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0007636502481865
----------------------------
Batch: 76.0 LinUCB Normalized Accumulated Payoff: 1.9300567107750473
Batch: 76.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0018903591682418
----------------------------
Batch: 77.0 LinUCB Normalized Accumulated Payoff: 1.928544706322484
Batch: 77.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9970071081182192
----------------------------
Batch: 78.0 LinUCB Normalized Accumulated Payoff: 1.922082717872969
Batch: 78.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9922451994091581
----------------------------
Batch: 80.0 LinUCB Normalized Accumulated Payoff: 1.9085673146148308
Batch: 80.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9827213822894167
----------------------------
Batch: 81.0 LinUCB Normalized Accumulated Payoff: 1.9124243503025988
Batch: 81.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9911000355998576
----------------------------
Batch: 84.0 LinUCB Normalized Accumulated Payoff: 1.9271408839779005
Batch: 84.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.014157458563536
----------------------------
Batch: 85.0 LinUCB Normalized Accumulated Payoff: 1.9150170648464164
Batch: 85.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.001023890784983
----------------------------
Batch: 86.0 LinUCB Normalized Accumulated Payoff: 1.9121027721433401
Batch: 86.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0
----------------------------
Batch: 87.0 LinUCB Normalized Accumulated Payoff: 1.905937291527685
Batch: 87.0 Basline_FactorUCB Normalized Accumulated Payoff: 1.9929953302201469
----------------------------
Batch: 88.0 LinUCB Normalized Accumulated Payoff: 1.918623883559378
Batch: 88.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0082699305325837
----------------------------
Batch: 89.0 LinUCB Normalized Accumulated Payoff: 1.9170747633039504
Batch: 89.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.006856023506366
----------------------------
Batch: 90.0 LinUCB Normalized Accumulated Payoff: 1.9085346215780998
Batch: 90.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0
----------------------------
Batch: 91.0 LinUCB Normalized Accumulated Payoff: 1.916427432216906
Batch: 91.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.00829346092504
----------------------------
Batch: 92.0 LinUCB Normalized Accumulated Payoff: 1.9245641838351824
Batch: 92.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.018066561014263
----------------------------
Batch: 93.0 LinUCB Normalized Accumulated Payoff: 1.928414442700157
Batch: 93.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.022605965463108
----------------------------
Batch: 94.0 LinUCB Normalized Accumulated Payoff: 1.93179694799128
Batch: 94.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0249143568981625
----------------------------
Batch: 96.0 LinUCB Normalized Accumulated Payoff: 1.925756186984418
Batch: 96.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.020164986251146
----------------------------
Batch: 97.0 LinUCB Normalized Accumulated Payoff: 1.9237980042334442
Batch: 97.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0175385545811912
----------------------------
Batch: 98.0 LinUCB Normalized Accumulated Payoff: 1.9296407185628743
Batch: 98.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0251497005988024
----------------------------
Batch: 99.0 LinUCB Normalized Accumulated Payoff: 1.9375743162901309
Batch: 99.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0315101070154578
----------------------------
Batch: 100.0 LinUCB Normalized Accumulated Payoff: 1.939304655274013
Batch: 100.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0344725987035948
----------------------------
Batch: 101.0 LinUCB Normalized Accumulated Payoff: 1.9334302325581396
Batch: 101.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.027906976744186
----------------------------
Batch: 102.0 LinUCB Normalized Accumulated Payoff: 1.938328530259366
Batch: 102.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.029971181556196
----------------------------
Batch: 103.0 LinUCB Normalized Accumulated Payoff: 1.9325170842824602
Batch: 103.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0247722095671983
----------------------------
Batch: 105.0 LinUCB Normalized Accumulated Payoff: 1.9172240802675586
Batch: 105.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0097547380156078
----------------------------
Batch: 106.0 LinUCB Normalized Accumulated Payoff: 1.9177022921844795
Batch: 106.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0138083402375035
----------------------------
Batch: 107.0 LinUCB Normalized Accumulated Payoff: 1.9273176083379044
Batch: 107.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0211190345584202
----------------------------
Batch: 108.0 LinUCB Normalized Accumulated Payoff: 1.9254136154054786
Batch: 108.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0197992948196366
----------------------------
Batch: 109.0 LinUCB Normalized Accumulated Payoff: 1.932454251883746
Batch: 109.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0266415500538213
----------------------------
Batch: 110.0 LinUCB Normalized Accumulated Payoff: 1.926913843691651
Batch: 110.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.021072285942918
----------------------------
Batch: 111.0 LinUCB Normalized Accumulated Payoff: 1.930084745762712
Batch: 111.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0235699152542375
----------------------------
Batch: 114.0 LinUCB Normalized Accumulated Payoff: 1.9371605896043445
Batch: 114.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0299974140160333
----------------------------
Batch: 115.0 LinUCB Normalized Accumulated Payoff: 1.9332139201637666
Batch: 115.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0276356192425795
----------------------------
Batch: 117.0 LinUCB Normalized Accumulated Payoff: 1.9299395161290323
Batch: 117.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0269657258064515
----------------------------
Batch: 119.0 LinUCB Normalized Accumulated Payoff: 1.9265326383718044
Batch: 119.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0203524447753787
----------------------------
Batch: 120.0 LinUCB Normalized Accumulated Payoff: 1.9268833087149189
Batch: 120.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.021171836533727
----------------------------
Batch: 121.0 LinUCB Normalized Accumulated Payoff: 1.9317848410757945
Batch: 121.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0251833740831295
----------------------------
Batch: 122.0 LinUCB Normalized Accumulated Payoff: 1.932055326377093
Batch: 122.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0252365930599368
----------------------------
Batch: 123.0 LinUCB Normalized Accumulated Payoff: 1.9373342974210652
Batch: 123.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0306097854904794
----------------------------
Batch: 124.0 LinUCB Normalized Accumulated Payoff: 1.930088284418993
Batch: 124.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0238606537819135
----------------------------
Batch: 126.0 LinUCB Normalized Accumulated Payoff: 1.9299318128379968
Batch: 126.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0244533270632497
----------------------------
Batch: 127.0 LinUCB Normalized Accumulated Payoff: 1.9251399253731343
Batch: 127.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.019356343283582
----------------------------
Batch: 131.0 LinUCB Normalized Accumulated Payoff: 1.9254306436990027
Batch: 131.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0176790571169536
----------------------------
Batch: 132.0 LinUCB Normalized Accumulated Payoff: 1.922679253764891
Batch: 132.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.013710946280063
----------------------------
Batch: 133.0 LinUCB Normalized Accumulated Payoff: 1.9212229413077437
Batch: 133.0 Basline_FactorUCB Normalized Accumulated Payoff: 2.0129435393885293
----------------------------
Traceback (most recent call last):
  File "D:\IIR\Server\PycharmProject\HyperBandit\RUN.py", line 187, in <module>
    alg.updateParameters(item_info_picked, reward, user_info, Theta)
  File "D:\IIR\Server\PycharmProject\HyperBandit\Baseline\factorUCB.py", line 206, in updateParameters
    self.USERS.updateParameters(articles, clicks, userID)
  File "D:\IIR\Server\PycharmProject\HyperBandit\Baseline\factorUCB.py", line 94, in updateParameters
    self.AInv = np.linalg.inv(self.A)
  File "<__array_function__ internals>", line 6, in inv
  File "D:\SoftwareFamily\Anaconda\envs\nlplab\lib\site-packages\numpy\linalg\linalg.py", line 546, in inv
    ainv = _umath_linalg.inv(a, signature=signature, extobj=extobj)
KeyboardInterrupt

进程已结束,退出代码1



"""

"""
D:\SoftwareFamily\Anaconda\envs\nlplab\python.exe D:\IIR\Server\PycharmProject\HyperBandit\RUN.py 
W:  [[0.0806435696861796, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.003539930801555768, 0.05290078314441142, 0.0, 0.0, 0.007145864239477776, 0.0, 0.0, 0.0, 0.0, 0.09198998208821133, 0.0, 0.0, 0.0, 0.0, 0.05363421195517386, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05238650803360783, 0.0, 0.0, 0.0, 0.012922870109231332, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09474116370077339, 0.0], [0.0, 0.21632077701744676, -0.0077209692643204495, 0.1093527266682547, 0.09079060829762073, 0.058324438266799, 0.0, 0.06304605422089501, 0.4678182287638181, 0.08382953011736981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07012247845078307, 0.21567210772346615, 0.08991735791814136, 0.0, 0.06770810883360022, 0.0, 0.0, 0.12235413016236589, 0.08731740954925543, 0.08494207893749674, 0.05125722905874721, 0.12186470493322092, 0.11228517002145937, 0.15765839692205072, 0.0, 0.0, 0.0, 0.0, 0.0, -0.009052659530958997, 0.0, 0.0788244919660337], [0.08558842262329837, -0.007114257600810782, 0.30629732745460203, 0.08295502438065834, 0.033616258015958855, 0.06430601367816043, 0.0, 0.07284856329384196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05130997474475957, 0.004612239171182217, 0.0, 0.0, 0.05799399937164132, 0.0, 0.0, 0.03633805198292215, 0.0, 0.08444979440204657, 0.09871718993229314, -0.002409014391602048, 0.047053472367530616, 0.003912344324664551, 0.0, 0.022324686855408674, 0.3764782974145802, 0.08110849803309132, 0.0, 0.2946254006104578, 0.0, 0.08466391263253734], [0.0, 0.0, 0.0, 0.08137825409593583, 0.0, 0.0, 0.014301708072753522, 0.0, -3.309913462475578e-05, 0.05413071094908193, 0.0, 0.003816153119313611, 0.0, 0.0408081828238396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03904341018658643, 0.0, 0.0, 0.0, -0.005275225896454161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0847635880709486, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004946425128016766, 0.0, 0.0, 0.0, 0.01842323538056873, 0.007358382157554758, 0.0, 0.0, 0.0, 0.0, 0.011754982149220047, 0.0, 0.017720046676567024, 0.034262774374026916, 0.01147243023238106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.012248322357956021, 0.0, 0.008605487935625699, 0.0, 0.004453220473280043, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.07659948342929117, 0.0, 0.0, -0.012874167779726138, 0.052698109503120406, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.013089596944159572, 0.04865240219918862, 0.0, 0.0, 0.0, 0.0, 0.008760860060265155, 0.015459458223006572, 0.0, 0.0, 0.0, 0.038050541097143106, 0.0, 0.0, 0.0, -0.008807793432212961, 0.0, 0.0, 0.051522679540597416, 0.0, 0.0, 0.015113169953855792, 0.0], [0.048747108481629844, 0.0, 0.0, 0.0, 0.032770312442594186, 0.05044683883328813, 0.31839144487915993, 0.06392756241882953, 0.0, 0.0, 0.0, 0.0, 0.034888147948218116, 0.10387930097830915, 0.05374731344243489, 0.0, 0.4595399582013698, 0.0, 0.05528682481707772, 0.0, 0.05588991311017409, 0.06577350157671273, 0.0, 0.030124785668595166, 0.0, 0.0, 0.0, 0.0, 0.06328894918218389, -0.007522103630268305, 0.03657566360156724, 0.0, 0.0, 0.0, 0.0, 0.11669316073487901, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.04356423877360527, 0.0, 0.0, 0.07359930763483918, 0.040582702167944296, 0.09520381750006995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06950241219213261, 0.0, 0.0, 0.0, 0.0, 0.05610153425271264, 0.0, 0.0, 0.0, 0.042409197832139436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0240457398441957, 0.0, 0.0, 0.0], [0.0, 0.24751992445532117, 0.0, 0.0, 0.04931037574526254, 0.0, 0.0, 0.0, 0.35622152587150896, 0.0513324938500082, 0.0, 0.0, 0.0, 0.0, 0.06703059578765541, 0.0, 0.0, 0.03919815299770815, 0.0, 0.0, 0.08127552157684657, 0.0, 0.06666412045762703, 0.0, 0.0, 0.0, 0.07901925055860039, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02989811384831817, 0.0, 0.10065342764469043, 0.0, 0.0, 0.0, 0.0], [0.05352925223950262, 0.0, 0.009236771223715884, 0.049121897946590165, 0.0, 0.053586758257722474, 0.02249578727507837, 0.06472845429466591, 0.0, 0.07864328499374475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005953218329591423, 0.025020954816446787, 0.001448209256749934, 0.0, 0.0, 0.0, 0.045095044958877326, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012287974153902087, 0.05665858372960087, 0.016811527070986224, 0.0032476588873620845, 0.0, 0.0], [0.0, 0.0, 0.0, 0.04802162579248517, 0.07233553738023496, 0.08099897335717361, 0.0, 0.0781924684986535, 0.0, 0.0, 0.31810623287958883, 0.03568726338324223, 0.0, 0.0, 0.05835010126400601, 0.0, 0.02319758930137136, 0.020700942628213922, 0.0, 0.0, 0.08599222526430093, 0.0, 0.0, 0.0, 0.0, 0.0077945140029775754, 0.0, 0.0, 0.0, 0.0, 0.08331360990247862, -0.0002581309650715688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2537608347563719, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.024500807896982547, 0.27814585201639286, 0.0, 0.0, 0.0, -0.004648746572356705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015650278302564335, 0.01581836102628305, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4951987857791844, 0.0, 0.0, 0.0, 0.054841895935337996, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.021341975456011495, 0.07316409875034699, 0.03673374341642718, 0.06346777175421621, 0.0, 0.0, 0.0, 0.0, 0.18450916795957045, 0.038070687853116694, 0.0, 0.0, 0.0, 0.0, 0.07672136635045501, 0.0, 0.0, 0.1789266864892314, 0.0, 0.07015182967251012, 0.01998584867952102, 0.0, 0.0, 0.0, 0.0, 0.1630143025778704, 0.0, 0.0, 0.0, 0.0, 0.008579662285950022, 0.0, 0.0, 0.0, 0.059590250334992684, 0.0], [0.0, -0.007399173134295426, 0.008041930829028704, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006946905777163669, 0.0, -0.0034135774524599607, 0.0760863533119122, 0.0, 0.0884391040187378, 0.06029202740523256, 0.0, 0.0, 0.019086030276691163, 0.0, -0.02314341587046523, 0.0, 0.0, 0.0, 0.03205394966303262, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0049035574050949006, 0.0, -0.013609013238464683, 0.0, 0.0, 0.0, 0.0, 0.01420746988973706, -0.00899335622671921, 0.0, 0.0], [0.0, -0.011021163116661715, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025530255753895954, 0.09406215335713787, 0.07619046601217143, 0.0, 0.01100938490370269, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.053394779756670285, 0.005006110955341321, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45072395764520823, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25027823539158867, 0.0, 0.023250049843865545, 0.0, 0.05757464132155655, 0.0, 0.0, 0.0, 0.0, 0.005190695218462829, 0.0, 0.0, 0.06157219142386973, 0.0, 0.0, 0.005912449865482711, 0.053274453207875244], [0.045234924925974224, 0.0, 0.0, 0.043164868975694756, 0.0, 0.0, 0.4473213028277142, 0.0, 0.0, 0.0, 0.01804397054909448, 0.0, 0.0, 0.042654487735341626, 0.0, -0.0013233372439168912, 0.31813664027601884, 0.0, 0.0, 0.0, 0.0, 0.026923561761954703, 0.0, 0.02339411467808647, 0.005738484633198154, 0.04498614659125222, 0.0, 0.05474954106887094, 0.0, 0.0, 0.0, 0.0, 0.0, 0.026746281970545686, 0.0, 0.054039214051038516, 0.0, 0.00205377025006485, 0.0, 0.05459735981822181], [0.0, -0.0004975203929792899, 0.0, 0.04672555231037885, 0.0, 0.0, 0.0, 0.0, 0.02605373955346767, 0.0, 0.009645293817038762, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1684397961223658, 0.058278623632060045, 0.0, 0.0, 0.0, 0.0, 0.016094673842381738, 0.042449311001528524, 0.0, 0.053769207030984215, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01306837236233131, 0.0, 0.0, 0.0, 0.0, -0.00945696967496565, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01219194623325461, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08169539520681535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05771934125922069, 0.01537887419252308, 0.0, 0.0, 0.0], [0.12816513422405953, 0.25760942882161286, 0.005978907424709121, 0.11679847058175362, 0.13248497172469406, 0.11868947635271718, 0.0, 0.08809399781649262, 0.0, 0.1228689101721676, 0.0, 0.0, 0.0, 0.10024032857308943, 0.10300640419504886, 0.0, 0.0, 0.0, 0.1362283605088465, 0.2800158612236948, 0.05169334986343683, 0.0, 0.06365899733011911, 0.0, 0.0, 0.12969371807233185, 0.07095476116226875, 0.09548521466408093, 0.12252572367593707, 0.2822344742971212, 0.11641964418936739, 0.36860027779446175, -0.006201949686459283, 0.0, 0.0, 0.04351490867656689, 0.0, 0.0, 0.0, 0.102528325041392], [0.0, 0.0, 0.007638527632430743, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.040562176573078416, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.016614679674382887, 0.07683229261748295, 0.0, 0.04193265928609906, 0.0, 0.0, 0.0, 0.04495202273952543, 0.0, 0.0, 0.0, 0.0545166705527227, 0.0, 0.022814322405173896, 0.0, 0.005900611283492626, 0.0, 0.0, 0.0, 0.02036865143154419, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.054073450135826616, 0.07437766463659792, 0.06544912851551205, 0.0, 0.0, 0.0, 0.0, 0.1921672705099158, 0.0, 0.0, 0.0, 0.03127719423637901, 0.0, 0.0, 0.0, 0.0, 0.16965954254315035, 0.0, 0.008133463510865958, 0.0, 0.0, 0.0, 0.0, 0.0, 0.037604406973709925, 0.0, 0.0, 0.0, 0.09656682253309883, 0.009714974736147297, 0.0, 0.36942002301156734, 0.0, 0.0, 0.06132194415516045], [0.06010250864199771, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.046651092935049265, 0.0, 0.0, 0.010960752168942158, 0.0, 0.0, 0.0, 0.0, 0.05082247738861471, 0.018644891389403465, 0.07819804791293619, 0.0, 0.0, 0.0, 0.0, 0.05599326288874423, 0.0, 0.0005967754728333389, 0.0, -0.008229663586274171, -0.011505435831365901, 0.0, 0.00984670726105189, 0.0, 0.0, 0.0, 0.07773556987190972, 0.05754580659515073], [0.06764185918845929, 0.0, 0.0, 0.04065221115298371, 0.0, 0.0, 0.05695046328371054, 0.0810531909959326, 0.0, 0.0, 0.0, 0.034436670491763466, 0.12595778808781816, 0.0, 0.06466431671949745, 0.49684540218538636, 0.04543427574221533, 0.05218228464007781, 0.039530384069785526, 0.0, 0.1055585399523575, 0.013597442886581327, 0.118769407697983, 0.2137389845171292, 0.45761521943321337, 0.0, 0.0, 0.06666601349045873, 0.0, -0.00026784071871963896, 0.05039671699794289, 0.0, 0.0, 0.0, 0.0, 0.046039327338238324, 0.0, 0.0, 0.0072446715548232085, 0.05768252806470051], [0.04323654237062384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.023694500421134065, 0.02442843250688135, 0.0, 0.0, -0.002559837722650817, 0.00758684481349851, 0.09369114309458465, 0.0, 0.0, 0.07085982763078925, 0.0, 0.09380595222453367, 0.31152108206097723, 0.31736781157956084, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04985127419244689, 0.0, 0.0, 0.031326141897884754, -0.0003556294799337944, 0.0, 0.0, 0.0, 0.12669904193999798, 0.0], [0.0, 0.036108353994489466, 0.01163839081211313, 0.06398267045756079, 0.0, 0.0, 0.0, 0.0, 0.0018342613869153836, 0.0, 0.003047656048979873, 0.0, 0.0, 0.056754482451758195, 0.044148947377112814, 0.01194628139437246, 0.022613430477068507, 0.0, 0.0, 0.03204352065077363, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16973726762523703, 0.0, 0.05729584567066393, 0.0, 0.07848066017266223, 0.0, 0.021081436866155337, 0.0, 0.034853783791329876, 0.0, 0.05584379870742076, 0.0, -0.008305498606614812, 0.0, 0.05786724430877036], [0.0, 0.0, 0.00827136370217271, 0.0, 0.0, 0.0, -0.004960883378360977, 0.0, 0.0, 0.0, -0.004474980742359003, 0.017755226952352737, 0.004948580766737581, 0.0, 0.0, 0.003934604762607718, 0.017311338472638896, 0.0, 0.0, -0.023545241375429245, 0.04641021144813091, 0.0, 0.0, 0.002377695955436445, 0.0038204540163933372, 0.02429459784574089, 0.07918839378942553, 0.05747452031057405, 0.0, 0.0, 0.05564269247301895, 0.0, 0.0, 0.03814626584695899, -0.005373547199274433, 0.0, 0.04242729505479266, 0.0, 0.06928129463521097, 0.05973108435734681], [0.0, 0.0, 0.0, 0.0, 0.1097516748494761, 0.06703012399814638, 0.07479019796607839, 0.0, 0.07242981338685701, 0.0697711510252293, 0.0, 0.0, 0.0, 0.1613394711207802, 0.07559188610750524, 0.0, 0.0, 0.0, 0.06915055400072369, 0.0, 0.0, 0.0, 0.10556934904208803, 0.0, 0.0, 0.0, 0.12720861983529994, 0.07469394882514771, 0.12103733952420839, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.047944397115607156, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.241824396760569, 0.009818270602776295, 0.0, 0.0, 0.0, 0.0, 0.014889811428668069, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007097984536663548, 0.009852058923389927, 0.0, 0.0, 0.08415174087486156, 0.0, 0.04858432770564164, -0.011738477211626578, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0936346380327344, 0.0, 0.0], [0.0878706305347936, 0.14382927230555206, -0.0030856806242166197, 0.10493427468048015, 0.09562575196206619, 0.10465734358439421, -0.01471985572719169, 0.07454933119685817, 0.0, 0.08872175311454361, 0.0, 0.0, 0.30297212890092695, 0.0, 0.0924161315856039, 0.0, 0.0, 0.0, 0.12106106195112731, 0.2788763146286721, 0.10685940535159576, 0.06507458547004895, 0.0, 0.0, 0.0, 0.3138650310575207, 0.06748270028713636, 0.0844649249641704, 0.11568060479513267, 0.18754943478412076, 0.0, 0.2044631828862502, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09804507770815492], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015057448779006444, 0.011006045761836853, 0.0, 0.0, 0.0, 0.0, 0.09589074978095445, 0.0, 0.0, 0.057141007770476974, 0.03943607260479, 0.0, 0.0, 0.0, 0.0, 0.05648880653487299, 0.0, 0.051445053886967074, 0.0, 0.08062853311053274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0679662319332262, 0.0, 0.02009554218691044, 0.0], [0.09034395433918103, 0.13962868191736208, 0.0037604242013826498, 0.06476329463392336, 0.06875282455479896, 0.059578348312255565, 0.0, 0.05481779722645733, 0.0, 0.09330293401161419, -0.00030289038687985605, 0.0, 0.0, 0.043705029595894854, 0.048986697678722116, 0.0, 0.0, 0.0, 0.07954655962090482, 0.27330364141740676, 0.06967711570292122, 0.0, 0.0, 0.0, 0.0, 0.06326575316147236, 0.04892839033915571, 0.057593149519646995, 0.0, 0.15342753488930022, 0.09545264931937072, 0.28548047569781887, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012979573654732882, 0.0, 0.06432690858395233], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.019337965558767833, 0.045616064083496374, 0.0, 0.43324074847413924, 0.0, 0.061561121184709085, 0.0, 0.004097079170823689, 0.0, 0.0, 0.0, -0.00235390807744109, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.060972743748655155, 0.0, 0.061808489399704195, 0.0, 0.06760268055839204, 0.0, 0.47626725614647203, 0.0, 0.0, 0.0, 0.04294649146546606, 0.0, 0.0, 0.0], [0.0, 0.00333293323825279, 0.00884109840735915, 0.0, 0.0, 0.0, -0.005739717552539324, 0.0, 0.020620034115171878, 0.0, -0.0084013393943844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01662419517306757, 0.01321960388539299, 0.0, 0.0, 0.0, 0.05166659932279537, 0.0, 0.0, 0.014727198592410605, 0.043096250498542696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15409397562087257, 0.010265654900480098, 0.0, 0.12540586351230287, -0.0036617681397584935, 0.2997946971927015, 0.0], [0.04958210780638345, 0.0, 0.26414149981276025, 0.0, 0.0, 0.0, -0.0068099245417097805, 0.054636482557838074, 0.0, 0.0, 0.0, 0.0, 0.008734383900346804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009208723993855658, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.008496293607626238, 0.01818706024704238, 0.29948190423108395, 0.07036801170899795, 0.0, 0.14233506483094022, 0.0, 0.0], [0.0, -0.0076449211788817304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.007132940595524474, 0.00786898603197662, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01032371254018818, 0.0, 0.0, 0.0, 0.0, 0.011626676581963513, 0.0, 0.0, 0.0, 0.0, -0.00966977795515396, 0.0, -0.011521468575152082, 0.0, 0.02948838800217947, 0.0, 0.08014727989462662, 0.013684813506719056, 0.0, 0.04611791585567385, 0.0], [0.0, 0.0, 0.0, 0.04594164007217941, 0.0, 0.0578587620771652, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05944610128797329, 0.0, 0.0, 0.07335439428695173, 0.002240033520294996, 0.0, 0.0, 0.0, 0.0013379460083746753, 0.0, 0.3042383648533788, 0.05046218866742952, 0.0, 0.0, 0.0, 0.0, 0.06841558243855786, 0.0, 0.0, 0.0, -0.013117072153191635, 0.053209463515881406, 0.1930320639530912, 0.0, 0.0, 0.20841055410986564, 0.0, 0.0, 0.0], [0.11136958782230973, -0.010672336326408361, 0.3769604083882627, 0.10220748825112116, 0.07739738464757093, 0.08068589096671328, 0.0, 0.10158988957496816, 0.0, 0.0, 0.3963654243860995, 0.0, 0.0, 0.03844610630085577, 0.10821570080566592, 0.0, 0.0, 0.0, 0.03946891763422703, -0.009880673285663487, 0.05107075440473089, 0.0, 0.05439167095969155, 0.0, 0.0, 0.0, 0.07029118674095695, 0.10020148149798451, 0.0, 0.0, 0.10167689500752808, 0.017277711238379234, 0.0, 0.0, 0.25956179152155306, 0.12411957725716143, 0.0, 0.23683331115635298, 0.0, 0.09425269471059093], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.031060840729325072, 0.04143826323061028, 0.0, 0.003407356981128648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05403041732223056, 0.0, 0.0, 0.0, 0.03610172258464061, 0.0, 0.0, 0.0, 0.0, 0.27194163567659924, 0.0, 0.0, 0.0, 0.0, 0.15730558147612347, 0.0], [0.0, 0.0, 0.0, 0.0, 0.08749449807915705, 0.0, -0.013131931157718203, 0.0, 0.04327795225182747, 0.07041828096206425, 0.0, 0.0, 0.0, 0.0886012807758195, 0.07400501733239184, 0.0, 0.0, 0.32156109609275774, 0.07294709681324571, 0.0, 0.0, 0.0, 0.09110975200836485, 0.0, 0.0, 0.0, 0.11101730985172371, 0.0, 0.06450549776798678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07533816885011275]]
GPU is available.
step: 1.0 LinUCB_ItemBased 4.239130434782608
step: 1.0 LinUCB_UserBased 1.8478260869565217
step: 1.0 HybridLinUCB 2.0434782608695654
step: 1.0 FactorUCB w/o W 1.2608695652173914
step: 1.0 Hyper_FactorUCB w/o W 1.108695652173913
step: 1.0 FactorUCB 0.6086956521739131
step: 1.0 DLinUCB 1.8043478260869565
step: 1.0 ColinUCB 1.608695652173913
----------------------------
step: 2.0 LinUCB_ItemBased 4.760869565217392
step: 2.0 LinUCB_UserBased 1.9456521739130435
step: 2.0 HybridLinUCB 2.0543478260869565
step: 2.0 FactorUCB w/o W 1.315217391304348
step: 2.0 Hyper_FactorUCB w/o W 1.2173913043478262
step: 2.0 FactorUCB 0.6413043478260869
step: 2.0 DLinUCB 1.9456521739130435
step: 2.0 ColinUCB 1.815217391304348
----------------------------
step: 3.0 LinUCB_ItemBased 5.093023255813954
step: 3.0 LinUCB_UserBased 1.8914728682170543
step: 3.0 HybridLinUCB 2.007751937984496
step: 3.0 FactorUCB w/o W 1.449612403100775
step: 3.0 Hyper_FactorUCB w/o W 1.3333333333333333
step: 3.0 FactorUCB 0.6511627906976745
step: 3.0 DLinUCB 1.8992248062015504
step: 3.0 ColinUCB 1.821705426356589
----------------------------
step: 4.0 LinUCB_ItemBased 5.634285714285714
step: 4.0 LinUCB_UserBased 1.8171428571428572
step: 4.0 HybridLinUCB 1.8914285714285715
step: 4.0 FactorUCB w/o W 1.6228571428571428
step: 4.0 Hyper_FactorUCB w/o W 1.3885714285714286
step: 4.0 FactorUCB 0.5942857142857143
step: 4.0 DLinUCB 1.8857142857142857
step: 4.0 ColinUCB 1.8057142857142856
----------------------------
step: 5.0 LinUCB_ItemBased 5.748815165876778
step: 5.0 LinUCB_UserBased 1.7630331753554502
step: 5.0 HybridLinUCB 1.8293838862559242
step: 5.0 FactorUCB w/o W 1.838862559241706
step: 5.0 Hyper_FactorUCB w/o W 1.7061611374407584
step: 5.0 FactorUCB 0.6066350710900474
step: 5.0 DLinUCB 1.8578199052132702
step: 5.0 ColinUCB 1.7867298578199051
----------------------------
step: 6.0 LinUCB_ItemBased 6.465020576131687
step: 6.0 LinUCB_UserBased 1.8847736625514404
step: 6.0 HybridLinUCB 1.97119341563786
step: 6.0 FactorUCB w/o W 2.119341563786008
step: 6.0 Hyper_FactorUCB w/o W 2.4156378600823047
step: 6.0 FactorUCB 0.6337448559670782
step: 6.0 DLinUCB 1.9588477366255144
step: 6.0 ColinUCB 1.9094650205761317
----------------------------
step: 7.0 LinUCB_ItemBased 6.303754266211604
step: 7.0 LinUCB_UserBased 1.764505119453925
step: 7.0 HybridLinUCB 1.8293515358361774
step: 7.0 FactorUCB w/o W 2.204778156996587
step: 7.0 Hyper_FactorUCB w/o W 2.689419795221843
step: 7.0 FactorUCB 0.5699658703071673
step: 7.0 DLinUCB 1.8771331058020477
step: 7.0 ColinUCB 1.8088737201365188
----------------------------
step: 8.0 LinUCB_ItemBased 6.306306306306307
step: 8.0 LinUCB_UserBased 1.7027027027027026
step: 8.0 HybridLinUCB 1.7657657657657657
step: 8.0 FactorUCB w/o W 2.324324324324324
step: 8.0 Hyper_FactorUCB w/o W 2.951951951951952
step: 8.0 FactorUCB 0.5585585585585585
step: 8.0 DLinUCB 1.8468468468468469
step: 8.0 ColinUCB 1.7387387387387387
----------------------------
step: 9.0 LinUCB_ItemBased 6.4032697547683926
step: 9.0 LinUCB_UserBased 1.6866485013623977
step: 9.0 HybridLinUCB 1.7438692098092643
step: 9.0 FactorUCB w/o W 2.455040871934605
step: 9.0 Hyper_FactorUCB w/o W 3.2370572207084467
step: 9.0 FactorUCB 0.5449591280653951
step: 9.0 DLinUCB 1.8337874659400546
step: 9.0 ColinUCB 1.7275204359673024
----------------------------
step: 10.0 LinUCB_ItemBased 6.401937046004843
step: 10.0 LinUCB_UserBased 1.639225181598063
step: 10.0 HybridLinUCB 1.6731234866828086
step: 10.0 FactorUCB w/o W 2.523002421307506
step: 10.0 Hyper_FactorUCB w/o W 3.4285714285714284
step: 10.0 FactorUCB 0.5157384987893463
step: 10.0 DLinUCB 1.774818401937046
step: 10.0 ColinUCB 1.6440677966101696
----------------------------
D:\IIR\Server\PycharmProject\HyperBandit\hypernet.py:130: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at  C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\torch\csrc\utils\tensor_new.cpp:210.)
  train_input_tensor = torch.Tensor(train_input)
Epoch [5/50], Loss: 3.1491
Epoch [10/50], Loss: 3.1419
Epoch [15/50], Loss: 3.1401
Epoch [20/50], Loss: 3.1420
Epoch [25/50], Loss: 3.1323
Epoch [30/50], Loss: 3.1288
Epoch [35/50], Loss: 3.1309
Epoch [40/50], Loss: 3.1239
Epoch [45/50], Loss: 3.1236
Epoch [50/50], Loss: 3.1201
step: 11.0 LinUCB_ItemBased 6.657718120805369
step: 11.0 LinUCB_UserBased 1.6532438478747205
step: 11.0 HybridLinUCB 1.6823266219239374
step: 11.0 FactorUCB w/o W 2.6711409395973154
step: 11.0 Hyper_FactorUCB w/o W 3.610738255033557
step: 11.0 FactorUCB 0.5257270693512305
step: 11.0 DLinUCB 1.8120805369127517
step: 11.0 ColinUCB 1.686800894854586
----------------------------
step: 12.0 LinUCB_ItemBased 6.791411042944786
step: 12.0 LinUCB_UserBased 1.6523517382413089
step: 12.0 HybridLinUCB 1.6625766871165644
step: 12.0 FactorUCB w/o W 2.7075664621676894
step: 12.0 Hyper_FactorUCB w/o W 3.7157464212678937
step: 12.0 FactorUCB 0.5296523517382413
step: 12.0 DLinUCB 1.803680981595092
step: 12.0 ColinUCB 1.687116564417178
----------------------------
step: 13.0 LinUCB_ItemBased 6.871028037383177
step: 13.0 LinUCB_UserBased 1.6355140186915889
step: 13.0 HybridLinUCB 1.6542056074766356
step: 13.0 FactorUCB w/o W 2.7570093457943927
step: 13.0 Hyper_FactorUCB w/o W 3.6710280373831776
step: 13.0 FactorUCB 0.5289719626168224
step: 13.0 DLinUCB 1.7738317757009345
step: 13.0 ColinUCB 1.674766355140187
----------------------------
step: 14.0 LinUCB_ItemBased 6.83419689119171
step: 14.0 LinUCB_UserBased 1.6632124352331605
step: 14.0 HybridLinUCB 1.677029360967185
step: 14.0 FactorUCB w/o W 2.8583765112262522
step: 14.0 Hyper_FactorUCB w/o W 3.6666666666666665
step: 14.0 FactorUCB 0.5233160621761658
step: 14.0 DLinUCB 1.7841105354058722
step: 14.0 ColinUCB 1.690846286701209
----------------------------
step: 15.0 LinUCB_ItemBased 6.886581469648562
step: 15.0 LinUCB_UserBased 1.6277955271565496
step: 15.0 HybridLinUCB 1.6501597444089458
step: 15.0 FactorUCB w/o W 2.8722044728434506
step: 15.0 Hyper_FactorUCB w/o W 3.5910543130990416
step: 15.0 FactorUCB 0.5063897763578274
step: 15.0 DLinUCB 1.7507987220447285
step: 15.0 ColinUCB 1.659744408945687
----------------------------
step: 16.0 LinUCB_ItemBased 7.033333333333333
step: 16.0 LinUCB_UserBased 1.6757575757575758
step: 16.0 HybridLinUCB 1.696969696969697
step: 16.0 FactorUCB w/o W 3.0303030303030303
step: 16.0 Hyper_FactorUCB w/o W 3.68030303030303
step: 16.0 FactorUCB 0.5242424242424243
step: 16.0 DLinUCB 1.793939393939394
step: 16.0 ColinUCB 1.6939393939393939
----------------------------
step: 17.0 LinUCB_ItemBased 7.224460431654676
step: 17.0 LinUCB_UserBased 1.6877697841726618
step: 17.0 HybridLinUCB 1.7093525179856115
step: 17.0 FactorUCB w/o W 3.129496402877698
step: 17.0 Hyper_FactorUCB w/o W 3.707913669064748
step: 17.0 FactorUCB 0.5280575539568345
step: 17.0 DLinUCB 1.7928057553956835
step: 17.0 ColinUCB 1.7007194244604316
----------------------------
step: 18.0 LinUCB_ItemBased 7.212365591397849
step: 18.0 LinUCB_UserBased 1.6814516129032258
step: 18.0 HybridLinUCB 1.6935483870967742
step: 18.0 FactorUCB w/o W 3.189516129032258
step: 18.0 Hyper_FactorUCB w/o W 3.7903225806451615
step: 18.0 FactorUCB 0.5309139784946236
step: 18.0 DLinUCB 1.7741935483870968
step: 18.0 ColinUCB 1.6774193548387097
----------------------------
step: 19.0 LinUCB_ItemBased 7.405161290322581
step: 19.0 LinUCB_UserBased 1.687741935483871
step: 19.0 HybridLinUCB 1.7058064516129032
step: 19.0 FactorUCB w/o W 3.2580645161290325
step: 19.0 Hyper_FactorUCB w/o W 3.9341935483870967
step: 19.0 FactorUCB 0.5316129032258065
step: 19.0 DLinUCB 1.7754838709677419
step: 19.0 ColinUCB 1.689032258064516
----------------------------
step: 20.0 LinUCB_ItemBased 7.388004895960832
step: 20.0 LinUCB_UserBased 1.675642594859241
step: 20.0 HybridLinUCB 1.696450428396573
step: 20.0 FactorUCB w/o W 3.3206854345165238
step: 20.0 Hyper_FactorUCB w/o W 4.048959608323133
step: 20.0 FactorUCB 0.5152998776009792
step: 20.0 DLinUCB 1.7711138310893513
step: 20.0 ColinUCB 1.6793145654834762
----------------------------
Epoch [5/50], Loss: 3.1252
Epoch [10/50], Loss: 3.1151
Epoch [15/50], Loss: 3.1104
Epoch [20/50], Loss: 3.1072
Epoch [25/50], Loss: 3.1064
Epoch [30/50], Loss: 3.1028
Epoch [35/50], Loss: 3.1014
Epoch [40/50], Loss: 3.0996
Epoch [45/50], Loss: 3.0992
Epoch [50/50], Loss: 3.0974
step: 21.0 LinUCB_ItemBased 7.3329452852153665
step: 21.0 LinUCB_UserBased 1.6542491268917345
step: 21.0 HybridLinUCB 1.670547147846333
step: 21.0 FactorUCB w/o W 3.332945285215367
step: 21.0 Hyper_FactorUCB w/o W 4.051222351571595
step: 21.0 FactorUCB 0.5133876600698487
step: 21.0 DLinUCB 1.7566938300349244
step: 21.0 ColinUCB 1.6472642607683352
----------------------------
step: 22.0 LinUCB_ItemBased 7.426008968609866
step: 22.0 LinUCB_UserBased 1.6771300448430493
step: 22.0 HybridLinUCB 1.687219730941704
step: 22.0 FactorUCB w/o W 3.431614349775785
step: 22.0 Hyper_FactorUCB w/o W 4.174887892376682
step: 22.0 FactorUCB 0.5190582959641256
step: 22.0 DLinUCB 1.7937219730941705
step: 22.0 ColinUCB 1.6659192825112108
----------------------------
step: 23.0 LinUCB_ItemBased 7.471030042918455
step: 23.0 LinUCB_UserBased 1.6909871244635193
step: 23.0 HybridLinUCB 1.7027896995708154
step: 23.0 FactorUCB w/o W 3.495708154506438
step: 23.0 Hyper_FactorUCB w/o W 4.25
step: 23.0 FactorUCB 0.509656652360515
step: 23.0 DLinUCB 1.8487124463519313
step: 23.0 ColinUCB 1.7317596566523605
----------------------------
step: 24.0 LinUCB_ItemBased 7.659019812304484
step: 24.0 LinUCB_UserBased 1.721584984358707
step: 24.0 HybridLinUCB 1.72992700729927
step: 24.0 FactorUCB w/o W 3.5818561001042752
step: 24.0 Hyper_FactorUCB w/o W 4.446298227320125
step: 24.0 FactorUCB 0.5224191866527633
step: 24.0 DLinUCB 1.8800834202294057
step: 24.0 ColinUCB 1.7570385818561
----------------------------
step: 25.0 LinUCB_ItemBased 7.712437810945274
step: 25.0 LinUCB_UserBased 1.727363184079602
step: 25.0 HybridLinUCB 1.7373134328358208
step: 25.0 FactorUCB w/o W 3.672636815920398
step: 25.0 Hyper_FactorUCB w/o W 4.534328358208955
step: 25.0 FactorUCB 0.5333333333333333
step: 25.0 DLinUCB 1.8935323383084577
step: 25.0 ColinUCB 1.7661691542288558
----------------------------
step: 26.0 LinUCB_ItemBased 7.736190476190476
step: 26.0 LinUCB_UserBased 1.7114285714285715
step: 26.0 HybridLinUCB 1.7228571428571429
step: 26.0 FactorUCB w/o W 3.677142857142857
step: 26.0 Hyper_FactorUCB w/o W 4.571428571428571
step: 26.0 FactorUCB 0.5314285714285715
step: 26.0 DLinUCB 1.8752380952380951
step: 26.0 ColinUCB 1.7523809523809524
----------------------------
step: 27.0 LinUCB_ItemBased 7.814102564102564
step: 27.0 LinUCB_UserBased 1.7261904761904763
step: 27.0 HybridLinUCB 1.7316849816849818
step: 27.0 FactorUCB w/o W 3.7142857142857144
step: 27.0 Hyper_FactorUCB w/o W 4.672161172161172
step: 27.0 FactorUCB 0.5439560439560439
step: 27.0 DLinUCB 1.8846153846153846
step: 27.0 ColinUCB 1.7619047619047619
----------------------------
step: 28.0 LinUCB_ItemBased 7.74430823117338
step: 28.0 LinUCB_UserBased 1.7075306479859895
step: 28.0 HybridLinUCB 1.7180385288966724
step: 28.0 FactorUCB w/o W 3.7075306479859895
step: 28.0 Hyper_FactorUCB w/o W 4.676882661996498
step: 28.0 FactorUCB 0.5394045534150613
step: 28.0 DLinUCB 1.8572679509632224
step: 28.0 ColinUCB 1.7425569176882663
----------------------------
step: 29.0 LinUCB_ItemBased 7.867061812023708
step: 29.0 LinUCB_UserBased 1.7095681625740897
step: 29.0 HybridLinUCB 1.7197290431837426
step: 29.0 FactorUCB w/o W 3.7595258255715494
step: 29.0 Hyper_FactorUCB w/o W 4.7739204064352245
step: 29.0 FactorUCB 0.5376799322607959
step: 29.0 DLinUCB 1.8526672311600338
step: 29.0 ColinUCB 1.7451312447078746
----------------------------
step: 30.0 LinUCB_ItemBased 7.961253091508656
step: 30.0 LinUCB_UserBased 1.7164056059356967
step: 30.0 HybridLinUCB 1.7254740313272878
step: 30.0 FactorUCB w/o W 3.794723825226711
step: 30.0 Hyper_FactorUCB w/o W 4.906018136850784
step: 30.0 FactorUCB 0.5383347073371806
step: 30.0 DLinUCB 1.8507831821929102
step: 30.0 ColinUCB 1.751030502885408
----------------------------
Epoch [5/50], Loss: 3.0991
Epoch [10/50], Loss: 3.0893
Epoch [15/50], Loss: 3.0847
Epoch [20/50], Loss: 3.0814
Epoch [25/50], Loss: 3.0786
Epoch [30/50], Loss: 3.0771
Epoch [35/50], Loss: 3.0757
Epoch [40/50], Loss: 3.0740
Epoch [45/50], Loss: 3.0716
Epoch [50/50], Loss: 3.0701
step: 31.0 LinUCB_ItemBased 7.952229299363057
step: 31.0 LinUCB_UserBased 1.714171974522293
step: 31.0 HybridLinUCB 1.7237261146496816
step: 31.0 FactorUCB w/o W 3.8136942675159236
step: 31.0 Hyper_FactorUCB w/o W 4.942675159235669
step: 31.0 FactorUCB 0.5326433121019108
step: 31.0 DLinUCB 1.84156050955414
step: 31.0 ColinUCB 1.7452229299363058
----------------------------
step: 32.0 LinUCB_ItemBased 7.9930286599535245
step: 32.0 LinUCB_UserBased 1.710302091402014
step: 32.0 HybridLinUCB 1.7234701781564679
step: 32.0 FactorUCB w/o W 3.859798605731991
step: 32.0 Hyper_FactorUCB w/o W 5.006196746707978
step: 32.0 FactorUCB 0.5391169635941131
step: 32.0 DLinUCB 1.8381099922540667
step: 32.0 ColinUCB 1.7397366382649109
----------------------------
step: 33.0 LinUCB_ItemBased 7.946188340807175
step: 33.0 LinUCB_UserBased 1.6793721973094171
step: 33.0 HybridLinUCB 1.6920777279521675
step: 33.0 FactorUCB w/o W 3.868460388639761
step: 33.0 Hyper_FactorUCB w/o W 5.011210762331839
step: 33.0 FactorUCB 0.5261584454409567
step: 33.0 DLinUCB 1.812406576980568
step: 33.0 ColinUCB 1.7055306427503736
----------------------------
step: 34.0 LinUCB_ItemBased 7.9231327048585936
step: 34.0 LinUCB_UserBased 1.675852066715011
step: 34.0 HybridLinUCB 1.6852791878172588
step: 34.0 FactorUCB w/o W 3.8883248730964466
step: 34.0 Hyper_FactorUCB w/o W 5.056562726613488
step: 34.0 FactorUCB 0.5358955765047135
step: 34.0 DLinUCB 1.8071065989847717
step: 34.0 ColinUCB 1.6976069615663525
----------------------------
step: 35.0 LinUCB_ItemBased 7.947703180212014
step: 35.0 LinUCB_UserBased 1.6791519434628974
step: 35.0 HybridLinUCB 1.6869257950530034
step: 35.0 FactorUCB w/o W 3.913780918727915
step: 35.0 Hyper_FactorUCB w/o W 5.107420494699647
step: 35.0 FactorUCB 0.5371024734982333
step: 35.0 DLinUCB 1.8014134275618374
step: 35.0 ColinUCB 1.693286219081272
----------------------------
step: 36.0 LinUCB_ItemBased 8.11634349030471
step: 36.0 LinUCB_UserBased 1.7029085872576177
step: 36.0 HybridLinUCB 1.709141274238227
step: 36.0 FactorUCB w/o W 3.994459833795014
step: 36.0 Hyper_FactorUCB w/o W 5.268698060941828
step: 36.0 FactorUCB 0.5443213296398892
step: 36.0 DLinUCB 1.8240997229916898
step: 36.0 ColinUCB 1.7146814404432134
----------------------------
step: 37.0 LinUCB_ItemBased 8.092131809011432
step: 37.0 LinUCB_UserBased 1.6960322797579017
step: 37.0 HybridLinUCB 1.70275722932078
step: 37.0 FactorUCB w/o W 4.010759919300606
step: 37.0 Hyper_FactorUCB w/o W 5.29119031607263
step: 37.0 FactorUCB 0.5447209145931405
step: 37.0 DLinUCB 1.8143913920645596
step: 37.0 ColinUCB 1.7041022192333557
----------------------------
step: 38.0 LinUCB_ItemBased 8.173342087984242
step: 38.0 LinUCB_UserBased 1.7005909389363099
step: 38.0 HybridLinUCB 1.7084701247537755
step: 38.0 FactorUCB w/o W 4.088640840446487
step: 38.0 Hyper_FactorUCB w/o W 5.395272488509521
step: 38.0 FactorUCB 0.5489166119500984
step: 38.0 DLinUCB 1.8161523309258043
step: 38.0 ColinUCB 1.7084701247537755
----------------------------
step: 39.0 LinUCB_ItemBased 8.133672819859962
step: 39.0 LinUCB_UserBased 1.6925525143220879
step: 39.0 HybridLinUCB 1.7001909611712285
step: 39.0 FactorUCB w/o W 4.081476766390834
step: 39.0 Hyper_FactorUCB w/o W 5.394653087205602
step: 39.0 FactorUCB 0.5436028007638447
step: 39.0 DLinUCB 1.807129217059198
step: 39.0 ColinUCB 1.6950986632718015
----------------------------
step: 40.0 LinUCB_ItemBased 8.11248454882571
step: 40.0 LinUCB_UserBased 1.6705809641532756
step: 40.0 HybridLinUCB 1.6773794808405438
step: 40.0 FactorUCB w/o W 4.1087762669962915
step: 40.0 Hyper_FactorUCB w/o W 5.4004944375772554
step: 40.0 FactorUCB 0.5333745364647713
step: 40.0 DLinUCB 1.7917181705809642
step: 40.0 ColinUCB 1.6718170580964153
----------------------------
Epoch [5/50], Loss: 3.1244
Epoch [10/50], Loss: 3.1193
Epoch [15/50], Loss: 3.1169
Epoch [20/50], Loss: 3.1152
Epoch [25/50], Loss: 3.1136
Epoch [30/50], Loss: 3.1123
Epoch [35/50], Loss: 3.1110
Epoch [40/50], Loss: 3.1098
Epoch [45/50], Loss: 3.1087
Epoch [50/50], Loss: 3.1077
step: 41.0 LinUCB_ItemBased 8.082629674306393
step: 41.0 LinUCB_UserBased 1.661037394451146
step: 41.0 HybridLinUCB 1.6688781664656211
step: 41.0 FactorUCB w/o W 4.121230398069963
step: 41.0 Hyper_FactorUCB w/o W 5.408926417370326
step: 41.0 FactorUCB 0.5379975874547648
step: 41.0 DLinUCB 1.787696019300362
step: 41.0 ColinUCB 1.6592279855247285
----------------------------
step: 42.0 LinUCB_ItemBased 8.064705882352941
step: 42.0 LinUCB_UserBased 1.6494117647058824
step: 42.0 HybridLinUCB 1.6552941176470588
step: 42.0 FactorUCB w/o W 4.145882352941176
step: 42.0 Hyper_FactorUCB w/o W 5.4282352941176475
step: 42.0 FactorUCB 0.5329411764705883
step: 42.0 DLinUCB 1.7823529411764707
step: 42.0 ColinUCB 1.65
----------------------------
step: 43.0 LinUCB_ItemBased 8.124066628374498
step: 43.0 LinUCB_UserBased 1.649052268811028
step: 43.0 HybridLinUCB 1.6547960941987363
step: 43.0 FactorUCB w/o W 4.190120620333142
step: 43.0 Hyper_FactorUCB w/o W 5.511200459506031
step: 43.0 FactorUCB 0.5301550832854681
step: 43.0 DLinUCB 1.7759908098793797
step: 43.0 ColinUCB 1.646754738655945
----------------------------
step: 44.0 LinUCB_ItemBased 8.158426966292135
step: 44.0 LinUCB_UserBased 1.6471910112359551
step: 44.0 HybridLinUCB 1.652808988764045
step: 44.0 FactorUCB w/o W 4.224719101123595
step: 44.0 Hyper_FactorUCB w/o W 5.580337078651685
step: 44.0 FactorUCB 0.5337078651685393
step: 44.0 DLinUCB 1.7685393258426967
step: 44.0 ColinUCB 1.646067415730337
----------------------------
step: 45.0 LinUCB_ItemBased 8.198019801980198
step: 45.0 LinUCB_UserBased 1.6457645764576458
step: 45.0 HybridLinUCB 1.6507150715071508
step: 45.0 FactorUCB w/o W 4.2502750275027505
step: 45.0 Hyper_FactorUCB w/o W 5.643564356435643
step: 45.0 FactorUCB 0.533003300330033
step: 45.0 DLinUCB 1.7645764576457645
step: 45.0 ColinUCB 1.6446644664466448
----------------------------
step: 46.0 LinUCB_ItemBased 8.245566899516389
step: 46.0 LinUCB_UserBased 1.6577109081139172
step: 46.0 HybridLinUCB 1.663084363245567
step: 46.0 FactorUCB w/o W 4.31058570660935
step: 46.0 Hyper_FactorUCB w/o W 5.718430951101558
step: 46.0 FactorUCB 0.5427189682966147
step: 46.0 DLinUCB 1.7748522299838796
step: 46.0 ColinUCB 1.6560988715744223
----------------------------
step: 47.0 LinUCB_ItemBased 8.288502109704641
step: 47.0 LinUCB_UserBased 1.6619198312236287
step: 47.0 HybridLinUCB 1.6671940928270041
step: 47.0 FactorUCB w/o W 4.345991561181434
step: 47.0 Hyper_FactorUCB w/o W 5.777426160337553
step: 47.0 FactorUCB 0.5453586497890295
step: 47.0 DLinUCB 1.7737341772151898
step: 47.0 ColinUCB 1.659282700421941
----------------------------
step: 48.0 LinUCB_ItemBased 8.291472868217054
step: 48.0 LinUCB_UserBased 1.6640826873385013
step: 48.0 HybridLinUCB 1.6666666666666667
step: 48.0 FactorUCB w/o W 4.365374677002584
step: 48.0 Hyper_FactorUCB w/o W 5.802067183462532
step: 48.0 FactorUCB 0.5467700258397933
step: 48.0 DLinUCB 1.779328165374677
step: 48.0 ColinUCB 1.6578811369509043
----------------------------
step: 49.0 LinUCB_ItemBased 8.367967562088191
step: 49.0 LinUCB_UserBased 1.6675114039533705
step: 49.0 HybridLinUCB 1.6725798276735935
step: 49.0 FactorUCB w/o W 4.409021794221997
step: 49.0 Hyper_FactorUCB w/o W 5.897110998479473
step: 49.0 FactorUCB 0.5484034465281298
step: 49.0 DLinUCB 1.789660415610745
step: 49.0 ColinUCB 1.666497719209326
----------------------------
step: 50.0 LinUCB_ItemBased 8.369781312127236
step: 50.0 LinUCB_UserBased 1.6729622266401591
step: 50.0 HybridLinUCB 1.6784294234592445
step: 50.0 FactorUCB w/o W 4.42544731610338
step: 50.0 Hyper_FactorUCB w/o W 5.92544731610338
step: 50.0 FactorUCB 0.5506958250497018
step: 50.0 DLinUCB 1.8001988071570576
step: 50.0 ColinUCB 1.66948310139165
----------------------------
Epoch [5/50], Loss: 3.0824
Epoch [10/50], Loss: 3.0756
Epoch [15/50], Loss: 3.0725
Epoch [20/50], Loss: 3.0701
Epoch [25/50], Loss: 3.0682
Epoch [30/50], Loss: 3.0666
Epoch [35/50], Loss: 3.0652
Epoch [40/50], Loss: 3.0639
Epoch [45/50], Loss: 3.0627
Epoch [50/50], Loss: 3.0617
step: 51.0 LinUCB_ItemBased 8.391516333495856
step: 51.0 LinUCB_UserBased 1.6684544124817162
step: 51.0 HybridLinUCB 1.6733300828863968
step: 51.0 FactorUCB w/o W 4.468064358849341
step: 51.0 Hyper_FactorUCB w/o W 5.955631399317406
step: 51.0 FactorUCB 0.5475377864456362
step: 51.0 DLinUCB 1.794734275962945
step: 51.0 ColinUCB 1.6640663091175036
----------------------------
step: 52.0 LinUCB_ItemBased 8.4066250600096
step: 52.0 LinUCB_UserBased 1.6663466154584734
step: 52.0 HybridLinUCB 1.671147383581373
step: 52.0 FactorUCB w/o W 4.499279884781565
step: 52.0 Hyper_FactorUCB w/o W 6.00384061449832
step: 52.0 FactorUCB 0.5453672587614018
step: 52.0 DLinUCB 1.7940470475276045
step: 52.0 ColinUCB 1.6620259241478637
----------------------------
step: 53.0 LinUCB_ItemBased 8.414427157001414
step: 53.0 LinUCB_UserBased 1.6633663366336633
step: 53.0 HybridLinUCB 1.667138142385667
step: 53.0 FactorUCB w/o W 4.52003771805752
step: 53.0 Hyper_FactorUCB w/o W 6.036775106082037
step: 53.0 FactorUCB 0.5464403583215465
step: 53.0 DLinUCB 1.7958510136727959
step: 53.0 ColinUCB 1.6572371522866571
----------------------------
step: 54.0 LinUCB_ItemBased 8.45754060324826
step: 54.0 LinUCB_UserBased 1.6696055684454756
step: 54.0 HybridLinUCB 1.671461716937355
step: 54.0 FactorUCB w/o W 4.555916473317866
step: 54.0 Hyper_FactorUCB w/o W 6.108120649651972
step: 54.0 FactorUCB 0.5438515081206496
step: 54.0 DLinUCB 1.8060324825986078
step: 54.0 ColinUCB 1.6700696055684454
----------------------------
step: 55.0 LinUCB_ItemBased 8.482915717539864
step: 55.0 LinUCB_UserBased 1.6706150341685648
step: 55.0 HybridLinUCB 1.6738041002277904
step: 55.0 FactorUCB w/o W 4.574031890660592
step: 55.0 Hyper_FactorUCB w/o W 6.174031890660593
step: 55.0 FactorUCB 0.5453302961275627
step: 55.0 DLinUCB 1.8050113895216402
step: 55.0 ColinUCB 1.6715261958997722
----------------------------
step: 56.0 LinUCB_ItemBased 8.529569892473118
step: 56.0 LinUCB_UserBased 1.6948924731182795
step: 56.0 HybridLinUCB 1.69668458781362
step: 56.0 FactorUCB w/o W 4.631720430107527
step: 56.0 Hyper_FactorUCB w/o W 6.228046594982079
step: 56.0 FactorUCB 0.5577956989247311
step: 56.0 DLinUCB 1.8261648745519714
step: 56.0 ColinUCB 1.6908602150537635
----------------------------
step: 57.0 LinUCB_ItemBased 8.516213847502192
step: 57.0 LinUCB_UserBased 1.6884312007011393
step: 57.0 HybridLinUCB 1.693689745836985
step: 57.0 FactorUCB w/o W 4.630148992112182
step: 57.0 Hyper_FactorUCB w/o W 6.246713409290097
step: 57.0 FactorUCB 0.5595968448729185
step: 57.0 DLinUCB 1.8181419807186678
step: 57.0 ColinUCB 1.6866783523225242
----------------------------
step: 58.0 LinUCB_ItemBased 8.611087050671287
step: 58.0 LinUCB_UserBased 1.707665656128194
step: 58.0 HybridLinUCB 1.7124296232135123
step: 58.0 FactorUCB w/o W 4.701169337375487
step: 58.0 Hyper_FactorUCB w/o W 6.35036812472932
step: 58.0 FactorUCB 0.56691208315288
step: 58.0 DLinUCB 1.8354265915980945
step: 58.0 ColinUCB 1.70376786487657
----------------------------
step: 59.0 LinUCB_ItemBased 8.65098374679213
step: 59.0 LinUCB_UserBased 1.7142857142857142
step: 59.0 HybridLinUCB 1.7189905902480753
step: 59.0 FactorUCB w/o W 4.73267750213858
step: 59.0 Hyper_FactorUCB w/o W 6.395637296834901
step: 59.0 FactorUCB 0.5718562874251497
step: 59.0 DLinUCB 1.8421727972626176
step: 59.0 ColinUCB 1.7070145423438836
----------------------------
step: 60.0 LinUCB_ItemBased 8.733220768256649
step: 60.0 LinUCB_UserBased 1.718868720979316
step: 60.0 HybridLinUCB 1.7243562684677078
step: 60.0 FactorUCB w/o W 4.78682988602786
step: 60.0 Hyper_FactorUCB w/o W 6.485436893203883
step: 60.0 FactorUCB 0.5728155339805825
step: 60.0 DLinUCB 1.8476150274377374
step: 60.0 ColinUCB 1.7142254115660616
----------------------------
Epoch [5/50], Loss: 3.0815
Epoch [10/50], Loss: 3.0733
Epoch [15/50], Loss: 3.0702
Epoch [20/50], Loss: 3.0681
Epoch [25/50], Loss: 3.0670
Epoch [30/50], Loss: 3.0651
Epoch [35/50], Loss: 3.0639
Epoch [40/50], Loss: 3.0628
Epoch [45/50], Loss: 3.0619
Epoch [50/50], Loss: 3.0610
step: 61.0 LinUCB_ItemBased 8.713871635610767
step: 61.0 LinUCB_UserBased 1.7122153209109732
step: 61.0 HybridLinUCB 1.718840579710145
step: 61.0 FactorUCB w/o W 4.786749482401656
step: 61.0 Hyper_FactorUCB w/o W 6.505175983436853
step: 61.0 FactorUCB 0.5726708074534161
step: 61.0 DLinUCB 1.8409937888198757
step: 61.0 ColinUCB 1.7080745341614907
----------------------------
step: 62.0 LinUCB_ItemBased 8.714285714285714
step: 62.0 LinUCB_UserBased 1.7114367114367115
step: 62.0 HybridLinUCB 1.7163207163207164
step: 62.0 FactorUCB w/o W 4.797313797313797
step: 62.0 Hyper_FactorUCB w/o W 6.529507529507529
step: 62.0 FactorUCB 0.5722425722425722
step: 62.0 DLinUCB 1.8392348392348392
step: 62.0 ColinUCB 1.7073667073667074
----------------------------
step: 63.0 LinUCB_ItemBased 8.767029423619508
step: 63.0 LinUCB_UserBased 1.7178557033454251
step: 63.0 HybridLinUCB 1.724304715840387
step: 63.0 FactorUCB w/o W 4.839580814187827
step: 63.0 Hyper_FactorUCB w/o W 6.594921402660217
step: 63.0 FactorUCB 0.5739621120515921
step: 63.0 DLinUCB 1.8500604594921404
step: 63.0 ColinUCB 1.7134220072551392
----------------------------
step: 64.0 LinUCB_ItemBased 8.742687747035573
step: 64.0 LinUCB_UserBased 1.709090909090909
step: 64.0 HybridLinUCB 1.715810276679842
step: 64.0 FactorUCB w/o W 4.836363636363636
step: 64.0 Hyper_FactorUCB w/o W 6.601581027667984
step: 64.0 FactorUCB 0.5699604743083004
step: 64.0 DLinUCB 1.8430830039525692
step: 64.0 ColinUCB 1.709486166007905
----------------------------
step: 65.0 LinUCB_ItemBased 8.796484375
step: 65.0 LinUCB_UserBased 1.719921875
step: 65.0 HybridLinUCB 1.728125
step: 65.0 FactorUCB w/o W 4.8828125
step: 65.0 Hyper_FactorUCB w/o W 6.659375
step: 65.0 FactorUCB 0.576953125
step: 65.0 DLinUCB 1.855859375
step: 65.0 ColinUCB 1.721875
----------------------------
step: 66.0 LinUCB_ItemBased 8.789655172413793
step: 66.0 LinUCB_UserBased 1.7149425287356321
step: 66.0 HybridLinUCB 1.7226053639846743
step: 66.0 FactorUCB w/o W 4.893103448275862
step: 66.0 Hyper_FactorUCB w/o W 6.684674329501916
step: 66.0 FactorUCB 0.5812260536398467
step: 66.0 DLinUCB 1.8517241379310345
step: 66.0 ColinUCB 1.7210727969348658
----------------------------
step: 67.0 LinUCB_ItemBased 8.766942771084338
step: 67.0 LinUCB_UserBased 1.7157379518072289
step: 67.0 HybridLinUCB 1.7232680722891567
step: 67.0 FactorUCB w/o W 4.90210843373494
step: 67.0 Hyper_FactorUCB w/o W 6.6780873493975905
step: 67.0 FactorUCB 0.5847138554216867
step: 67.0 DLinUCB 1.8524096385542168
step: 67.0 ColinUCB 1.7228915662650603
----------------------------
step: 68.0 LinUCB_ItemBased 8.796213808463252
step: 68.0 LinUCB_UserBased 1.7152932442464737
step: 68.0 HybridLinUCB 1.7249443207126949
step: 68.0 FactorUCB w/o W 4.932442464736451
step: 68.0 Hyper_FactorUCB w/o W 6.726429101707498
step: 68.0 FactorUCB 0.5846325167037862
step: 68.0 DLinUCB 1.8570898292501856
step: 68.0 ColinUCB 1.7268002969561989
----------------------------
step: 69.0 LinUCB_ItemBased 8.782449725776965
step: 69.0 LinUCB_UserBased 1.7111517367458866
step: 69.0 HybridLinUCB 1.7210237659963437
step: 69.0 FactorUCB w/o W 4.945521023765997
step: 69.0 Hyper_FactorUCB w/o W 6.730530164533821
step: 69.0 FactorUCB 0.5835466179159049
step: 69.0 DLinUCB 1.8526508226691043
step: 69.0 ColinUCB 1.7210237659963437
----------------------------
step: 70.0 LinUCB_ItemBased 8.758620689655173
step: 70.0 LinUCB_UserBased 1.6986350574712643
step: 70.0 HybridLinUCB 1.7101293103448276
step: 70.0 FactorUCB w/o W 4.942887931034483
step: 70.0 Hyper_FactorUCB w/o W 6.721623563218391
step: 70.0 FactorUCB 0.5779454022988506
step: 70.0 DLinUCB 1.8444683908045978
step: 70.0 ColinUCB 1.7086925287356323
----------------------------
Epoch [5/50], Loss: 3.1210
Epoch [10/50], Loss: 3.1135
Epoch [15/50], Loss: 3.1104
Epoch [20/50], Loss: 3.1090
Epoch [25/50], Loss: 3.1078
Epoch [30/50], Loss: 3.1071
Epoch [35/50], Loss: 3.1056
Epoch [40/50], Loss: 3.1044
Epoch [45/50], Loss: 3.1033
Epoch [50/50], Loss: 3.1026
step: 71.0 LinUCB_ItemBased 8.741684359518754
step: 71.0 LinUCB_UserBased 1.686482661004954
step: 71.0 HybridLinUCB 1.6988676574663837
step: 71.0 FactorUCB w/o W 4.950460014154282
step: 71.0 Hyper_FactorUCB w/o W 6.71514508138712
step: 71.0 FactorUCB 0.5721868365180467
step: 71.0 DLinUCB 1.835456475583864
step: 71.0 ColinUCB 1.6992215145081386
----------------------------
step: 72.0 LinUCB_ItemBased 8.746242572527088
step: 72.0 LinUCB_UserBased 1.682278923453338
step: 72.0 HybridLinUCB 1.6973086333449843
step: 72.0 FactorUCB w/o W 4.966794826983572
step: 72.0 Hyper_FactorUCB w/o W 6.733659559594547
step: 72.0 FactorUCB 0.5714785040195736
step: 72.0 DLinUCB 1.8402656413841314
step: 72.0 ColinUCB 1.6976581614819992
----------------------------
step: 73.0 LinUCB_ItemBased 8.79751209398756
step: 73.0 LinUCB_UserBased 1.6872840359364203
step: 73.0 HybridLinUCB 1.7035245335176226
step: 73.0 FactorUCB w/o W 5.011402902557014
step: 73.0 Hyper_FactorUCB w/o W 6.802695231513476
step: 73.0 FactorUCB 0.5742916378714582
step: 73.0 DLinUCB 1.8503800967519004
step: 73.0 ColinUCB 1.7059433310297167
----------------------------
step: 74.0 LinUCB_ItemBased 8.806066802999318
step: 74.0 LinUCB_UserBased 1.6905248807089297
step: 74.0 HybridLinUCB 1.7082481254260395
step: 74.0 FactorUCB w/o W 5.034764826175869
step: 74.0 Hyper_FactorUCB w/o W 6.81799591002045
step: 74.0 FactorUCB 0.5760054533060668
step: 74.0 DLinUCB 1.8588957055214723
step: 74.0 ColinUCB 1.7167689161554192
----------------------------
step: 75.0 LinUCB_ItemBased 8.836535220761712
step: 75.0 LinUCB_UserBased 1.6895854398382204
step: 75.0 HybridLinUCB 1.7094708459723627
step: 75.0 FactorUCB w/o W 5.067408156386922
step: 75.0 Hyper_FactorUCB w/o W 6.857094708459724
step: 75.0 FactorUCB 0.5760026963262554
step: 75.0 DLinUCB 1.8608021570610045
step: 75.0 ColinUCB 1.7209302325581395
----------------------------
step: 76.0 LinUCB_ItemBased 8.87462487495832
step: 76.0 LinUCB_UserBased 1.6922307435811936
step: 76.0 HybridLinUCB 1.7125708569523175
step: 76.0 FactorUCB w/o W 5.099033011003668
step: 76.0 Hyper_FactorUCB w/o W 6.911303767922641
step: 76.0 FactorUCB 0.5761920640213405
step: 76.0 DLinUCB 1.8589529843281094
step: 76.0 ColinUCB 1.7232410803601201
----------------------------
step: 77.0 LinUCB_ItemBased 8.892515661061655
step: 77.0 LinUCB_UserBased 1.6874381800197824
step: 77.0 HybridLinUCB 1.7085393999340588
step: 77.0 FactorUCB w/o W 5.120672601384768
step: 77.0 Hyper_FactorUCB w/o W 6.943620178041543
step: 77.0 FactorUCB 0.5740191229805474
step: 77.0 DLinUCB 1.8555885262116716
step: 77.0 ColinUCB 1.7184305967688758
----------------------------
step: 78.0 LinUCB_ItemBased 8.904373368146214
step: 78.0 LinUCB_UserBased 1.683746736292428
step: 78.0 HybridLinUCB 1.7065926892950392
step: 78.0 FactorUCB w/o W 5.136749347258486
step: 78.0 Hyper_FactorUCB w/o W 6.9585509138381205
step: 78.0 FactorUCB 0.5724543080939948
step: 78.0 DLinUCB 1.858028720626632
step: 78.0 ColinUCB 1.718668407310705
----------------------------
step: 79.0 LinUCB_ItemBased 8.92404248471194
step: 79.0 LinUCB_UserBased 1.6868361763759254
step: 79.0 HybridLinUCB 1.710975217251368
step: 79.0 FactorUCB w/o W 5.170582555519794
step: 79.0 Hyper_FactorUCB w/o W 6.992919214676537
step: 79.0 FactorUCB 0.5770840038622466
step: 79.0 DLinUCB 1.8606372706791117
step: 79.0 ColinUCB 1.7228838107499196
----------------------------
step: 80.0 LinUCB_ItemBased 8.940672588832488
step: 80.0 LinUCB_UserBased 1.6855964467005076
step: 80.0 HybridLinUCB 1.7119289340101522
step: 80.0 FactorUCB w/o W 5.191307106598985
step: 80.0 Hyper_FactorUCB w/o W 7.043781725888325
step: 80.0 FactorUCB 0.5815355329949239
step: 80.0 DLinUCB 1.8562817258883249
step: 80.0 ColinUCB 1.7211294416243654
----------------------------
Epoch [5/50], Loss: 3.0438
Epoch [10/50], Loss: 3.0259
Epoch [15/50], Loss: 3.0217
Epoch [20/50], Loss: 3.0197
Epoch [25/50], Loss: 3.0218
Epoch [30/50], Loss: 3.0162
Epoch [35/50], Loss: 3.0149
Epoch [40/50], Loss: 3.0167
Epoch [45/50], Loss: 3.0128
Epoch [50/50], Loss: 3.0115
step: 81.0 LinUCB_ItemBased 8.936696960200564
step: 81.0 LinUCB_UserBased 1.688498903165152
step: 81.0 HybridLinUCB 1.714509558132247
step: 81.0 FactorUCB w/o W 5.192416170479474
step: 81.0 Hyper_FactorUCB w/o W 7.0557818865559385
step: 81.0 FactorUCB 0.5832027577561892
step: 81.0 DLinUCB 1.859918520839862
step: 81.0 ColinUCB 1.7226574741460356
----------------------------
step: 82.0 LinUCB_ItemBased 8.9182603331277
step: 82.0 LinUCB_UserBased 1.6810610734114744
step: 82.0 HybridLinUCB 1.7069710055521283
step: 82.0 FactorUCB w/o W 5.190623072177668
step: 82.0 Hyper_FactorUCB w/o W 7.067550894509562
step: 82.0 FactorUCB 0.581122763726095
step: 82.0 DLinUCB 1.8488587291795189
step: 82.0 ColinUCB 1.715607649599013
----------------------------
step: 83.0 LinUCB_ItemBased 8.934756097560976
step: 83.0 LinUCB_UserBased 1.6832317073170733
step: 83.0 HybridLinUCB 1.7094512195121951
step: 83.0 FactorUCB w/o W 5.210060975609756
step: 83.0 Hyper_FactorUCB w/o W 7.101219512195122
step: 83.0 FactorUCB 0.5844512195121951
step: 83.0 DLinUCB 1.8487804878048781
step: 83.0 ColinUCB 1.7170731707317073
----------------------------
step: 84.0 LinUCB_ItemBased 8.924374811690269
step: 84.0 LinUCB_UserBased 1.6836396504971376
step: 84.0 HybridLinUCB 1.7110575474540524
step: 84.0 FactorUCB w/o W 5.21120819523953
step: 84.0 Hyper_FactorUCB w/o W 7.1066586321181076
step: 84.0 FactorUCB 0.5827056342271768
step: 84.0 DLinUCB 1.8493522145224466
step: 84.0 ColinUCB 1.7167821633021993
----------------------------
step: 85.0 LinUCB_ItemBased 8.920166815609175
step: 85.0 LinUCB_UserBased 1.67828418230563
step: 85.0 HybridLinUCB 1.706881143878463
step: 85.0 FactorUCB w/o W 5.224605302353291
step: 85.0 Hyper_FactorUCB w/o W 7.119749776586238
step: 85.0 FactorUCB 0.5817694369973191
step: 85.0 DLinUCB 1.8513553768245457
step: 85.0 ColinUCB 1.716115579386357
----------------------------
step: 86.0 LinUCB_ItemBased 8.926362297496318
step: 86.0 LinUCB_UserBased 1.6745213549337261
step: 86.0 HybridLinUCB 1.7042709867452135
step: 86.0 FactorUCB w/o W 5.241826215022091
step: 86.0 Hyper_FactorUCB w/o W 7.1393225331369665
step: 86.0 FactorUCB 0.5811487481590575
step: 86.0 DLinUCB 1.8474226804123712
step: 86.0 ColinUCB 1.7119293078055964
----------------------------
step: 87.0 LinUCB_ItemBased 8.942599067599067
step: 87.0 LinUCB_UserBased 1.6777389277389276
step: 87.0 HybridLinUCB 1.7083333333333333
step: 87.0 FactorUCB w/o W 5.261072261072261
step: 87.0 Hyper_FactorUCB w/o W 7.171328671328672
step: 87.0 FactorUCB 0.5801282051282052
step: 87.0 DLinUCB 1.8511072261072261
step: 87.0 ColinUCB 1.7156177156177157
----------------------------
step: 88.0 LinUCB_ItemBased 8.961095100864553
step: 88.0 LinUCB_UserBased 1.6804034582132565
step: 88.0 HybridLinUCB 1.7097982708933717
step: 88.0 FactorUCB w/o W 5.278674351585014
step: 88.0 Hyper_FactorUCB w/o W 7.198847262247838
step: 88.0 FactorUCB 0.5827089337175793
step: 88.0 DLinUCB 1.8530259365994237
step: 88.0 ColinUCB 1.7172910662824208
----------------------------
step: 89.0 LinUCB_ItemBased 8.976644830532612
step: 89.0 LinUCB_UserBased 1.6821418399316435
step: 89.0 HybridLinUCB 1.7123326687553404
step: 89.0 FactorUCB w/o W 5.3019082882369695
step: 89.0 Hyper_FactorUCB w/o W 7.228709769296497
step: 89.0 FactorUCB 0.5841640558245514
step: 89.0 DLinUCB 1.8524636855596697
step: 89.0 ColinUCB 1.7177442324124181
----------------------------
step: 90.0 LinUCB_ItemBased 9.011851015801355
step: 90.0 LinUCB_UserBased 1.6853837471783295
step: 90.0 HybridLinUCB 1.715293453724605
step: 90.0 FactorUCB w/o W 5.334650112866817
step: 90.0 Hyper_FactorUCB w/o W 7.275959367945824
step: 90.0 FactorUCB 0.5857787810383747
step: 90.0 DLinUCB 1.8515801354401806
step: 90.0 ColinUCB 1.7206546275395034
----------------------------
Epoch [5/50], Loss: 3.0847
Epoch [10/50], Loss: 3.0691
Epoch [15/50], Loss: 3.0651
Epoch [20/50], Loss: 3.0614
Epoch [25/50], Loss: 3.0611
Epoch [30/50], Loss: 3.0636
Epoch [35/50], Loss: 3.0572
Epoch [40/50], Loss: 3.0599
Epoch [45/50], Loss: 3.0557
Epoch [50/50], Loss: 3.0560
step: 91.0 LinUCB_ItemBased 9.004453103256331
step: 91.0 LinUCB_UserBased 1.685221263568049
step: 91.0 HybridLinUCB 1.716671305315892
step: 91.0 FactorUCB w/o W 5.344002226551628
step: 91.0 Hyper_FactorUCB w/o W 7.291956582243251
step: 91.0 FactorUCB 0.5858613971611467
step: 91.0 DLinUCB 1.8508210409128862
step: 91.0 ColinUCB 1.7216810464792653
----------------------------
step: 92.0 LinUCB_ItemBased 8.991467107074044
step: 92.0 LinUCB_UserBased 1.677401596476741
step: 92.0 HybridLinUCB 1.7087806220754198
step: 92.0 FactorUCB w/o W 5.345169281585466
step: 92.0 Hyper_FactorUCB w/o W 7.299201761629507
step: 92.0 FactorUCB 0.5816129920176163
step: 92.0 DLinUCB 1.8406275805119736
step: 92.0 ColinUCB 1.712634186622626
----------------------------
step: 93.0 LinUCB_ItemBased 8.975238095238096
step: 93.0 LinUCB_UserBased 1.6737414965986395
step: 93.0 HybridLinUCB 1.7053061224489796
step: 93.0 FactorUCB w/o W 5.342312925170068
step: 93.0 Hyper_FactorUCB w/o W 7.295510204081633
step: 93.0 FactorUCB 0.5801360544217687
step: 93.0 DLinUCB 1.8402721088435374
step: 93.0 ColinUCB 1.7099319727891156
----------------------------
step: 94.0 LinUCB_ItemBased 8.999191592562651
step: 94.0 LinUCB_UserBased 1.6769064942064134
step: 94.0 HybridLinUCB 1.7084343842630019
step: 94.0 FactorUCB w/o W 5.364861223389922
step: 94.0 Hyper_FactorUCB w/o W 7.338992185394773
step: 94.0 FactorUCB 0.5817838857450822
step: 94.0 DLinUCB 1.844246833737537
step: 94.0 ColinUCB 1.7130153597413096
----------------------------
step: 95.0 LinUCB_ItemBased 9.037433155080214
step: 95.0 LinUCB_UserBased 1.6898395721925135
step: 95.0 HybridLinUCB 1.7213903743315508
step: 95.0 FactorUCB w/o W 5.401871657754011
step: 95.0 Hyper_FactorUCB w/o W 7.382620320855615
step: 95.0 FactorUCB 0.5844919786096257
step: 95.0 DLinUCB 1.856149732620321
step: 95.0 ColinUCB 1.7240641711229947
----------------------------
step: 96.0 LinUCB_ItemBased 9.043950225046332
step: 96.0 LinUCB_UserBased 1.6907598623245963
step: 96.0 HybridLinUCB 1.7225311093460418
step: 96.0 FactorUCB w/o W 5.4209690230341545
step: 96.0 Hyper_FactorUCB w/o W 7.408790045009266
step: 96.0 FactorUCB 0.5885623510722796
step: 96.0 DLinUCB 1.8580884299708764
step: 96.0 ColinUCB 1.7272967963992587
----------------------------
step: 97.0 LinUCB_ItemBased 9.061154855643045
step: 97.0 LinUCB_UserBased 1.694225721784777
step: 97.0 HybridLinUCB 1.7257217847769029
step: 97.0 FactorUCB w/o W 5.440682414698163
step: 97.0 Hyper_FactorUCB w/o W 7.443832020997375
step: 97.0 FactorUCB 0.5897637795275591
step: 97.0 DLinUCB 1.8593175853018373
step: 97.0 ColinUCB 1.7288713910761155
----------------------------
step: 98.0 LinUCB_ItemBased 9.08849557522124
step: 98.0 LinUCB_UserBased 1.695991671004685
step: 98.0 HybridLinUCB 1.7280062467464863
step: 98.0 FactorUCB w/o W 5.468245705361791
step: 98.0 Hyper_FactorUCB w/o W 7.482821447162936
step: 98.0 FactorUCB 0.5913586673607496
step: 98.0 DLinUCB 1.8597084851639771
step: 98.0 ColinUCB 1.7300884955752212
----------------------------
step: 99.0 LinUCB_ItemBased 9.085390946502057
step: 99.0 LinUCB_UserBased 1.6929012345679013
step: 99.0 HybridLinUCB 1.725051440329218
step: 99.0 FactorUCB w/o W 5.477880658436214
step: 99.0 Hyper_FactorUCB w/o W 7.496913580246914
step: 99.0 FactorUCB 0.5941358024691358
step: 99.0 DLinUCB 1.8587962962962963
step: 99.0 ColinUCB 1.7286522633744856
----------------------------
step: 100.0 LinUCB_ItemBased 9.1017597551645
step: 100.0 LinUCB_UserBased 1.6952308084672278
step: 100.0 HybridLinUCB 1.7268553940321347
step: 100.0 FactorUCB w/o W 5.497322111706198
step: 100.0 Hyper_FactorUCB w/o W 7.518745218056618
step: 100.0 FactorUCB 0.5975516449885233
step: 100.0 DLinUCB 1.866360622290232
step: 100.0 ColinUCB 1.7319561336393776
----------------------------
Epoch [5/50], Loss: 3.1158
Epoch [10/50], Loss: 3.1187
Epoch [15/50], Loss: 3.1097
Epoch [20/50], Loss: 3.1080
Epoch [25/50], Loss: 3.1054
Epoch [30/50], Loss: 3.1045
Epoch [35/50], Loss: 3.1033
Epoch [40/50], Loss: 3.1025
Epoch [45/50], Loss: 3.1056
Epoch [50/50], Loss: 3.1034
step: 101.0 LinUCB_ItemBased 9.128354430379748
step: 101.0 LinUCB_UserBased 1.700759493670886
step: 101.0 HybridLinUCB 1.7326582278481013
step: 101.0 FactorUCB w/o W 5.52632911392405
step: 101.0 Hyper_FactorUCB w/o W 7.557721518987342
step: 101.0 FactorUCB 0.5964556962025317
step: 101.0 DLinUCB 1.8762025316455697
step: 101.0 ColinUCB 1.740506329113924
----------------------------
step: 102.0 LinUCB_ItemBased 9.123308270676691
step: 102.0 LinUCB_UserBased 1.7020050125313284
step: 102.0 HybridLinUCB 1.7330827067669172
step: 102.0 FactorUCB w/o W 5.534085213032581
step: 102.0 Hyper_FactorUCB w/o W 7.567167919799498
step: 102.0 FactorUCB 0.5977443609022557
step: 102.0 DLinUCB 1.874185463659148
step: 102.0 ColinUCB 1.7401002506265664
----------------------------
step: 103.0 LinUCB_ItemBased 9.119851116625311
step: 103.0 LinUCB_UserBased 1.6977667493796527
step: 103.0 HybridLinUCB 1.7277915632754342
step: 103.0 FactorUCB w/o W 5.541935483870968
step: 103.0 Hyper_FactorUCB w/o W 7.574441687344913
step: 103.0 FactorUCB 0.5965260545905707
step: 103.0 DLinUCB 1.8704714640198512
step: 103.0 ColinUCB 1.7359801488833746
----------------------------
step: 104.0 LinUCB_ItemBased 9.106335952848722
step: 104.0 LinUCB_UserBased 1.693762278978389
step: 104.0 HybridLinUCB 1.723968565815324
step: 104.0 FactorUCB w/o W 5.5456777996070725
step: 104.0 Hyper_FactorUCB w/o W 7.580550098231827
step: 104.0 FactorUCB 0.593811394891945
step: 104.0 DLinUCB 1.8661591355599214
step: 104.0 ColinUCB 1.731090373280943
----------------------------
step: 105.0 LinUCB_ItemBased 9.123995127892814
step: 105.0 LinUCB_UserBased 1.6984165651644336
step: 105.0 HybridLinUCB 1.7278928136419
step: 105.0 FactorUCB w/o W 5.565164433617539
step: 105.0 Hyper_FactorUCB w/o W 7.606577344701583
step: 105.0 FactorUCB 0.5963459196102314
step: 105.0 DLinUCB 1.8706455542021925
step: 105.0 ColinUCB 1.7317904993909865
----------------------------
step: 106.0 LinUCB_ItemBased 9.125482625482626
step: 106.0 LinUCB_UserBased 1.70246138996139
step: 106.0 HybridLinUCB 1.7328667953667953
step: 106.0 FactorUCB w/o W 5.582046332046332
step: 106.0 Hyper_FactorUCB w/o W 7.61969111969112
step: 106.0 FactorUCB 0.599903474903475
step: 106.0 DLinUCB 1.875
step: 106.0 ColinUCB 1.7338320463320462
----------------------------
step: 107.0 LinUCB_ItemBased 9.14124970074216
step: 107.0 LinUCB_UserBased 1.7060090974383528
step: 107.0 HybridLinUCB 1.7368925065836724
step: 107.0 FactorUCB w/o W 5.600191525017955
step: 107.0 Hyper_FactorUCB w/o W 7.648551592051712
step: 107.0 FactorUCB 0.6037826191046205
step: 107.0 DLinUCB 1.878860426143165
step: 107.0 ColinUCB 1.7380895379458943
----------------------------
step: 108.0 LinUCB_ItemBased 9.168448562603944
step: 108.0 LinUCB_UserBased 1.709432169161321
step: 108.0 HybridLinUCB 1.7415062960323118
step: 108.0 FactorUCB w/o W 5.624376336421953
step: 108.0 Hyper_FactorUCB w/o W 7.684248039914469
step: 108.0 FactorUCB 0.6058446186742694
step: 108.0 DLinUCB 1.8823948681397007
step: 108.0 ColinUCB 1.7422190544072227
----------------------------
step: 109.0 LinUCB_ItemBased 9.184452296819789
step: 109.0 LinUCB_UserBased 1.7071849234393404
step: 109.0 HybridLinUCB 1.7399293286219082
step: 109.0 FactorUCB w/o W 5.641460541813899
step: 109.0 Hyper_FactorUCB w/o W 7.7114252061248525
step: 109.0 FactorUCB 0.6054181389870436
step: 109.0 DLinUCB 1.8829210836277974
step: 109.0 ColinUCB 1.7411071849234394
----------------------------
step: 110.0 LinUCB_ItemBased 9.170083876980428
step: 110.0 LinUCB_UserBased 1.7043336439888164
step: 110.0 HybridLinUCB 1.7371854613233924
step: 110.0 FactorUCB w/o W 5.633737185461324
step: 110.0 Hyper_FactorUCB w/o W 7.708294501397949
step: 110.0 FactorUCB 0.6039142590866728
step: 110.0 DLinUCB 1.8776794035414726
step: 110.0 ColinUCB 1.7374184529356944
----------------------------
Epoch [5/50], Loss: 3.1268
Epoch [10/50], Loss: 3.1227
Epoch [15/50], Loss: 3.1228
Epoch [20/50], Loss: 3.1236
Epoch [25/50], Loss: 3.1202
Epoch [30/50], Loss: 3.1182
Epoch [35/50], Loss: 3.1168
Epoch [40/50], Loss: 3.1152
Epoch [45/50], Loss: 3.1151
Epoch [50/50], Loss: 3.1166
step: 111.0 LinUCB_ItemBased 9.158331412767918
step: 111.0 LinUCB_UserBased 1.6985480525466696
step: 111.0 HybridLinUCB 1.7310440193592993
step: 111.0 FactorUCB w/o W 5.638626411615579
step: 111.0 Hyper_FactorUCB w/o W 7.712606591380503
step: 111.0 FactorUCB 0.6019820235077207
step: 111.0 DLinUCB 1.8713989398478912
step: 111.0 ColinUCB 1.7298916801106246
----------------------------
step: 112.0 LinUCB_ItemBased 9.171617915904935
step: 112.0 LinUCB_UserBased 1.6994972577696525
step: 112.0 HybridLinUCB 1.731946983546618
step: 112.0 FactorUCB w/o W 5.658135283363802
step: 112.0 Hyper_FactorUCB w/o W 7.7356032906764165
step: 112.0 FactorUCB 0.6055758683729433
step: 112.0 DLinUCB 1.8711151736745886
step: 112.0 ColinUCB 1.731946983546618
----------------------------
step: 113.0 LinUCB_ItemBased 9.164327750113173
step: 113.0 LinUCB_UserBased 1.6994114984155726
step: 113.0 HybridLinUCB 1.7313263920325939
step: 113.0 FactorUCB w/o W 5.6609325486645545
step: 113.0 Hyper_FactorUCB w/o W 7.739927569035763
step: 113.0 FactorUCB 0.6047985513807153
step: 113.0 DLinUCB 1.8703033046627433
step: 113.0 ColinUCB 1.7326844726120416
----------------------------
step: 114.0 LinUCB_ItemBased 9.158189558592875
step: 114.0 LinUCB_UserBased 1.6966166255881694
step: 114.0 HybridLinUCB 1.727313466278288
step: 114.0 FactorUCB w/o W 5.665471655836881
step: 114.0 Hyper_FactorUCB w/o W 7.749047725745014
step: 114.0 FactorUCB 0.6027335872731346
step: 114.0 DLinUCB 1.8657853461796998
step: 114.0 ColinUCB 1.7293300470535515
----------------------------
step: 115.0 LinUCB_ItemBased 9.153197158081705
step: 115.0 LinUCB_UserBased 1.6951598579040852
step: 115.0 HybridLinUCB 1.7255772646536411
step: 115.0 FactorUCB w/o W 5.674289520426288
step: 115.0 Hyper_FactorUCB w/o W 7.755994671403197
step: 115.0 FactorUCB 0.6016873889875666
step: 115.0 DLinUCB 1.8650088809946714
step: 115.0 ColinUCB 1.728019538188277
----------------------------
step: 116.0 LinUCB_ItemBased 9.14974675181678
step: 116.0 LinUCB_UserBased 1.6914776480951332
step: 116.0 HybridLinUCB 1.7218674300814798
step: 116.0 FactorUCB w/o W 5.684210526315789
step: 116.0 Hyper_FactorUCB w/o W 7.766351024003524
step: 116.0 FactorUCB 0.6007487337590839
step: 116.0 DLinUCB 1.863025765249945
step: 116.0 ColinUCB 1.7234089407619466
----------------------------
step: 117.0 LinUCB_ItemBased 9.150436681222708
step: 117.0 LinUCB_UserBased 1.6888646288209608
step: 117.0 HybridLinUCB 1.7207423580786025
step: 117.0 FactorUCB w/o W 5.695196506550218
step: 117.0 Hyper_FactorUCB w/o W 7.7796943231441045
step: 117.0 FactorUCB 0.5997816593886462
step: 117.0 DLinUCB 1.8609170305676856
step: 117.0 ColinUCB 1.72117903930131
----------------------------
step: 118.0 LinUCB_ItemBased 9.157792207792207
step: 118.0 LinUCB_UserBased 1.6924242424242424
step: 118.0 HybridLinUCB 1.722943722943723
step: 118.0 FactorUCB w/o W 5.704761904761905
step: 118.0 Hyper_FactorUCB w/o W 7.796536796536796
step: 118.0 FactorUCB 0.5987012987012987
step: 118.0 DLinUCB 1.861904761904762
step: 118.0 ColinUCB 1.7216450216450216
----------------------------
step: 119.0 LinUCB_ItemBased 9.152593227603944
step: 119.0 LinUCB_UserBased 1.689669952850407
step: 119.0 HybridLinUCB 1.7203171881697386
step: 119.0 FactorUCB w/o W 5.709601371624518
step: 119.0 Hyper_FactorUCB w/o W 7.805400771538792
step: 119.0 FactorUCB 0.5987998285469353
step: 119.0 DLinUCB 1.8583369052721816
step: 119.0 ColinUCB 1.7183883411915988
----------------------------
step: 120.0 LinUCB_ItemBased 9.134394904458599
step: 120.0 LinUCB_UserBased 1.6872611464968152
step: 120.0 HybridLinUCB 1.718471337579618
step: 120.0 FactorUCB w/o W 5.70552016985138
step: 120.0 Hyper_FactorUCB w/o W 7.7989384288747345
step: 120.0 FactorUCB 0.6
step: 120.0 DLinUCB 1.8547770700636943
step: 120.0 ColinUCB 1.7146496815286625
----------------------------
Epoch [5/50], Loss: 3.1359
Epoch [10/50], Loss: 3.1283
Epoch [15/50], Loss: 3.1264
Epoch [20/50], Loss: 3.1244
Epoch [25/50], Loss: 3.1232
Epoch [30/50], Loss: 3.1222
Epoch [35/50], Loss: 3.1227
Epoch [40/50], Loss: 3.1215
Epoch [45/50], Loss: 3.1222
Epoch [50/50], Loss: 3.1233
step: 121.0 LinUCB_ItemBased 9.1319722163755
step: 121.0 LinUCB_UserBased 1.6825931382866766
step: 121.0 HybridLinUCB 1.7143759208587666
step: 121.0 FactorUCB w/o W 5.711008208798148
step: 121.0 Hyper_FactorUCB w/o W 7.798989686381814
step: 121.0 FactorUCB 0.5973479267522627
step: 121.0 DLinUCB 1.8503472953062514
step: 121.0 ColinUCB 1.7091138707640496
----------------------------
step: 122.0 LinUCB_ItemBased 9.12
step: 122.0 LinUCB_UserBased 1.6808333333333334
step: 122.0 HybridLinUCB 1.713125
step: 122.0 FactorUCB w/o W 5.713541666666667
step: 122.0 Hyper_FactorUCB w/o W 7.799166666666666
step: 122.0 FactorUCB 0.5979166666666667
step: 122.0 DLinUCB 1.8477083333333333
step: 122.0 ColinUCB 1.7079166666666667
----------------------------
step: 123.0 LinUCB_ItemBased 9.120140612076096
step: 123.0 LinUCB_UserBased 1.6811414392059554
step: 123.0 HybridLinUCB 1.7136062861869314
step: 123.0 FactorUCB w/o W 5.715260545905707
step: 123.0 Hyper_FactorUCB w/o W 7.799007444168734
step: 123.0 FactorUCB 0.6000827129859387
step: 123.0 DLinUCB 1.847394540942928
step: 123.0 ColinUCB 1.706782464846981
----------------------------
step: 124.0 LinUCB_ItemBased 9.120795734208368
step: 124.0 LinUCB_UserBased 1.6800656275635768
step: 124.0 HybridLinUCB 1.7128794093519277
step: 124.0 FactorUCB w/o W 5.723133716160787
step: 124.0 Hyper_FactorUCB w/o W 7.811525840853158
step: 124.0 FactorUCB 0.5976210008203445
step: 124.0 DLinUCB 1.8468006562756358
step: 124.0 ColinUCB 1.705906480721903
----------------------------
step: 125.0 LinUCB_ItemBased 9.115470624110591
step: 125.0 LinUCB_UserBased 1.679203090058955
step: 125.0 HybridLinUCB 1.7119333197804432
step: 125.0 FactorUCB w/o W 5.723317747509657
step: 125.0 Hyper_FactorUCB w/o W 7.812156942467981
step: 125.0 FactorUCB 0.5980890424883106
step: 125.0 DLinUCB 1.8459036389510064
step: 125.0 ColinUCB 1.7035982923358406
----------------------------
step: 126.0 LinUCB_ItemBased 9.105868118572293
step: 126.0 LinUCB_UserBased 1.680782415809639
step: 126.0 HybridLinUCB 1.7136519459568462
step: 126.0 FactorUCB w/o W 5.7275660415406335
step: 126.0 Hyper_FactorUCB w/o W 7.811050615043356
step: 126.0 FactorUCB 0.59931437789877
step: 126.0 DLinUCB 1.8457350272232305
step: 126.0 ColinUCB 1.7051824964710627
----------------------------
step: 127.0 LinUCB_ItemBased 9.108486789431545
step: 127.0 LinUCB_UserBased 1.6797437950360288
step: 127.0 HybridLinUCB 1.7129703763010409
step: 127.0 FactorUCB w/o W 5.736989591673339
step: 127.0 Hyper_FactorUCB w/o W 7.822257806244996
step: 127.0 FactorUCB 0.5986789431545236
step: 127.0 DLinUCB 1.845476381104884
step: 127.0 ColinUCB 1.705564451561249
----------------------------
step: 128.0 LinUCB_ItemBased 9.080063416567578
step: 128.0 LinUCB_UserBased 1.6710265556876733
step: 128.0 HybridLinUCB 1.7033293697978598
step: 128.0 FactorUCB w/o W 5.726317875544986
step: 128.0 Hyper_FactorUCB w/o W 7.8059849385652
step: 128.0 FactorUCB 0.5969084423305588
step: 128.0 DLinUCB 1.8396749900911613
step: 128.0 ColinUCB 1.698969480776853
----------------------------
step: 129.0 LinUCB_ItemBased 9.083070866141732
step: 129.0 LinUCB_UserBased 1.6698818897637795
step: 129.0 HybridLinUCB 1.7021653543307087
step: 129.0 FactorUCB w/o W 5.735826771653543
step: 129.0 Hyper_FactorUCB w/o W 7.8153543307086615
step: 129.0 FactorUCB 0.5978346456692913
step: 129.0 DLinUCB 1.8377952755905511
step: 129.0 ColinUCB 1.697244094488189
----------------------------
step: 130.0 LinUCB_ItemBased 9.07725321888412
step: 130.0 LinUCB_UserBased 1.667577058134998
step: 130.0 HybridLinUCB 1.6997658993367148
step: 130.0 FactorUCB w/o W 5.738782676550917
step: 130.0 Hyper_FactorUCB w/o W 7.818767069840031
step: 130.0 FactorUCB 0.5983222785797893
step: 130.0 DLinUCB 1.832618025751073
step: 130.0 ColinUCB 1.6927428794381585
----------------------------
Epoch [5/50], Loss: 3.1452
Epoch [10/50], Loss: 3.1371
Epoch [15/50], Loss: 3.1364
Epoch [20/50], Loss: 3.1367
Epoch [25/50], Loss: 3.1452
Epoch [30/50], Loss: 3.1354
Epoch [35/50], Loss: 3.1330
Epoch [40/50], Loss: 3.1303
Epoch [45/50], Loss: 3.1299
Epoch [50/50], Loss: 3.1294
step: 131.0 LinUCB_ItemBased 9.076074332171894
step: 131.0 LinUCB_UserBased 1.6666666666666667
step: 131.0 HybridLinUCB 1.6989934185056137
step: 131.0 FactorUCB w/o W 5.748161053039102
step: 131.0 Hyper_FactorUCB w/o W 7.82365466511808
step: 131.0 FactorUCB 0.5981416957026713
step: 131.0 DLinUCB 1.8304297328687573
step: 131.0 ColinUCB 1.6898954703832754
----------------------------
step: 132.0 LinUCB_ItemBased 9.083092902481246
step: 132.0 LinUCB_UserBased 1.6710905943450665
step: 132.0 HybridLinUCB 1.7032121561838816
step: 132.0 FactorUCB w/o W 5.770532794768225
step: 132.0 Hyper_FactorUCB w/o W 7.839007501442585
step: 132.0 FactorUCB 0.6014618195806886
step: 132.0 DLinUCB 1.8338141950375073
step: 132.0 ColinUCB 1.6932102327370648
----------------------------
step: 133.0 LinUCB_ItemBased 9.08957219251337
step: 133.0 LinUCB_UserBased 1.6728418640183347
step: 133.0 HybridLinUCB 1.7049274255156608
step: 133.0 FactorUCB w/o W 5.784186401833461
step: 133.0 Hyper_FactorUCB w/o W 7.849503437738732
step: 133.0 FactorUCB 0.6042780748663101
step: 133.0 DLinUCB 1.8334606569900687
step: 133.0 ColinUCB 1.6932773109243697
----------------------------
step: 134.0 LinUCB_ItemBased 9.08128078817734
step: 134.0 LinUCB_UserBased 1.6707086017430846
step: 134.0 HybridLinUCB 1.702917771883289
step: 134.0 FactorUCB w/o W 5.787798408488063
step: 134.0 Hyper_FactorUCB w/o W 7.849753694581281
step: 134.0 FactorUCB 0.6028798787419477
step: 134.0 DLinUCB 1.8285335354300871
step: 134.0 ColinUCB 1.6890867752936718
----------------------------
step: 135.0 LinUCB_ItemBased 9.064461567374554
step: 135.0 LinUCB_UserBased 1.6666040218004134
step: 135.0 HybridLinUCB 1.6979890997932718
step: 135.0 FactorUCB w/o W 5.782183800037587
step: 135.0 Hyper_FactorUCB w/o W 7.84194700244315
step: 135.0 FactorUCB 0.6008269122345424
step: 135.0 DLinUCB 1.8239052809622252
step: 135.0 ColinUCB 1.6829543318925013
----------------------------
step: 136.0 LinUCB_ItemBased 9.049198658218412
step: 136.0 LinUCB_UserBased 1.6707044353335818
step: 136.0 HybridLinUCB 1.7055534849049572
step: 136.0 FactorUCB w/o W 5.781028699217294
step: 136.0 Hyper_FactorUCB w/o W 7.834140887066717
step: 136.0 FactorUCB 0.6000745434215431
step: 136.0 DLinUCB 1.8389862094670146
step: 136.0 ColinUCB 1.6941856131196422
----------------------------
step: 137.0 LinUCB_ItemBased 9.033216460601587
step: 137.0 LinUCB_UserBased 1.664513747923971
step: 137.0 HybridLinUCB 1.6992064956634065
step: 137.0 FactorUCB w/o W 5.776527034508212
step: 137.0 Hyper_FactorUCB w/o W 7.82764347665621
step: 137.0 FactorUCB 0.5977117549363351
step: 137.0 DLinUCB 1.8317032662852926
step: 137.0 ColinUCB 1.6885034139140063
----------------------------
step: 138.0 LinUCB_ItemBased 9.022327964860908
step: 138.0 LinUCB_UserBased 1.6634333821376281
step: 138.0 HybridLinUCB 1.699121522693997
step: 138.0 FactorUCB w/o W 5.776354319180088
step: 138.0 Hyper_FactorUCB w/o W 7.826134699853587
step: 138.0 FactorUCB 0.6017569546120058
step: 138.0 DLinUCB 1.8303440702781846
step: 138.0 ColinUCB 1.6874084919472914
----------------------------
step: 139.0 LinUCB_ItemBased 9.014166363966583
step: 139.0 LinUCB_UserBased 1.6621867054122774
step: 139.0 HybridLinUCB 1.6976026153287322
step: 139.0 FactorUCB w/o W 5.7776970577551765
step: 139.0 Hyper_FactorUCB w/o W 7.824736650926262
step: 139.0 FactorUCB 0.6000726480203414
step: 139.0 DLinUCB 1.829095532146749
step: 139.0 ColinUCB 1.6839811115147112
----------------------------
step: 140.0 LinUCB_ItemBased 9.003063615065777
step: 140.0 LinUCB_UserBased 1.6606595783023967
step: 140.0 HybridLinUCB 1.6961614705352315
step: 140.0 FactorUCB w/o W 5.777978014056587
step: 140.0 Hyper_FactorUCB w/o W 7.823932240043251
step: 140.0 FactorUCB 0.5997477022887007
step: 140.0 DLinUCB 1.8271760677599567
step: 140.0 ColinUCB 1.6833663723193368
----------------------------
Epoch [5/50], Loss: 3.1571
Epoch [10/50], Loss: 3.1564
Epoch [15/50], Loss: 3.1628
Epoch [20/50], Loss: 3.1516
Epoch [25/50], Loss: 3.1489
Epoch [30/50], Loss: 3.1485
Epoch [35/50], Loss: 3.1496
Epoch [40/50], Loss: 3.1471
Epoch [45/50], Loss: 3.1460
Epoch [50/50], Loss: 3.1504
step: 141.0 LinUCB_ItemBased 8.98125
step: 141.0 LinUCB_UserBased 1.6553571428571427
step: 141.0 HybridLinUCB 1.6916071428571429
step: 141.0 FactorUCB w/o W 5.76625
step: 141.0 Hyper_FactorUCB w/o W 7.811964285714286
step: 141.0 FactorUCB 0.6007142857142858
step: 141.0 DLinUCB 1.8217857142857143
step: 141.0 ColinUCB 1.6792857142857143
----------------------------
step: 142.0 LinUCB_ItemBased 8.966335931963147
step: 142.0 LinUCB_UserBased 1.648830616583983
step: 142.0 HybridLinUCB 1.6855067328136073
step: 142.0 FactorUCB w/o W 5.757795889440113
step: 142.0 Hyper_FactorUCB w/o W 7.802622253720766
step: 142.0 FactorUCB 0.5999291282778172
step: 142.0 DLinUCB 1.8155563430191353
step: 142.0 ColinUCB 1.6731041814316088
----------------------------
step: 143.0 LinUCB_ItemBased 8.963223649480907
step: 143.0 LinUCB_UserBased 1.6487770543726905
step: 143.0 HybridLinUCB 1.6857293682913954
step: 143.0 FactorUCB w/o W 5.762449410522612
step: 143.0 Hyper_FactorUCB w/o W 7.804856589829315
step: 143.0 FactorUCB 0.6028506070737286
step: 143.0 DLinUCB 1.8140066866091853
step: 143.0 ColinUCB 1.6728840401196552
----------------------------
step: 144.0 LinUCB_ItemBased 8.959490134450848
step: 144.0 LinUCB_UserBased 1.648507071765322
step: 144.0 HybridLinUCB 1.6867469879518073
step: 144.0 FactorUCB w/o W 5.768290553518422
step: 144.0 Hyper_FactorUCB w/o W 7.812991094814039
step: 144.0 FactorUCB 0.6020604155753448
step: 144.0 DLinUCB 1.816832547581631
step: 144.0 ColinUCB 1.675746464117339
----------------------------
step: 145.0 LinUCB_ItemBased 8.968212610734758
step: 145.0 LinUCB_UserBased 1.6492965085982283
step: 145.0 HybridLinUCB 1.6869897516067396
step: 145.0 FactorUCB w/o W 5.779225290950148
step: 145.0 Hyper_FactorUCB w/o W 7.820392565572346
step: 145.0 FactorUCB 0.6020496786520757
step: 145.0 DLinUCB 1.8181344450234498
step: 145.0 ColinUCB 1.6744832377974639
----------------------------
step: 146.0 LinUCB_ItemBased 8.988754325259515
step: 146.0 LinUCB_UserBased 1.6522491349480968
step: 146.0 HybridLinUCB 1.6908304498269897
step: 146.0 FactorUCB w/o W 5.802076124567474
step: 146.0 Hyper_FactorUCB w/o W 7.8430795847750865
step: 146.0 FactorUCB 0.6060553633217993
step: 146.0 DLinUCB 1.8219723183391003
step: 146.0 ColinUCB 1.6768166089965397
----------------------------
step: 147.0 LinUCB_ItemBased 8.96690672153635
step: 147.0 LinUCB_UserBased 1.6450617283950617
step: 147.0 HybridLinUCB 1.6841563786008231
step: 147.0 FactorUCB w/o W 5.790466392318244
step: 147.0 Hyper_FactorUCB w/o W 7.833504801097393
step: 147.0 FactorUCB 0.6025377229080933
step: 147.0 DLinUCB 1.8136145404663924
step: 147.0 ColinUCB 1.6695816186556927
----------------------------
step: 148.0 LinUCB_ItemBased 8.948826929615777
step: 148.0 LinUCB_UserBased 1.638728323699422
step: 148.0 HybridLinUCB 1.6790207412444746
step: 148.0 FactorUCB w/o W 5.781706902414145
step: 148.0 Hyper_FactorUCB w/o W 7.8216592995579735
step: 148.0 FactorUCB 0.6004760285617137
step: 148.0 DLinUCB 1.8083985039102346
step: 148.0 ColinUCB 1.6645698741924515
----------------------------
step: 149.0 LinUCB_ItemBased 8.949653657712451
step: 149.0 LinUCB_UserBased 1.6386213887480994
step: 149.0 HybridLinUCB 1.6798445683392464
step: 149.0 FactorUCB w/o W 5.790336205440108
step: 149.0 Hyper_FactorUCB w/o W 7.835107281635412
step: 149.0 FactorUCB 0.6021287379624937
step: 149.0 DLinUCB 1.8090893732049333
step: 149.0 ColinUCB 1.6644703497212368
----------------------------
step: 150.0 LinUCB_ItemBased 8.952660735269431
step: 150.0 LinUCB_UserBased 1.6372335067987243
step: 150.0 HybridLinUCB 1.6798724190028538
step: 150.0 FactorUCB w/o W 5.797381232163841
step: 150.0 Hyper_FactorUCB w/o W 7.850092328353198
step: 150.0 FactorUCB 0.603323820715125
step: 150.0 DLinUCB 1.8081248950814168
step: 150.0 ColinUCB 1.6635890548934027
----------------------------
Epoch [5/50], Loss: 3.1378
Epoch [10/50], Loss: 3.1312
Epoch [15/50], Loss: 3.1298
Epoch [20/50], Loss: 3.1306
Epoch [25/50], Loss: 3.1277
Epoch [30/50], Loss: 3.1316
Epoch [35/50], Loss: 3.1332
Epoch [40/50], Loss: 3.1283
Epoch [45/50], Loss: 3.1261
Epoch [50/50], Loss: 3.1240
step: 151.0 LinUCB_ItemBased 8.939540306462359
step: 151.0 LinUCB_UserBased 1.6329113924050633
step: 151.0 HybridLinUCB 1.6767155229846769
step: 151.0 FactorUCB w/o W 5.792638241172551
step: 151.0 Hyper_FactorUCB w/o W 7.843437708194537
step: 151.0 FactorUCB 0.6015989340439707
step: 151.0 DLinUCB 1.804130579613591
step: 151.0 ColinUCB 1.6610592938041306
----------------------------
step: 152.0 LinUCB_ItemBased 8.924322538003965
step: 152.0 LinUCB_UserBased 1.6283873099801718
step: 152.0 HybridLinUCB 1.673165895571712
step: 152.0 FactorUCB w/o W 5.786186384666226
step: 152.0 Hyper_FactorUCB w/o W 7.834600132187706
step: 152.0 FactorUCB 0.6002974223397224
step: 152.0 DLinUCB 1.8022141440846002
step: 152.0 ColinUCB 1.6579643093192333
----------------------------
step: 153.0 LinUCB_ItemBased 8.925098554533509
step: 153.0 LinUCB_UserBased 1.6277923784494086
step: 153.0 HybridLinUCB 1.6736202365308803
step: 153.0 FactorUCB w/o W 5.79122864651774
step: 153.0 Hyper_FactorUCB w/o W 7.843134034165572
step: 153.0 FactorUCB 0.6008541392904073
step: 153.0 DLinUCB 1.802562417871222
step: 153.0 ColinUCB 1.6590013140604467
----------------------------
step: 154.0 LinUCB_ItemBased 8.91648996900995
step: 154.0 LinUCB_UserBased 1.6245310716033274
step: 154.0 HybridLinUCB 1.6703637253302888
step: 154.0 FactorUCB w/o W 5.7877997064100475
step: 154.0 Hyper_FactorUCB w/o W 7.8430924808351
step: 154.0 FactorUCB 0.5994128200946012
step: 154.0 DLinUCB 1.798401565813081
step: 154.0 ColinUCB 1.6558473332245962
----------------------------
step: 155.0 LinUCB_ItemBased 8.914248662668179
step: 155.0 LinUCB_UserBased 1.621494569622305
step: 155.0 HybridLinUCB 1.6681796077159994
step: 155.0 FactorUCB w/o W 5.787647917004377
step: 155.0 Hyper_FactorUCB w/o W 7.844869508834495
step: 155.0 FactorUCB 0.5992867563624574
step: 155.0 DLinUCB 1.7962392608202302
step: 155.0 ColinUCB 1.6539147349651484
----------------------------
step: 156.0 LinUCB_ItemBased 8.911797843231932
step: 156.0 LinUCB_UserBased 1.619024625784645
step: 156.0 HybridLinUCB 1.666022855303396
step: 156.0 FactorUCB w/o W 5.792531788186062
step: 156.0 Hyper_FactorUCB w/o W 7.852406244970224
step: 156.0 FactorUCB 0.5998712377273459
step: 156.0 DLinUCB 1.7925317881860614
step: 156.0 ColinUCB 1.653146628037985
----------------------------
step: 157.0 LinUCB_ItemBased 8.908770806658131
step: 157.0 LinUCB_UserBased 1.6198783610755443
step: 157.0 HybridLinUCB 1.6674135723431498
step: 157.0 FactorUCB w/o W 5.795294494238156
step: 157.0 Hyper_FactorUCB w/o W 7.855473751600512
step: 157.0 FactorUCB 0.6016325224071702
step: 157.0 DLinUCB 1.7933738796414853
step: 157.0 ColinUCB 1.6541293213828425
----------------------------
step: 158.0 LinUCB_ItemBased 8.903974562798092
step: 158.0 LinUCB_UserBased 1.6181240063593005
step: 158.0 HybridLinUCB 1.6655007949125595
step: 158.0 FactorUCB w/o W 5.797138314785373
step: 158.0 Hyper_FactorUCB w/o W 7.85389507154213
step: 158.0 FactorUCB 0.6006359300476948
step: 158.0 DLinUCB 1.7912559618441972
step: 158.0 ColinUCB 1.6513513513513514
----------------------------
step: 159.0 LinUCB_ItemBased 8.89559311325225
step: 159.0 LinUCB_UserBased 1.6157005212446691
step: 159.0 HybridLinUCB 1.6627704943926709
step: 159.0 FactorUCB w/o W 5.795450955615227
step: 159.0 Hyper_FactorUCB w/o W 7.852945822145001
step: 159.0 FactorUCB 0.6000631811720107
step: 159.0 DLinUCB 1.7888169325540988
step: 159.0 ColinUCB 1.648870636550308
----------------------------
step: 160.0 LinUCB_ItemBased 8.912551116703366
step: 160.0 LinUCB_UserBased 1.6203208556149733
step: 160.0 HybridLinUCB 1.6684491978609626
step: 160.0 FactorUCB w/o W 5.8122050959421205
step: 160.0 Hyper_FactorUCB w/o W 7.875117961623152
step: 160.0 FactorUCB 0.600975149418056
step: 160.0 DLinUCB 1.7982069833280907
step: 160.0 ColinUCB 1.6557093425605536
----------------------------
Epoch [5/50], Loss: 3.1655
Epoch [10/50], Loss: 3.1718
Epoch [15/50], Loss: 3.1609
Epoch [20/50], Loss: 3.1578
Epoch [25/50], Loss: 3.1569
Epoch [30/50], Loss: 3.1535
Epoch [35/50], Loss: 3.1538
Epoch [40/50], Loss: 3.1546
Epoch [45/50], Loss: 3.1545
Epoch [50/50], Loss: 3.1544
step: 161.0 LinUCB_ItemBased 8.9096875
step: 161.0 LinUCB_UserBased 1.61953125
step: 161.0 HybridLinUCB 1.66796875
step: 161.0 FactorUCB w/o W 5.81359375
step: 161.0 Hyper_FactorUCB w/o W 7.87515625
step: 161.0 FactorUCB 0.60234375
step: 161.0 DLinUCB 1.79796875
step: 161.0 ColinUCB 1.654375
----------------------------
step: 162.0 LinUCB_ItemBased 8.89941012108041
step: 162.0 LinUCB_UserBased 1.6178205526234088
step: 162.0 HybridLinUCB 1.666097485253027
step: 162.0 FactorUCB w/o W 5.814653834212978
step: 162.0 Hyper_FactorUCB w/o W 7.870381868984787
step: 162.0 FactorUCB 0.6026078857497672
step: 162.0 DLinUCB 1.7957156162682397
step: 162.0 ColinUCB 1.6530580565041912
----------------------------
step: 163.0 LinUCB_ItemBased 8.901404104304891
step: 163.0 LinUCB_UserBased 1.619194568739392
step: 163.0 HybridLinUCB 1.6679524764696807
step: 163.0 FactorUCB w/o W 5.825181299182225
step: 163.0 Hyper_FactorUCB w/o W 7.880573985496065
step: 163.0 FactorUCB 0.6046906341613948
step: 163.0 DLinUCB 1.7952476469680605
step: 163.0 ColinUCB 1.6537571362444068
----------------------------
step: 164.0 LinUCB_ItemBased 8.900905046786317
step: 164.0 LinUCB_UserBased 1.620187145267679
step: 164.0 HybridLinUCB 1.6689676330725571
step: 164.0 FactorUCB w/o W 5.828654701641356
step: 164.0 Hyper_FactorUCB w/o W 7.884951679705476
step: 164.0 FactorUCB 0.6056143580303728
step: 164.0 DLinUCB 1.7975149562816384
step: 164.0 ColinUCB 1.6562356189599632
----------------------------
step: 165.0 LinUCB_ItemBased 8.898186252095718
step: 165.0 LinUCB_UserBased 1.6188081085200428
step: 165.0 HybridLinUCB 1.6678859929888736
step: 165.0 FactorUCB w/o W 5.8314281359548845
step: 165.0 Hyper_FactorUCB w/o W 7.892242036274958
step: 165.0 FactorUCB 0.6055479347660417
step: 165.0 DLinUCB 1.7969821673525377
step: 165.0 ColinUCB 1.6547782350251485
----------------------------
step: 166.0 LinUCB_ItemBased 8.904198878278006
step: 166.0 LinUCB_UserBased 1.6204335303926027
step: 166.0 HybridLinUCB 1.6695467636804608
step: 166.0 FactorUCB w/o W 5.8420494164013945
step: 166.0 Hyper_FactorUCB w/o W 7.910565408519024
step: 166.0 FactorUCB 0.6069425496437775
step: 166.0 DLinUCB 1.7980900409276943
step: 166.0 ColinUCB 1.65514627861149
----------------------------
step: 167.0 LinUCB_ItemBased 8.903765060240964
step: 167.0 LinUCB_UserBased 1.6218373493975904
step: 167.0 HybridLinUCB 1.6706325301204819
step: 167.0 FactorUCB w/o W 5.847439759036145
step: 167.0 Hyper_FactorUCB w/o W 7.918975903614458
step: 167.0 FactorUCB 0.6091867469879518
step: 167.0 DLinUCB 1.800301204819277
step: 167.0 ColinUCB 1.6569277108433735
----------------------------
step: 168.0 LinUCB_ItemBased 8.912157097886373
step: 168.0 LinUCB_UserBased 1.6226952480887422
step: 168.0 HybridLinUCB 1.6727627042422426
step: 168.0 FactorUCB w/o W 5.853845000749513
step: 168.0 Hyper_FactorUCB w/o W 7.928946184979763
step: 168.0 FactorUCB 0.6105531404587018
step: 168.0 DLinUCB 1.804227252286014
step: 168.0 ColinUCB 1.6594213761055314
----------------------------
step: 169.0 LinUCB_ItemBased 8.910919112170415
step: 169.0 LinUCB_UserBased 1.623119320720989
step: 169.0 HybridLinUCB 1.6725755995828988
step: 169.0 FactorUCB w/o W 5.861611798003873
step: 169.0 Hyper_FactorUCB w/o W 7.93937136898555
step: 169.0 FactorUCB 0.611798003873082
step: 169.0 DLinUCB 1.802174884552361
step: 169.0 ColinUCB 1.6588708476091167
----------------------------
step: 170.0 LinUCB_ItemBased 8.924228944246737
step: 170.0 LinUCB_UserBased 1.6255931198102016
step: 170.0 HybridLinUCB 1.6754151838671412
step: 170.0 FactorUCB w/o W 5.876631079478055
step: 170.0 Hyper_FactorUCB w/o W 7.956702253855279
step: 170.0 FactorUCB 0.6147686832740213
step: 170.0 DLinUCB 1.805011862396204
step: 170.0 ColinUCB 1.6623665480427046
----------------------------
Epoch [5/50], Loss: 3.1335
Epoch [10/50], Loss: 3.1371
Epoch [15/50], Loss: 3.1334
Epoch [20/50], Loss: 3.1307
Epoch [25/50], Loss: 3.1272
Epoch [30/50], Loss: 3.1265
Epoch [35/50], Loss: 3.1272
Epoch [40/50], Loss: 3.1275
Epoch [45/50], Loss: 3.1277
Epoch [50/50], Loss: 3.1293
step: 171.0 LinUCB_ItemBased 8.926559504497861
step: 171.0 LinUCB_UserBased 1.6251290370151894
step: 171.0 HybridLinUCB 1.6752691343459667
step: 171.0 FactorUCB w/o W 5.878189057661112
step: 171.0 Hyper_FactorUCB w/o W 7.962394927001917
step: 171.0 FactorUCB 0.6149535466745317
step: 171.0 DLinUCB 1.8046010912844712
step: 171.0 ColinUCB 1.6617018138917563
----------------------------
step: 172.0 LinUCB_ItemBased 8.936591809775429
step: 172.0 LinUCB_UserBased 1.6282107735212095
step: 172.0 HybridLinUCB 1.6787024805518862
step: 172.0 FactorUCB w/o W 5.889476001761339
step: 172.0 Hyper_FactorUCB w/o W 7.979451049464259
step: 172.0 FactorUCB 0.618229854689564
step: 172.0 DLinUCB 1.8084544253632762
step: 172.0 ColinUCB 1.665785997357992
----------------------------
step: 173.0 LinUCB_ItemBased 8.949306062819577
step: 173.0 LinUCB_UserBased 1.6296566837107378
step: 173.0 HybridLinUCB 1.6800584368151936
step: 173.0 FactorUCB w/o W 5.89890430971512
step: 173.0 Hyper_FactorUCB w/o W 7.994156318480643
step: 173.0 FactorUCB 0.6192841490138787
step: 173.0 DLinUCB 1.8092037983929876
step: 173.0 ColinUCB 1.6676406135865596
----------------------------
step: 174.0 LinUCB_ItemBased 8.952775356001162
step: 174.0 LinUCB_UserBased 1.6309212438244696
step: 174.0 HybridLinUCB 1.681342632955536
step: 174.0 FactorUCB w/o W 5.907294391165359
step: 174.0 Hyper_FactorUCB w/o W 8.005376344086022
step: 174.0 FactorUCB 0.6219122348154607
step: 174.0 DLinUCB 1.8100842778262134
step: 174.0 ColinUCB 1.6681197326358617
----------------------------
Traceback (most recent call last):
  File "D:\IIR\Server\PycharmProject\HyperBandit\RUN.py", line 229, in <module>
    with open("./result_log/resultdata_{}.pkl".format(str(util.get_time())), "wb") as f:
OSError: [Errno 22] Invalid argument: './result_log/resultdata_2023-04-05_02:10:18.pkl'

进程已结束,退出代码1




"""
"""

step: 24146.0 LinUCB 4.591748768472907
step: 24146.0 HybridLinUCB 3.643472906403941
step: 24146.0 HyperBandit 5.322044334975369
step: 24146.0 FactorUCB w/o W 4.92179802955665
step: 24146.0 FactorUCB 4.633825944170772
step: 24146.0 DLinUCB 2.9823481116584567
step: 24146.0 ColinUCB 3.019088669950739

"""


