import torch
import torch.nn as nn
import config
from util import ReadFeatureFile, ParseLine
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
# from RUN import algresult
# 定义一个简单的 MLP 模型

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# # 定义数据维度和模型超参数
# input_dim = 7 + 5 + config.FactorUCB_setting.user_observed_dimension # 输入层维度
#
# hidden_dim = 256 # 隐藏层维度
# output_dim = config.FactorUCB_setting.user_observed_dimension * config.FactorUCB_setting.item_dimension # 输出层 theta的向量展开
# learning_rate = 0.01 # 学习率
# num_epochs = 50 # epoch
# # 创建 MLP 模型实例
# mlp = MLP(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
class MyCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, reward, context, user_feature):
        #reshape Theta
        context.requires_grad = False
        user_feature.requires_grad = False
        context = torch.unsqueeze(context, dim=1).double()
        user_feature = torch.unsqueeze(user_feature, dim=2).double()
        Theta_Matrix = outputs.view((-1, config.FactorUCB_setting.item_dimension, config.FactorUCB_setting.user_observed_dimension)).double()
        #batch 矩阵乘法
        temp = torch.bmm(context, Theta_Matrix)
        reward_estimate = torch.bmm(temp, user_feature)
        # reward_estimate = torch.sigmoid(reward_estimate)
        reward_estimate = reward_estimate.squeeze()
        reward = reward.squeeze()
        squared_difference = torch.square(reward_estimate - reward)
        mean_squared_difference = torch.mean(squared_difference)
        return mean_squared_difference
    
class ListNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
  
    def forward(self, reward_list_pred, reward_list_true ):
        """
        Args: 
            reward_list_true 是armpool中每个item的真实reward eg. [1,0,0,0,0...] (batch 中每一行是armpool)
            reward_list_pred 是armpool中每个item的预测分数 eg.[0.234,0.159,-0.178...]
        Return:
            loss 是一个标量表示两个概率分布的交叉熵
        """
        #首先归一化每个list，得到概率
        temperature = 0.5
        P_list_true = torch.softmax(reward_list_true/temperature, dim=1)
        P_list_pred = torch.softmax(reward_list_pred/temperature, dim=1)   
        P_list_pred_log = torch.log(P_list_pred)
        # element_wise 
        p_logp = torch.mul(P_list_true, P_list_pred_log)
        # 对每一行求和
        p_logp_sum_line = torch.sum(p_logp, dim=1)
        # batch中每个loss求和
        loss = - torch.sum(p_logp_sum_line)/config.hypernet_setting.batch_size
        return loss

def onehot_encode(time_week, time_day):
    # onehot 编码
    # 创建一个包含三个类别的变量
    categories_week = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    categories_day = np.array(['rest', 'morning', 'noon', 'afternoon', 'night', 'rest'])
    # 创建一个OneHotEncoder对象并拟合数据
    encoder_week = OneHotEncoder(sparse=False)
    encoder_day = OneHotEncoder(sparse=False)
    encoder_week.fit(categories_week.reshape(-1, 1))
    encoder_day.fit(categories_day.reshape(-1, 1))
    one_hot_week = encoder_week.transform(time_week.reshape(-1, 1)).flatten()
    one_hot_day = encoder_day.transform(time_day.reshape(-1, 1)).flatten()
    return one_hot_week, one_hot_day
# 加载数据集
def dataload(UserFeatureVectors, algresult):
    """
    Args:
        UserFeatureVectors
        algresult: RUN.py 中一个实例，包含各种算法的近10000个事件的结果
    Return:
        dataloader(train_input_tensor, rewards_tensor, item_features_tensor)
        (train_input_tensor, armpool_shuffle_lists, label_shuffle_lists, armpool_shuffle_feature)
    """
    train_window = config.hypernet_setting.train_window
    user_ids = algresult.user_id[-train_window:]
    times = algresult.tim[-train_window:]
    #item_picked_ids = algresult.AlgPicked["Hypernet_FactorUCB"][-10000:]
    # reward_list = algresult.AlgReward["Hypernet_FactorUCB"][-10000:]
    armpool_shuffle_lists = algresult.Armpool[-train_window:]
    label_shuffle_lists = algresult.Labelpool[-train_window:]
    train_input = []
    for i in range(len(user_ids)):
        user_id, tim = user_ids[i], times[i]
        time_list = tim.strip('\n').split(" ")
        time_week = np.array(time_list[0])
        time_day = np.array(time_list[1])
        one_hot_week, one_hot_day = onehot_encode(time_week, time_day)
        time_line = np.concatenate([one_hot_week, one_hot_day], axis=0)
        #user feature
        user_feature = UserFeatureVectors[user_id]
        user_feature = user_feature[:config.FactorUCB_setting.user_observed_dimension]

        train_input.append(np.concatenate([time_line, user_feature], axis=0))

        # train_input.append(time_line)
    train_input_tensor = torch.Tensor(train_input)

    
    #根据itemid找feature
    armpool_shuffle_feature = []
    for i in range(len(armpool_shuffle_lists)):
        pool_line_feature = []
        for j in range(len(armpool_shuffle_lists[0])):
            item_context_feature = algresult.algorithms["HyperBandit"].ItemStruct_total[armpool_shuffle_lists[i][j]].context_feature
            pool_line_feature.append(item_context_feature)
        armpool_shuffle_feature.append(pool_line_feature)
    armpool_shuffle_feature_temsor = torch.tensor(armpool_shuffle_feature)

    armpool_shuffle_lists = torch.tensor(armpool_shuffle_lists)
    label_shuffle_lists = torch.tensor(label_shuffle_lists)
    
    #构造数据集
    train_dataset = TensorDataset(train_input_tensor, armpool_shuffle_lists, label_shuffle_lists, armpool_shuffle_feature_temsor)
    #构造 DataLoader
    dataloader = DataLoader(train_dataset, batch_size=config.hypernet_setting.batch_size, shuffle=False)
    return dataloader


# def getProb(self, alpha, pool_items_info, user_info, Theta): 
#     items_context = []
#     Psis = []
#     Psis_inv = []
#     for i in range(len(pool_items_info)):
#         ItemStruct_current = self.ItemStruct_total[pool_items_info[i].id]
#         item_context_feature = ItemStruct_current.get_context_feature()
#         items_context.append(item_context_feature)

#     user_feature = user_info.observed_feature[:config.FactorUCB_setting.user_observed_dimension]# 截取合适长度

#     # 计算整个armpool的mean
#     temps = np.matmul(items_context, Theta)
#     means = np.matmul(temps, user_feature)
#     return means 

# 训练模型
def trainnet(dataloader, criterion, optimizer, mlp, device):
    loss = 0
    for epoch in range(config.hypernet_setting.num_epochs):
        for i, (input_batch, armpool_id_batch, labelpool_batch, armpool_feature_batch) in enumerate(dataloader):
            # 将数据转换为张量并展平
            inputs_batch = input_batch.view(-1, config.hypernet_setting.input_dim).to(device)
            labelpool_batch = torch.unsqueeze(labelpool_batch, dim=2).to(device)
            # reward = labels_batch.to(device)

            armpool_feature_batch = armpool_feature_batch.to(device)
            #user
            user_feature = input_batch[:,-config.FactorUCB_setting.user_observed_dimension:].to(device)
            # 正向传播计算输出

            outputs = mlp(inputs_batch[:,:config.hypernet_setting.input_dim])
            Theta_Matrix = outputs.view((-1, config.FactorUCB_setting.item_dimension,
                                         config.FactorUCB_setting.user_observed_dimension)).double()
            # 不更新 item and user feature
            armpool_feature_batch.requires_grad = False
            user_feature.requires_grad = False
            # 计算损失并反向传播
            # context = torch.unsqueeze(context, dim=1).double()
            user_feature = torch.unsqueeze(user_feature, dim=2).double()
           
            # batch 矩阵乘法得到预估收益
            temp = torch.matmul(armpool_feature_batch, Theta_Matrix)
            mean_pta_batch = torch.matmul(temp, user_feature) #batchsize   armpoolsize  

            # mean_pta_batch = mean_pta_batch.squeeze()
            # reward = reward.squeeze()
            loss = criterion(mean_pta_batch, labelpool_batch)

            # grads = torch.autograd.grad(loss, mlp.parameters())
            # print("查看梯度链是否存在",grads)
            # break
            #loss = criterion(outputs, reward, context, user_feature)
            optimizer.zero_grad()
            loss.backward()

            # for name, param in mlp.named_parameters():
            #     if param.requires_grad:
            #         print("更新前", name, param.data)

            optimizer.step()
            # for name, param in mlp.named_parameters():
            #     if param.requires_grad:
            #         print("更新后", name, param.data)

        # 输出训练过程信息
        if (epoch+1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, config.hypernet_setting.num_epochs, loss.item()))


# if __name__ == '__main__':
#
#     criterion = MyCustomLoss()  # 自定义损失函数
#     optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)  # 随机梯度下降优化器
#
#     #读取文件
#     UserFeatureVectors = {}
#     ItemFeatureVectors = {}
#     fileName = ""
#     if config.general_setting.dataset == "foursquare":
#         UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/User_FeatureVectors.dat")  # 读入userfeature
#         ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/POI_Cat_FeatureVectors.dat")  # 读入itemfeature
#         fileName = "./dataset/foursquare/Events_NYC.dat"  # 用户 时间 armpool
#     elif config.general_setting.dataset == "lastfm":
#         UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/User_FeatureVectors.dat")  # 读入userfeature
#         ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/POI_Cat_FeatureVectors.dat")  # 读入itemfeature
#         fileName = "./dataset/foursquare/Events_NYC.dat"  # 用户 时间 armpool
#
#     dataloader = dataload(user_id_list, tim_list, item_picked_list, reward_list)
#     trainnet(dataloader)
#
#     output = mlp(inputs)



    # data_week = np.array(['Mon'])
    # data_day = np.array(['morning'])
    #
    # one_hot_week = encoder_week.transform(data_week.reshape(-1, 1))
    # one_hot_day = encoder_day.transform(data_day.reshape(-1, 1))
