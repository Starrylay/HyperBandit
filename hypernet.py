import torch
import torch.nn as nn
import config
from util import ReadFeatureFile, ParseLine
import util
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
import time as pytime
from torch.utils.data import RandomSampler
from Base_algorithm import randomStruct, Item, User
# from RUN import algresult
# 定义一个简单的 MLP 模型

# # Define your model
# class YourModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim):
#         super(YourModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         # Add other layers as needed

#     def forward(self, input_data):
#         embedded_data = self.embedding(input_data)
#         # Implement the forward pass with other layers as needed
#         return embedded_data

# # Example usage
# vocab_size = 10000  # Set according to your vocabulary size
# embedding_dim = 300  # Set according to the desired embedding dimension
# model = YourModel(vocab_size, embedding_dim)

# # Input data (replace this with your actual input)
# input_data = torch.LongTensor([1, 5, 7, 2, 9])

# # Forward pass
# output = model(input_data)
# print(output)

class MLP_learn_embedding(nn.Module):
    def __init__(self, hidden_dim, output_dim, vocab_size,embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, 4*hidden_dim)
        self.fc4 = nn.Linear(4*hidden_dim, 4*hidden_dim)
        self.fc5 = nn.Linear(4*hidden_dim, 4*hidden_dim)
        self.fc6 = nn.Linear(4*hidden_dim, 4*hidden_dim)
        self.fc7 = nn.Linear(4*hidden_dim, 2*hidden_dim)
        self.fc8 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, output_dim)
        #relu的斜率设为：0.01     
        self.relu = nn.ReLU()
        # self.ln = nn.LayerNorm(4*hidden_dim)

    def forward(self, x):
        x = self.embedding(x.long())
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, 4*hidden_dim)
        self.fc4 = nn.Linear(4*hidden_dim, 4*hidden_dim)
        self.fc5 = nn.Linear(4*hidden_dim, 4*hidden_dim)
        self.fc6 = nn.Linear(4*hidden_dim, 4*hidden_dim)
        self.fc7 = nn.Linear(4*hidden_dim, 2*hidden_dim)
        self.fc8 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, output_dim)
        #relu的斜率设为：0.01     
        self.relu = nn.ReLU()
        # self.ln = nn.LayerNorm(4*hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
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
        temperature = 1
        P_list_true = torch.softmax(reward_list_true/temperature, dim=1)
        P_list_pred = torch.softmax(reward_list_pred/temperature, dim=1)   
        P_list_pred_log = torch.log(P_list_pred)
        # element_wise 
        p_logp = torch.mul(P_list_true, P_list_pred_log)
        # 对每一行求和
        p_logp_sum_line = torch.sum(p_logp, dim=1)
        # batch中每个loss求和
        loss = - torch.sum(p_logp_sum_line)
        return loss

def onehot_encode(time_week, time_day):
    # onehot 编码
    # 创建一个包含三个类别的变量
    categories_week = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    categories_day = np.array(['morning', 'noon', 'afternoon', 'night', 'rest'])
    # 创建一个OneHotEncoder对象并拟合数据
    encoder_week = OneHotEncoder(sparse_output=False)
    encoder_day = OneHotEncoder(sparse_output=False)
    encoder_week.fit(categories_week.reshape(-1, 1))
    encoder_day.fit(categories_day.reshape(-1, 1))
    one_hot_week = encoder_week.transform(time_week.reshape(-1, 1)).flatten()
    one_hot_day = encoder_day.transform(time_day.reshape(-1, 1)).flatten()
    return one_hot_week, one_hot_day



#构建超网络输入
def hypernetwork_input_building(time,Timevectors, user_info, hypernet_setting):
    """
    Args:
        time(str): week period， day period
        user_id(int)
        UserFeatureVectors()
    Return:
        input_tensor: concatenate(week period one hot, day period one hot, user feature)jupyter
    """
    if hypernet_setting.time_embedding == "glove":
        
        time = time.strip('\n')
        time_line = Timevectors[time]
        input_tensor = torch.tensor(time_line, dtype=torch.float32) # 不要user_feature只要时间feature

    elif hypernet_setting.time_embedding == "onehot":
        time_list = time.strip('\n').split(" ")
        time_week = np.array(time_list[0])
        time_day = np.array(time_list[1])
        one_hot_week, one_hot_day = onehot_encode(time_week, time_day)
        time_line = np.concatenate([one_hot_week, one_hot_day], axis=0)
        input_tensor = torch.tensor(time_line, dtype=torch.float32)

    elif hypernet_setting.time_embedding == "polar":
        time_list = time.strip('\n').split(" ")
        time_week = np.array(time_list[0])
        time_day = np.array(time_list[1])
        week_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        block_list = ['morning', 'noon', 'afternoon', 'night', 'rest']
        week_index = week_list.index(time_week)
        block_index = block_list.index(time_day)
        # 对星期几进行编码
        day_sin = np.sin(2 * np.pi * week_index / 7.0 + 0.1*np.pi)
        day_cos = np.cos(2 * np.pi * week_index / 7.0 + 0.1*np.pi)
        
        # 对一天中的blcok进行编码
        block_sin = np.sin(2 * np.pi * block_index / 5.0 + 0.1*np.pi)
        block_cos = np.cos(2 * np.pi * block_index / 5.0 + 0.1*np.pi)
        time_line = np.array([day_sin, day_cos, block_sin, block_cos]).tolist()
        input_tensor = torch.tensor(time_line, dtype=torch.float32)

    elif hypernet_setting.time_embedding == "polar3":
        time_list = time.strip('\n').split(" ")
        time_week = np.array(time_list[0])
        time_day = np.array(time_list[1])
        week_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        block_list = ['morning', 'noon', 'afternoon', 'night', 'rest']
        week_index = week_list.index(time_week)
        block_index = block_list.index(time_day)
        if block_index in [0,1]:
            block_index = 0
        elif block_index in [2,3]:
            block_index = 1
        else:
            block_index = 2

        # 对星期几进行编码
        day_sin = np.sin(2 * np.pi * week_index / 7.0 + 0.1*np.pi)
        day_cos = np.cos(2 * np.pi * week_index / 7.0 + 0.1*np.pi)
        # 对一天中的blcok进行编码
        block_sin = np.sin(2 * np.pi * block_index / 3.0 + 0.1*np.pi)
        block_cos = np.cos(2 * np.pi * block_index / 3.0 + 0.1*np.pi)
        time_line = np.array([day_sin, day_cos, block_sin, block_cos]).tolist()
        input_tensor = torch.tensor(time_line, dtype=torch.float32)
    elif hypernet_setting.time_embedding == "polar1":
        time_list = time.strip('\n').split(" ")
        time_week = np.array(time_list[0])
        time_day = np.array(time_list[1])
        week_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        block_list = ['morning', 'noon', 'afternoon', 'night', 'rest']
        week_index = week_list.index(time_week)
        block_index = block_list.index(time_day)
        block_index = 0

        # 对星期几进行编码
        day_sin = np.sin(2 * np.pi * week_index / 7.0 + 0.1*np.pi)
        day_cos = np.cos(2 * np.pi * week_index / 7.0 + 0.1*np.pi)
        # 对一天中的blcok进行编码
        block_sin = np.sin(2 * np.pi * block_index / 1.0 + 0.1*np.pi)
        block_cos = np.cos(2 * np.pi * block_index / 1.0 + 0.1*np.pi)
        time_line = np.array([day_sin, day_cos, block_sin, block_cos]).tolist()
        input_tensor = torch.tensor(time_line, dtype=torch.float32)
    

    elif hypernet_setting.time_embedding == "learn":
        time_list = time.strip('\n').split(" ")
        time_week = np.array(time_list[0])
        time_day = np.array(time_list[1])
        week_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        block_list = ['morning', 'noon', 'afternoon', 'night', 'rest']
        week_index = week_list.index(time_week)
        block_index = block_list.index(time_day)
        time_line = [week_index*5 + block_index]
        input_tensor = torch.tensor(time_line, dtype=torch.int32)
    else:
        pass

    # weekdict = {'Mon':"Monday", 'Tue':"Tuesday", 'Wed':"Wednsday", 'Thu':"Thursday", 'Fri':"Friday", 'Sat':"Saturday", 'Sun':"Sunday"}
    # time_week = weekdict[time_week]

    if hypernet_setting.user_input:
        # user feature
        user_feature  = user_info.observed_feature
        user_feature = torch.tensor(user_feature[:hypernet_setting.input_dim],dtype=torch.float32)
        input_tensor = torch.concat([input_tensor, user_feature], axis=0)#注意类型提升*** learn ****的时候
    
    return input_tensor   



# 加载数据集
def dataload(UserFeatureVectors, algresult, hypernet_setting, Timevectors):
    """
    Args:
        UserFeatureVectors
        algresult: RUN.py 中一个实例，包含各种算法的近10000个事件的结果
    Return:
        dataloader(train_input_tensor, rewards_tensor, item_features_tensor)
        (train_input_tensor, armpool_shuffle_lists, label_shuffle_lists, armpool_shuffle_feature)
    """
    train_window = hypernet_setting.train_window
    user_ids = algresult.user_id[-train_window:]
    times = algresult.tim[-train_window:]
    #item_picked_ids = algresult.AlgPicked["Hypernet_FactorUCB"][-10000:]
    # reward_list = algresult.AlgReward["Hypernet_FactorUCB"][-10000:]
    armpool_shuffle_lists = algresult.Armpool[-train_window:]
    label_shuffle_lists = algresult.Labelpool[-train_window:]
    # train_input = []
    train_input_tensor = torch.tensor([])
    for i in range(len(user_ids)):

        user_id, tim = user_ids[i], times[i]
        user_info  = User(user_id, UserFeatureVectors[user_id])

        time_line_tensor = hypernetwork_input_building(tim, Timevectors, user_info,hypernet_setting)

        # time_list = tim.strip('\n').split(" ")
        # time_week = np.array(time_list[0])
        # time_day = np.array(time_list[1])
        # one_hot_week, one_hot_day = onehot_encode(time_week, time_day)
        # time_line = np.concatenate([one_hot_week, one_hot_day], axis=0)

        #user feature
        user_feature = UserFeatureVectors[user_id]
        user_feature = user_feature[:hypernet_setting.user_observed_dimension]
        user_feature_tensor = torch.Tensor(user_feature)
        input_line_tensor = torch.cat((time_line_tensor, user_feature_tensor), dim=0)
        train_input_tensor = torch.cat((train_input_tensor, input_line_tensor.unsqueeze(0)), dim=0)
        # train_input.append(np.concatenate([time_line, user_feature], axis=0))#虽然有userfeature但并不是作为输入
    # train_input_onearray = np.concatenate(train_input, axis=0)
    # train_input_tensor = torch.Tensor(train_input_onearray)
    # train_input_tensor = train_input_tensor.view(-1, hypernet_setting.input_dim + hypernet_setting.user_observed_dimension)

    
    #根据itemid找feature
    armpool_shuffle_feature = []
    for i in range(len(armpool_shuffle_lists)):
        pool_line_feature = []
        for j in range(len(armpool_shuffle_lists[0])):
            item_context_feature = algresult.algorithms["HyperBandit"].ItemStruct_total[armpool_shuffle_lists[i][j]].context_feature
            pool_line_feature.append(item_context_feature)
        armpool_shuffle_feature.append(pool_line_feature)

    armpool_shuffle_feature_tensor = torch.tensor(armpool_shuffle_feature)
    armpool_shuffle_lists = torch.tensor(armpool_shuffle_lists)
    label_shuffle_lists = torch.tensor(label_shuffle_lists)
    
    #构造数据集
    train_dataset = TensorDataset(train_input_tensor, armpool_shuffle_lists, label_shuffle_lists, armpool_shuffle_feature_tensor)
    #构造 DataLoader
    dataloader = DataLoader(train_dataset, batch_size=hypernet_setting.batch_size, shuffle=True)
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

def trainnet(args, dataloader, criterion, optimizer, mlp, device, hypernet_setting):
    traintime = 0
    best_loss = float('inf')
    patience = 1
    epochcount = 0
    iter_no_improve = 0

    if hypernet_setting.sample_rate < 1 :
        num_samples = int(hypernet_setting.train_window * hypernet_setting.sample_rate) # 指定要采样的样本数量
        sampler = RandomSampler(dataloader.dataset, num_samples=num_samples)
        # 创建带有 RandomSampler 的 DataLoader
        dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=sampler)
    
    for epoch in range(hypernet_setting.num_epochs):
        epochloss = 0
        for i, (input_batch, armpool_id_batch, labelpool_batch, armpool_feature_batch) in enumerate(dataloader):
            
            input_batch = input_batch.to(device)
            # 将数据转换为张量并展平  不用展平
            # inputs_batch = input_batch.view(-1, config.hypernet_setting.input_dim + config.FactorUCB_setting.user_observed_dimension).to(device)
            labelpool_batch = torch.unsqueeze(labelpool_batch, dim=2).to(device)
            # reward = labels_batch.to(device)

            armpool_feature_batch = armpool_feature_batch.to(device)
            #user
            user_feature = input_batch[:,-hypernet_setting.user_observed_dimension:].to(device)
            # 正向传播计算输出
            outputs = mlp(input_batch[:,:hypernet_setting.input_dim])

            if hypernet_setting.time_embedding == "learn":
                outputs = outputs.squeeze(dim=1)
        
            Theta_Matrix = util.output_transform(outputs,hypernet_setting).double()
            # outputs.view((-1, config.FactorUCB_setting.item_dimension, config.FactorUCB_setting.user_observed_dimension)).double()                         
            # 不更新 item and user feature
            armpool_feature_batch.requires_grad = False
            user_feature.requires_grad = False
            # 计算损失并反向传播
            user_feature = torch.unsqueeze(user_feature, dim=2).double()
            # batch 矩阵乘法得到预估收益
            temp = torch.matmul(armpool_feature_batch, Theta_Matrix)
            mean_pta_batch = torch.matmul(temp, user_feature) #batchsize   armpoolsize  
            # mean_pta_batch = mean_pta_batch.squeeze()
            # reward = reward.squeeze()
            start_time = pytime.time()
            loss = criterion(mean_pta_batch, labelpool_batch)/hypernet_setting.batch_size
            optimizer.zero_grad()    
            loss.backward()  
            traintime += pytime.time() - start_time   
            epochloss += loss.item() 
            optimizer.step()

        epochloss = epochloss/len(dataloader)
        epochcount = epoch + 1
        if epochloss < best_loss:
            best_loss = epochloss
            iter_no_improve = 0
        else:
            iter_no_improve += 1
            if iter_no_improve >= patience:
                print("Early stopping!")
                break
        # if best_loss - epochloss <= 0.0003:
        #     print("Early stopping!")
        #     break
        # else:
        #     best_loss = epochloss
        #     iter_no_improve = 0
        # 输出训练过程信息
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, hypernet_setting.num_epochs, epochloss))
    return traintime, epochcount

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
