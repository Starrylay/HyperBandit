import numpy as np
import config
import util
import random
import torch
from Base_algorithm import randomStruct, Item, User
import FactorUCB
import LinUCB
import ColinUCB
import Baseline.dLinUCB as dLinUCB
import Baseline.HybridLinUCB as HybridLinUCB
import Baseline.factorUCB as factorUCB
import Baseline.ADTS as ADTS
import matplotlib.pyplot as plt
from result import AlgResult
import hypernet
import pickle
import torch_backend
import copy

#构建超网络输入
def hypernetwork_input_building(time, user_id, UserFeatureVectors):
    """
    Args:
        time(str): week period， day period
        user_id(int)
        UserFeatureVectors()
    Return:
        input_tensor: concatenate(week period one hot, day period one hot, user feature)jupyter
    """
    time_list = time.strip('\n').split(" ")
    time_week = np.array(time_list[0])
    time_day = np.array(time_list[1])
    one_hot_week, one_hot_day = hypernet.onehot_encode(time_week, time_day)
    time_line = np.concatenate([one_hot_week, one_hot_day], axis=0)
    # user feature
    if user_id in UserFeatureVectors:
        user_feature = UserFeatureVectors[user_id]
    else:
        user_feature = np.random.rand(len(UserFeatureVectors[1]))
    user_feature = user_feature[:config.FactorUCB_setting.user_observed_dimension]
    input_mlp = np.concatenate([time_line, user_feature], axis=0)
    input_tensor = torch.Tensor(input_mlp)
    return input_tensor


#global 实例
algresult = AlgResult()
algresult.algorithms["LinUCB_ItemBased"] = LinUCB.LinUCBAlgorithm_ItemBased()
# algresult.algorithms["LinUCB"] = LinUCB.LinUCBAlgorithm_UserBased()
# algresult.algorithms["HybridLinUCB"] = HybridLinUCB.Hybrid_LinUCBAlgorithm()
# algresult.algorithms["FactorUCB N Hyper"] = FactorUCB.FactorUCBAlgorithm() # 实例1
for item_observed_dim in range(0,31):
    algresult.algorithms["HyperBandit w/o Hyper {}".format(item_observed_dim)] = FactorUCB.FactorUCBAlgorithm(item_observed_dim)
#algresult.algorithms["HyperBandit"] = FactorUCB.FactorUCBAlgorithm() # 实例2
# algresult.algorithms["FactorUCB w/o W"] = factorUCB.FactorUCBAlgorithm(W_type = "None")
# algresult.algorithms["FactorUCB"] = factorUCB.FactorUCBAlgorithm(W_type = "Have")
# algresult.algorithms["DLinUCB"] = dLinUCB.DLinUCBAlgorithm()
# algresult.algorithms["ColinUCB w/o W"] = ColinUCB.CoLinUCBAlgorithm(W_type = "None")
# algresult.algorithms["ColinUCB"] = ColinUCB.CoLinUCBAlgorithm(W_type = "Have")
# algresult.algorithms["ADTS"] = ADTS.AdaptiveThompson()

for alg_name, alg in algresult.algorithms.items():
    algresult.AlgReward[alg_name] = []
    algresult.AlgPicked[alg_name] = []
    algresult.AlgRegret[alg_name] = []
    algresult.BatchCumlateRegret[alg_name] = []
    algresult.AlgRewardRatio_vsRandom[alg_name] = []

# 超网络参数
input_dim =  config.hypernet_setting.input_dim # 输入层维度
hidden_dim = config.hypernet_setting.hidden_dim # 隐藏层维度
output_dim = config.hypernet_setting.output_dim
learning_rate = config.hypernet_setting.learning_rate # 学习3+4率
num_epochs = config.hypernet_setting.num_epochs # epoch
train_window = config.hypernet_setting.train_window


if __name__ == '__main__':
    # if config.hypernet_setting.is_train:

    if torch.cuda.is_available():
        device = torch.device("cuda")  # 将模型移动到GPU上
        print("GPU is available.")
    else:
        device = torch.device("cpu")  # 使用CPU
        print("GPU is not available.")
    # 创建 MLP 模型实例
    mlp = hypernet.MLP(input_dim, hidden_dim, output_dim).to(device)

    # mlptest  = hypernet.MLP(input_dim, hidden_dim, output_dim).to(device)
    mlptest = copy.deepcopy(mlp)
    
    criterion = hypernet.ListNetLoss()  # 自定义损失函数
    # criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(mlp.parameters())  # 随机梯度下降优化器 lr=learning_rate # momentum=0.9, weight_decay=0.001

    # 从文件中读取 user item feature
    UserFeatureVectors, ItemFeatureVectors, ClusterFeatureVectors = util.read_in_observed_feature(config.general_setting.dataset)
    cluster_map = util.read_label(config.general_setting.dataset)

    fileName = None
    if config.general_setting.dataset == "lastfm":
        fileName = "./dataset/lastfm/hyper_processed_events_noshuffled.dat" # 用户 时间 armpool
    if config.general_setting.dataset == "NYC":
        fileName = "./dataset/foursquare/NYC/Events_NYC.dat"  # 用户 时间 armpool
    if config.general_setting.dataset == "TKY":
        fileName = "./dataset/foursquare/TKY/Events_TKY.dat"  # 用户 时间 armpool
    random_alg = randomStruct()
    # 初始化Theata
    Theta_init = np.identity(config.FactorUCB_setting.item_dimension)
    #np.random.rand(config.FactorUCB_setting.item_dimension,config.FactorUCB_setting.item_dimension)

    #, config.FactorUCB_setting.user_observed_dimension 保持行列一样
    # Theta_init = 0.5 * torch.ones((config.FactorUCB_setting.item_dimension, config.FactorUCB_setting.user_observed_dimension))
    # nx = torch_backend.get_torch_backend(Theta_init)
    # Theta_init1 = 0.5 * nx.ones((config.FactorUCB_setting.item_dimension, config.FactorUCB_setting.user_observed_dimension))
    # Theta_init2 = 0.5 * torch.ones((config.FactorUCB_setting.item_dimension, config.FactorUCB_setting.user_observed_dimension))
    count = 0
    with open(fileName, "r") as f:
        f.readline()
        for i, line in enumerate(f, 1):
            # 解析每一条event
            # if i > 10000:
            #     break
            user_id, time, pool_item_ids = util.ParseLine(line)
            pool_item_ids = pool_item_ids[:config.general_setting.pool_size]# 只取前k个


            if config.general_setting.dataset == "NYC" and user_id in UserFeatureVectors:
                user_feature = UserFeatureVectors[user_id]
                cluster_id = cluster_map[user_id]# 映射成类别id
                cluster_feature = ClusterFeatureVectors[cluster_id]
                cluster_user_info = User(cluster_id, cluster_feature)
            elif config.general_setting.dataset == "TKY" and user_id in cluster_map:
                user_feature = UserFeatureVectors[user_id]
                cluster_id = cluster_map[user_id]# 映射成类别id
                cluster_feature = ClusterFeatureVectors[cluster_id]
                cluster_user_info = User(cluster_id, cluster_feature)
            elif config.general_setting.dataset == "lastfm" and user_id in cluster_map :
                user_feature = UserFeatureVectors[user_id]
                cluster_id = cluster_map[user_id]  # 映射成类别id
                cluster_feature = ClusterFeatureVectors[cluster_id]
                cluster_user_info = User(cluster_id, cluster_feature)
            else:
                # print("新用户")
                # user_feature = None
                # cluster_id = None  # 映射成类别id
                # cluster_feature = ClusterFeatureVectors[cluster_id]
                # cluster_user_info = User(cluster_id, cluster_feature)
                continue
            # 将 user id 和 observed feature 打包
            user_info = User(user_id, user_feature)

            # if user_id >=config.general_setting.user_num:
            #     continue
            # 记录信息，准备 data for hypernet
            algresult.user_id.append(user_id)
            algresult.tim.append(time)
            #确定实际选择的 item
            item_chosen = pool_item_ids[0]  # pool中第一个是用户实际选择的action
            
            # 将armpool中将item observd feature 与对应的item id 打包
            pool_item_info = []
            #存储poolids and labels 并打乱
            flag = 0
            for item_id in pool_item_ids:
                if item_id in ItemFeatureVectors:
                    item_feature_observed = ItemFeatureVectors[item_id]
                else:# 避免没有profile的新商品
                    # print("新商品")
                    flag = 1
                    item_feature_observed = (np.random.rand(len(ItemFeatureVectors[1])) - 0.5) * 0.5
                    #添加到 ItemFeatureVectors 避免下次遇到继续随机生成
                    ItemFeatureVectors[item_id] = item_feature_observed
                pool_item_info.append(Item(item_id, item_feature_observed))
            if flag == 1:
                continue
            # if config.hypernet_setting.is_train:
            #     # 构造input
            #
            # baseline 随机选择策略
            random_id = random.choice(pool_item_ids)  # pool中随机挑选一个id
            # baseline 随机选择策略的accumulated reward        
            if random_id == item_chosen:
                random_alg.accumulated_reward += 1
            else:
                random_alg.accumulated_regret += 1 
            #遍历每一种策略
           
            for alg_name, alg in algresult.algorithms.items():
                if alg_name == "HyperBandit" :
                    input_tensor = hypernetwork_input_building(time, user_id, UserFeatureVectors).to(device)
                    input_tensor = input_tensor[:config.hypernet_setting.input_dim]
                    # 利用超网络生成Theta
                    output = mlp(input_tensor)
                    Theta_Matrix = output.reshape(config.FactorUCB_setting.item_dimension,
                                                  config.FactorUCB_setting.user_observed_dimension)
                    Theta_hypernet = Theta_Matrix.detach().cpu().numpy()
                    Theta = Theta_hypernet

                # elif alg_name == "FactorUCB N Hyper" :
                #     input_tensor = hypernetwork_input_building(time, user_id, UserFeatureVectors).to(device)
                #     input_tensor = input_tensor[:config.hypernet_setting.input_dim]
                #     # 利用超网络生成Theta
                #     output = mlptest(input_tensor)
                #     Theta_Matrix = output.reshape(config.FactorUCB_setting.item_dimension,
                #                                   config.FactorUCB_setting.user_observed_dimension)
                #     Theta_hypernet = Theta_Matrix.detach().cpu().numpy()
                #     Theta = Theta_hypernet

                else:
                    Theta = Theta_init
                if config.general_setting.iscluster and alg_name in ["ColinUCB", 'ColinUCB w/o W', "FactorUCB","FactorUCB w/o W"]: # "DLinUCB"
                    user_info = cluster_user_info
                random.shuffle(pool_item_info)
                #算法选择出 item
                item_info_picked = alg.decide(pool_item_info, user_info, Theta)
                #计算 reward
                if alg_name == "HyperBandit":
                    label = np.zeros(len(pool_item_ids))
                    if item_info_picked.id == item_chosen:
                        reward = 1
                        label[0] = 3
                    else:
                        reward = 0
                        position = pool_item_ids.tolist().index(item_info_picked.id)#找到等于chosen id 的位置
                        label[position] = -3
                    zipped = list(zip(pool_item_ids, label))
                    random.shuffle(zipped)
                    pool_item_ids_shuffle, label_shuffle = zip(*zipped)
                    algresult.Armpool.append(pool_item_ids_shuffle)
                    algresult.Labelpool.append(label_shuffle)

                else:
                    if item_info_picked.id == item_chosen:
                        reward = 1
                    else:
                        reward = 0
                #根据 reward 更新参数
                if config.test_setting.isupdate:
                    alg.updateParameters(item_info_picked, reward, user_info, Theta)

                #统计各个算法 reward,regreat,itemid
                # if reward == -1: 
                #     reward = 0
                algresult.AlgReward[alg_name].append(reward)
                algresult.AlgRegret[alg_name].append(1 - reward)
                algresult.AlgPicked[alg_name].append(item_info_picked.id)

                count = len(algresult.AlgReward[alg_name])
                if count % config.test_setting.show_window == 0:
                    # 每 1000 条 event 计算 accumulated regret
                    algresult.BatchCumlateRegret[alg_name].append(sum(algresult.AlgRegret[alg_name]))

                    #如果随机选择策略收益不为0,即分母不为0
                    if random_alg.accumulated_reward != 0:
                        relative_accumulated_reward = (count - algresult.BatchCumlateRegret[alg_name][-1]) / (1.0 * random_alg.accumulated_reward)
                    else:
                        relative_accumulated_reward = 0
                    algresult.AlgRewardRatio_vsRandom[alg_name].append(relative_accumulated_reward)
                    # 每1000条event 输出一下累计收益
                    print("step:", count/config.test_setting.show_window, alg_name,  relative_accumulated_reward)
            if count % config.test_setting.show_window == 0:
                print("----------------------------")

            if count % train_window == 0 and config.hypernet_setting.is_train:
                #遍历完所有算法之后，更新超网络

                dataloader = hypernet.dataload(UserFeatureVectors, algresult)
                hypernet.trainnet(dataloader, criterion, optimizer, mlp, device)
                #保存模型参数
                # 将模型参数保存为文件
                with open("./model_log/model_trained_{}.pkl".format(str(count/train_window)), "wb") as f:
                    pickle.dump(mlp, f)

        iteration = [x * config.test_setting.show_window for x in range(int(count/config.test_setting.show_window))]
        for alg_name, alg in algresult.algorithms.items():
            plt.plot(iteration, algresult.AlgRewardRatio_vsRandom[alg_name], label=alg_name)
        plt.xlabel('iterations')
        plt.ylabel('Normalized Accumulated Payoff')
        plt.legend()
        plt.savefig("./result_log/Payoff_{}_{}.png".format(config.general_setting.dataset, util.get_time()))
        plt.show()

        for alg_name, alg in algresult.algorithms.items():
            plt.plot(iteration, algresult.BatchCumlateRegret[alg_name], label=alg_name)
        plt.xlabel('iterations')
        plt.ylabel('Accumulated Regret')
        plt.legend()
        plt.savefig("./result_log/regret_{}_{}.png".format(config.general_setting.dataset, util.get_time()))
        plt.show()

    with open("./result_log/resultdata_{}_{}.pkl".format(config.general_setting.dataset, str(util.get_time())), "wb") as f:
        pickle.dump(algresult, f)






