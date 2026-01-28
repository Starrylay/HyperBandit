import numpy as np
import config
import util
import random
import torch
from Base_algorithm import randomStruct, Item, User

from matplotlib.transforms import Bbox
import hypernet
import pickle
import copy
import time as pytime
import json
from datetime import datetime
import os

# 固定随机种子
seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def run_rec(args, algresult, general_setting, hypernet_setting):

    if torch.cuda.is_available():
        # 指定要使用的GPU设备索引，比如第一个GPU
        gpu_index = args.gpu
        device = torch.device(f"cuda:{gpu_index}")
        print(f"GPU {gpu_index} is available.")
    else:
        device = torch.device("cpu")  # 如果没有GPU，则使用CPU
        print("GPU is not available.")

    dataset = general_setting.dataset
    # 创建 MLP 模型实例
    torch.cuda.empty_cache()
    if hypernet_setting.time_embedding == "learn":
        mlp = hypernet.MLP_learn_embedding(hypernet_setting.hidden_dim, hypernet_setting.output_dim, hypernet_setting.vocab_size, hypernet_setting.embedding_dim).to(device)
    else:
        mlp = hypernet.MLP(hypernet_setting.hidden_dim, hypernet_setting.output_dim, hypernet_setting.input_dim).to(device)
    mlptest = copy.deepcopy(mlp)
    criterion = hypernet.ListNetLoss()  
    optimizer = torch.optim.Adam(mlp.parameters())  # y优化器 lr=learning_rate # momentum=0.9, weight_decay=0.001
    # 从文件中读取 user item feature

    UserFeatureVectors, ItemFeatureVectors, ClusterFeatureVectors, Timevectors = util.read_in_observed_feature(dataset, hypernet_setting.feature)
    fileName = None
    if dataset == "lastfm":
        fileName = "./dataset/lastfm/hyper_processed_events_noshuffled.dat" # 用户 时间 armpool
    if dataset == "NYC":
        # fileName = "./dataset/foursquare/NYC/Events_NYC.dat"  # 用户 时间 armpool
        if hypernet_setting.warm_start:
            fileName = "events/Events_NYC_merged_48735.dat" 
        else:
            fileName = "events/Events_NYC_delete_each7.dat"  # 用户 时间 armpool
        cluster_map = util.read_label(dataset)
        warmcount = 48735

    if dataset == "TKY":
        if hypernet_setting.warm_start:
            fileName = "events/Events_TKY_merged_103185.dat"#
        else:
            fileName = "events/Events_TKY_delete_each7.dat"  # 用户 时间 armpool
        cluster_map = util.read_label(dataset)
        warmcount = 103185

    if dataset == "kuai":
        if hypernet_setting.warm_start:
            fileName = "events/Events_kuai_merged_34021.dat"
        else:
            fileName = "events/Events_kuai_last16_17week.dat"  # 用户 时间 armpool
        cluster_map = util.read_label(dataset)
        warmcount = 34021

    random_alg = randomStruct()
    # 初始化Theata
    Theta_init = 1 / general_setting.item_dimension * np.identity(general_setting.item_dimension)
    count = 0
    rec_time = 0

    with open(fileName, "r") as f:
        # with open(fileName, "r") as f_Bandit:
        for i, line in enumerate(f, 1):    
            # 解析每一条event
            if i<48735-args.llm_dada_size:  
                 continue
            # if i > 1000:
            #     break
            if i % 100 == 0:
                print("step:", i)
            rec_tim_start = pytime.time()
            user_id, time, pool_item_ids = util.ParseLine(line,dataset)
            # import ipdb; ipdb.set_trace()
            pool_item_ids = pool_item_ids[:general_setting.pool_size]# 只取前k个
            # import ipdb; ipdb.set_trace()
            if dataset == "NYC" and user_id in UserFeatureVectors :
                user_feature = UserFeatureVectors[user_id]
                cluster_id = cluster_map[user_id]# 映射成类别id
                cluster_feature = ClusterFeatureVectors[cluster_id]
                cluster_user_info = User(cluster_id, cluster_feature)
            elif dataset == "TKY" and user_id in UserFeatureVectors:
                user_feature = UserFeatureVectors[user_id]
                cluster_id = cluster_map[user_id]# 映射成类别id
                cluster_feature = ClusterFeatureVectors[cluster_id]
                cluster_user_info = User(cluster_id, cluster_feature)
            elif dataset == "lastfm" and user_id in UserFeatureVectors :
                user_feature = UserFeatureVectors[user_id]
                cluster_id = cluster_map[user_id]  # 映射成类别id
                cluster_feature = ClusterFeatureVectors[cluster_id]
                cluster_user_info = User(cluster_id, cluster_feature)
            elif dataset == "kuai" and user_id in UserFeatureVectors:
                user_feature = UserFeatureVectors[user_id]
                cluster_id = cluster_map[user_id]# 映射成类别id
                cluster_feature = ClusterFeatureVectors[cluster_id]
                cluster_user_info = User(cluster_id, cluster_feature)
            else:
                print("新用户")
                print(user_id)
                # 从UserFeatureVectors中采样10个已有用户求平均向量
                user_feature = np.zeros(len(UserFeatureVectors[1]))
                sampled_keys = random.sample(list(UserFeatureVectors.keys()), 10)
                for i in sampled_keys:
                    user_feature += UserFeatureVectors[i]
                user_feature = user_feature/10
                UserFeatureVectors[user_id] = user_feature
                
                if "HybridLinUCB" in algresult.algorithms.keys():
                    algresult.algorithms["HybridLinUCB"].update_user(user_id, user_feature)

            original_user_info = User(user_id, user_feature)

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
                    print("新商品")   
                    print(item_id)  
                    print(i)              
                    flag = 1
                    break

                pool_item_info.append(Item(item_id, item_feature_observed))
            if flag == 1:
                break
            
            algresult.user_id.append(user_id)
            algresult.tim.append(time)
            # if config.hypernet_setting.is_train:
            #     # 构造input
            # baseline 随机选择策略
            random_id = random.choice(pool_item_ids)  # pool中随机挑选一个id
            # baseline 随机选择策略的accumulated reward        
            if random_id == item_chosen:
                random_alg.accumulated_reward += 1
            else:
                random_alg.accumulated_regret += 1 
            #遍历每一种策略
                
            for alg_name, alg in algresult.algorithms.items():
                if alg_name == "HyperBandit": #and hypernet_setting.is_train
                    input_tensor = hypernet.hypernetwork_input_building(time,Timevectors,original_user_info, hypernet_setting).to(device)
                    input_tensor = input_tensor[:hypernet_setting.input_dim]

                    with torch.no_grad():
                        output = mlp(input_tensor)
                    #numpy 增加维度
                    if hypernet_setting.time_embedding == "learn":
                        output = output.detach().cpu().numpy().reshape(-1)
                    else:
                        output = output.detach().cpu().numpy()

                    Theta_Matrix = util.output_transform(output, hypernet_setting )
                    Theta = Theta_Matrix
                    

                elif alg_name == "HyperBandit fixed mlp":
                    input_tensor = hypernet.hypernetwork_input_building(time,Timevectors,original_user_info,hypernet_setting).to(device)
                    input_tensor = input_tensor[:hypernet_setting.input_dim]

                    # 利用超网络生成Theta
                    with torch.no_grad():
                        output = mlptest(input_tensor)
                    output = output.detach().cpu().numpy()
                    Theta_Matrix = util.output_transform(output,hypernet_setting)
                    Theta = Theta_Matrix
                    
                else:
                    Theta = Theta_init

                if general_setting.iscluster and alg_name in ["ColinUCB", 'ColinUCB w/o W', "FactorUCB","FactorUCB w/o W"]: # "DLinUCB"
                    user_info = cluster_user_info
                else :
                    user_info = original_user_info
                random.shuffle(pool_item_info)
                
                # 算法选择出 item
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
                        label[position] = -1   
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

                rec_time += (pytime.time()-rec_tim_start)
                
                #统计各个算法 reward,regreat,itemid
                # if reward == -1: 
                #     reward = 0
                if i == warmcount and hypernet_setting.warm_start: #fileName == "./dataset/foursquare/NYC/Events_NYC_merged.dat"
                    # random_alg.accumulated_reward = 1
                    # algresult.AlgReward[alg_name] = [0 for i in range(len(algresult.AlgReward[alg_name]))]
                    # algresult.AlgPicked[alg_name] = [0 for i in range(len(algresult.AlgPicked[alg_name]))]
                    # algresult.AlgRegret[alg_name] = [0 for i in range(len(algresult.AlgRegret[alg_name]))]

                    random_alg.accumulated_reward = 1
                    algresult.AlgReward[alg_name] = []
                    algresult.AlgPicked[alg_name] = []
                    algresult.AlgRegret[alg_name] = []
                    algresult.AlgRewardRatio_vsRandom[alg_name] = []
                    algresult.BatchCumlateRegret[alg_name] = []
                 
                algresult.AlgReward[alg_name].append(reward)
                algresult.AlgRegret[alg_name].append(1 - reward)
                algresult.AlgPicked[alg_name].append(item_info_picked.id)
                count = i 
                # 如果上一条记录time是Friday，这条记录是Starday，就记录一下count
                
                if len(algresult.tim)>2 and hypernet_setting.warm_start and count> warmcount:#fileName == "./dataset/foursquare/NYC/Events_NYC_merged.dat":
                    time_list_last = algresult.tim[-2].strip('\n').split(" ")
                    time_week_last = np.array(time_list_last[0])
                    time_list = algresult.tim[-1].strip('\n').split(" ")
                    time_week = np.array(time_list[0])
                    
                    if count > warmcount and time_week_last not in ["Sat","Sun"] and time_week in ["Sat","Sun"]:
                        algresult.Fri2Sat.append(count-warmcount)

                elif  hypernet_setting.warm_start == False and len(algresult.tim)>2:#fileName != "./dataset/foursquare/NYC/Events_NYC_merged.dat"

                    time_list_last = algresult.tim[-2].strip('\n').split(" ")
                    time_week_last = np.array(time_list_last[0])
                    time_list = algresult.tim[-1].strip('\n').split(" ")
                    time_week = np.array(time_list[0])
                    if time_week_last not in ["Sat","Sun"] and time_week in ["Sat","Sun"]:
                        algresult.Fri2Sat.append(count)

                if count % config.test_setting.show_window == 0:

                    algresult.BatchCumlateReward[alg_name].append(sum(algresult.AlgReward[alg_name]))
                    #如果随机选择策略收益不为0,即分母不为0
                    if random_alg.accumulated_reward != 0:
                        relative_accumulated_reward = algresult.BatchCumlateReward[alg_name][-1] / (1.0 * random_alg.accumulated_reward)
                        # relative_accumulated_reward = (count - algresult.BatchCumlateRegret[alg_name][-1]) / (1.0 * random_alg.accumulated_reward)
                    else:
                        relative_accumulated_reward = 0

                    algresult.AlgRewardRatio_vsRandom[alg_name].append(relative_accumulated_reward)
                    # 每1000条event 输出一下累计收益
                    print("step:", count/config.test_setting.show_window, alg_name,  relative_accumulated_reward)

            if count % config.test_setting.show_window == 0:
                print("----------------------------")
            if count % hypernet_setting.train_window == 0  and hypernet_setting.is_train and "HyperBandit" in algresult.algorithms.keys():

                if config.test_setting.is_hypernet == True:
                    dataloader = hypernet.dataload(UserFeatureVectors, algresult, hypernet_setting, Timevectors)
                    traintime, epochcount = hypernet.trainnet(args, dataloader, criterion, optimizer, mlp, device, hypernet_setting)
                else:
                    print("no need to train hypernet")
                    pass
                dataloader = None
    

    save_path = "./model_log/mlp_{}_{}_rank_{}.pkl".format(hypernet_setting.time_embedding, dataset, hypernet_setting.rank )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(mlp, f)
    print("have saved mlp model")

    # 获取当前日期
    current_date = datetime.now()
    # 获取天部分，并确保它是两位数的格式
    day_formatted = current_date.strftime("%m-%d")
    
    os.makedirs(args.result_path, exist_ok=True)
    for alg_name, _ in algresult.AlgRewardRatio_vsRandom.items():
        with open(f"./{args.result_path}/{day_formatted}_{dataset}_{alg_name}_feature{hypernet_setting.feature}_warmstart{hypernet_setting.warm_start}.pkl","wb") as f: #util.get_time()
            pickle.dump(algresult.AlgRewardRatio_vsRandom[alg_name], f)

    with open(f"./{args.result_path}/{dataset}_F2S.pkl", "wb") as f:
        pickle.dump(algresult.Fri2Sat, f)
    return algresult.AlgRewardRatio_vsRandom, algresult.traintime, algresult.epochcount, rec_time, count



