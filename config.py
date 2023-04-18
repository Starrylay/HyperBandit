import numpy as np
import torch




"""
general setting
"""
class general_setting: 
    """
    clusters:
        yahool 40 80 160  
        lastfm delicious 50 100 200 
    poolsize:
        yahool 10,
        lastfm delicious  25,
    feature dimension:
        yahool 6
        lastfm delicious 25
    """
    dataset = "NYC"  
    if dataset == "NYC":
        user_num = 1500
        cluster_num = 40
        item_num = 400
    if dataset == "TKY":
        user_num = 2500
        cluster_num = 40
        item_num = 400
    if dataset == "lastfm":
        user_num = 2100
        cluster_num = 40
        item_num = 20000

    iscluster = False
    alpha = 0.1
    lambda_ = 0.1

    pool_size = 25
    device = torch.device("cuda")



"""
LinUCB
"""
class LinUCB_setting(general_setting):
    # lambda_ =
    item_dimension = 25
    user_dimension = item_dimension
    alpha = 0.1

"""
HybridLinUCB
"""
class HybridLinUCB_setting(general_setting):

    item_dimension = 25
    user_dimension = item_dimension
    alpha = 0.1

"""
dLinUCB
"""
class dLinUCB_setting(general_setting):

    item_dimension = 25 # 10 if foursquare 25 if lastfm
    user_dimension = item_dimension
    tau = 200 #200lastfm
    #defult
    tilde_delta_1 = 0.01
    delta_1 = 0.01
    if general_setting.dataset == "lastfm":
        eta = 0.9  # 0.9 lastfm
    if general_setting.dataset == "NYC":
        eta = 0.8 
    if general_setting.dataset == "TKY":
        eta = 0.8  
    #模拟参数设置，不用
    alpha = 0.01 # 使用alpha_t
    lambda_ = 0.01
    delta_2 = 0.1
    NoiseScale = 0.1

"""
ColinUCB
"""
class ColinUCB_setting(general_setting):
    #参数设置
        # item
    item_dimension = 25
    user_dimension = item_dimension
"""
CLUB
"""
class CLUB_setting(general_setting):
    #参数设置
        # item
    item_dimension = 25
    user_dimension = item_dimension  

"""
FactorUCB
"""
class FactorUCB_setting(general_setting):
    #参数设置
    # item
    item_observed_dimension = 1 #do <= 10 5
    user_observed_dimension = 25 # du(待学习)

    item_latent_dimension = user_observed_dimension - item_observed_dimension  #dl 5 待学习
    
    item_dimension = user_observed_dimension
    # user
   
    alpha2 = 0.1






"""

ADTS
"""
class ADTS_setting(general_setting):
    #参数设置
    # item
    item_observed_dimension = 15 #do <= 10 5
    item_latent_dimension = 10 #dl 5
    item_dimension = item_observed_dimension + item_latent_dimension
    # user
    user_observed_dimension = 25 # du(待学习)
    alpha2 = 0.1


"""
hypernet_setting
"""
class hypernet_setting():
    is_train = False

    timedim = 7+5
    input_dim = 7 + 5 + FactorUCB_setting.user_observed_dimension  # 输入层维度
    hidden_dim = 256  # 隐藏层维度
    output_dim = FactorUCB_setting.user_observed_dimension * FactorUCB_setting.item_dimension  # 输出层 theta的向量展开
    learning_rate = 0.1  # 学习率
    
    batch_size = 20
    if general_setting.dataset == "lastfm":
        train_window = 2000
        num_epochs = 100  # epoch
    if general_setting.dataset == "NYC":
        train_window = 100#00
        num_epochs = 50  # epoch
    if general_setting.dataset == "TKY":
        train_window = 200
        num_epochs = 20  # epoch

class test_setting():
    isupdate = True
    show_window = 100
# version 1
