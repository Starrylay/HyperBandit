import numpy as np
import torch




class global_setting:
 

 
    # if torch.cuda.is_available():
    #     device  = torch.device("cuda")  # 将模型移动到GPU上
    #     # device = torch.device("cpu")
    #     print("GPU is available.")
    # else:
    #     device  = torch.device("cpu")  # 使用CPU
    #     print("GPU is not available.")

    # timeembeding = "glove" #"glove" "onehot" "polar"
    iscluster = True
    # train = True
    # user_input = False
    # sample = False

class test_setting():
    @classmethod
    def from_args(cls, args):
        cls.isupdate = args.isupdate
        cls.show_window = 200
        cls.is_hypernet = args.is_hypernet
        return cls
    

# version 1
#test


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

    def __init__(self,dataset,args): 
        # for key, value in vars(args).items():
        #     setattr(self, key, value)
          
        self.dataset = dataset
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1

        self.pool_size = args.pool_size
       


"""
NeuralLinear
"""
class NeuralLinear_setting():
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
    # lambda_ =
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = "128,256,256"
        self.a0 = 6
        self.b0 = 6

        self.item_dimension = 25
        self.user_dimension = self.item_dimension
        self.alpha = 0.1


"""
FactorUCB
"""
class FactorUCB_setting():
    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        # self.dataset = dataset
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25

        if test_setting.isupdate == False:
            self.item_latent_dimension = 0 #dl  待学习
        else:
            self.item_latent_dimension = 10 #dl 待学习

        self.item_dimension = 25
        self.item_observed_dimension = self.item_dimension - self.item_latent_dimension #do <= 10 5
        # user
        self.user_observed_dimension = 25 # du给定
        self.alpha2 = 0.1



"""
hypernet_setting
"""
class hypernet_setting_SVE():
    def __init__(self,rank=-1):
        # super().__init__()
        self.rank = rank
        # self.is_train = args.train #global_setting.train
        # self.sample_rate = args.sample_rate #global_setting.sample
        # self.user_input = args.user_input#global_setting.user_input
        # self.time_embedding = args.time_embedding
        # self.feature = args.feature
        # self.dataset = args.dataset
        # self.warm_start = args.warm_start

        self.vocab_size = 35 # 时间段
        self.embedding_dim = 30 # time_embedding dimension
        
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25 # 没用，直接用的general setting的pool size

        # self.item_latent_dimension = 0 #dl 5 待学习
        self.item_dimension = 25
        # self.item_observed_dimension = self.item_dimension - self.item_latent_dimension #do <= 10 5
        # user
        self.user_observed_dimension = 25 # du给定
        self.alpha2 = 0.1
        
        if self.time_embedding == "glove":
            self.input_dim = 30 # 7 + 5 # + FactorUCB_setting.user_observed_dimension  # 输入层维度
        elif self.time_embedding == "onehot":
            self.input_dim = 7+5
        elif self.time_embedding == "polar":
            self.input_dim = 4
        elif self.time_embedding == "learn":
            self.input_dim = 1
        if self.user_input == True:
            self.input_dim += self.user_observed_dimension


        self.hidden_dim = 256  # 隐藏层维度

        if self.rank == -1:
            self.output_dim = self.user_observed_dimension * self.item_dimension
        else:
            self.output_dim = self.rank*(self.user_observed_dimension + self.item_dimension)  # 输出层 theta的向量展开

        self.learning_rate = 0.001  # 学习率       
        self.batch_size = 100
        if self.dataset == "lastfm":
            self.train_window = 1000
            self.num_epochs = 1 # epoch
        if self.dataset == "NYC":
            self.train_window = 2000 
            self.num_epochs = 30 # epoch15
        if self.dataset == "TKY":
            self.train_window = 5000
            self.num_epochs = 20 # epoch10
        if self.dataset == "kuai":
            self.train_window = 2000
            self.num_epochs = 15 # epoch10


class hypernet_setting():
    def __init__(self,args):
        # super().__init__()
        self.rank = args.rank
        self.is_train = args.train #global_setting.train
        self.sample_rate = args.sample_rate #global_setting.sample
        self.user_input = args.user_input#global_setting.user_input
        self.time_embedding = args.time_embedding
        self.feature = args.feature
        self.dataset = args.dataset
        self.warm_start = args.warm_start

        self.vocab_size = 35 # 时间段
        self.embedding_dim = 30 # time_embedding dimension
        
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25

        # self.item_latent_dimension = 0 #dl 5 待学习
        self.item_dimension = 25
        # self.item_observed_dimension = self.item_dimension - self.item_latent_dimension #do <= 10 5
        # user
        self.user_observed_dimension = 25 # du给定
        self.alpha2 = 0.1
        
        if self.time_embedding == "glove":
            self.input_dim = 30 # 7 + 5 # + FactorUCB_setting.user_observed_dimension  # 输入层维度
        elif self.time_embedding == "onehot":
            self.input_dim = 7+5
        elif "polar" in  self.time_embedding:
            self.input_dim = 4
        elif self.time_embedding == "learn":
            self.input_dim = 1
        if self.user_input == True:
            self.input_dim += self.user_observed_dimension


        self.hidden_dim = 256  # 隐藏层维度

        if self.rank == -1:
            self.output_dim = self.user_observed_dimension * self.item_dimension
        else:
            self.output_dim = self.rank*(self.user_observed_dimension + self.item_dimension)  # 输出层 theta的向量展开

        self.learning_rate = 0.001  # 学习率       
        self.batch_size = 100
        if self.dataset == "lastfm":
            self.train_window = 1000
            self.num_epochs = 1 # epoch
        if self.dataset == "NYC":
            self.train_window = 2000 
            self.num_epochs = 30 # epoch15
        if self.dataset == "TKY":
            self.train_window = 5000
            self.num_epochs = 20 # epoch10
        if self.dataset == "kuai":
            self.train_window = 2000
            self.num_epochs = 15 # epoch10

"""
LinUCB
"""
class LinUCB_setting():
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
    # lambda_ =
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25



        self.item_dimension = 25
        self.user_dimension = self.item_dimension
        self.alpha = 0.1

"""
HybridLinUCB
"""
class HybridLinUCB_setting():
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25
    
        self.item_dimension = 25
        self.user_dimension = self.item_dimension
        self.alpha = 0.1

"""
dLinUCB
"""
class dLinUCB_setting():
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25    


        self.item_dimension = 25 # 10 if foursquare 25 if lastfm
        self.user_dimension = self.item_dimension
        self.tau = 200 #200lastfm
        #defult
        self.tilde_delta_1 = 0.01
        self.delta_1 = 0.01
        if self.dataset == "lastfm":
            self.eta = 0.9  # 0.9 lastfm
        if self.dataset == "NYC":
            self.eta = 0.8 
        if self.dataset == "TKY":
            self.eta = 0.8  
        if self.dataset == "kuai":
            self.eta = 0.8  
         
        #模拟参数设置，不用
        self.alpha = 0.01 # 使用alpha_t
        self.lambda_ = 0.01
        self.delta_2 = 0.1
        self.NoiseScale = 0.1

"""
ColinUCB
"""
class ColinUCB_setting():
    def __init__(self,dataset):
        self.dataset = dataset
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25




    #参数设置
        # item
        self.item_dimension = 25
        self.user_dimension = self.item_dimension
"""
CLUB
"""
class CLUB_setting():
    def __init__(self,dataset):
        self.dataset = dataset
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25
    


    #参数设置
        # item
        self.item_dimension = 25
        self.user_dimension = self.item_dimension  

"""
ADTS
"""
class ADTS_setting():
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        if self.dataset == "NYC":
            self.user_num = 1500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "TKY":
            self.user_num = 2500
            self.cluster_num = 40
            self.item_num = 400
        if self.dataset == "lastfm":
            self.user_num = 2100
            self.cluster_num = 40
            self.item_num = 20000
        if self.dataset == "kuai":
            self.user_num = 7500
            self.cluster_num = 40
            self.item_num = 2000
        self.item_dimension = 25
        self.iscluster = global_setting.iscluster
        self.alpha = 0.1
        self.lambda_ = 0.1
        self.pool_size = 25

    #参数设置
    # item
        self.item_observed_dimension = 15 #do <= 10 5
        self.item_latent_dimension = 10 #dl 5
        self.item_dimension = self.item_observed_dimension + self.item_latent_dimension
        # user
        self.user_observed_dimension = 25 # du(待学习)
        self.alpha2 = 0.1

