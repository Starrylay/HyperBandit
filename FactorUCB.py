import numpy as np
from util import vectorize

class BaseItemStruct:
    def __init__(self, item_id, item_observed_dimension, latent_dimension, user_observed_dimension, lambda_, init="zero"):
        self.item_id = item_id
        self.observed_dimension = item_observed_dimension
        self.latent_dimension = latent_dimension
        self.context_dimension = item_observed_dimension + latent_dimension
        self.user_observed_dimension = user_observed_dimension
        if (init == "random"):
            self.latent_feature = np.random.rand(self.latent_dimension)
        else:
            self.latent_feature = np.zeros(self.latent_dimension)
        self.observed_feature = np.zeros(self.observed_dimension) #初始化（为了格式正确）在decide处，根据armpool赋值
        self.context_feature = np.concatenate((self.observed_feature, self.latent_feature), axis=0)
        self.lambda_ = lambda_
        self.Phi = np.zeros((self.latent_dimension,self.latent_dimension))
        self.Psi = self.lambda_ * np.identity(n = self.latent_dimension)
        self.b = np.zeros(self.latent_dimension)
        self.PhiT = self.Phi.T

    def updateParameters(self, users_info, rewards, Theta):#Users 元素是 User
        ThetaS = Theta[:self.observed_dimension, :]  # do * du(user dimension)
        ThetaX = Theta[self.observed_dimension:, :]  # dl * du(user dimension)
        Batch_user_context_feature = []
        Batch_reward = []
        for user, reward in zip(users_info, rewards):
            user_observed_feature = user.observed_feature[:self.user_observed_dimension]
            Batch_user_context_feature.append(user_observed_feature)
            Batch_reward.append(reward)
        Batch_user_array = np.array(Batch_user_context_feature).T # du(user dimension) *  n user
        Batch_reward_array = np.array(Batch_reward) # n user * 1
        P = np.dot(ThetaX, Batch_user_array).T # n user * dl
        Q = np.dot(ThetaS, Batch_user_array).T # n user * do
        self.Phi = 0.99 * self.Phi + np.dot(P.T, P) # dl * dl
        self.b = 0.99 * self.b + np.dot(P.T, Batch_reward_array - np.dot(Q, self.observed_feature)) # dl * 1
        self.Psi = self.Phi + self.lambda_ * np.identity(self.Phi.shape[0])
        self.latent_feature = np.dot(np.linalg.inv(self.Psi), self.b)
        #同步更新一下contex_feature
        self.update_contex_feature(self.observed_feature, self.latent_feature)
        # self.context_feature = np.concatenate((self.observed_feature, self.latent_feature), axis=0)

    def get_item_parameters(self):
        return self.latent_feature, self.Psi

    def get_context_feature(self):
        return self.context_feature
    
    def update_contex_feature(self, observed_feature=None, latent_feature=None):  
        if observed_feature is None:
            observed_feature = self.observed_feature
        if latent_feature is None:
            latent_feature = self.latent_feature
        self.context_feature = np.concatenate((observed_feature, latent_feature), axis=0)
        return 
    
    def get_Psi(self):
        return self.Psi


class FactorUCBAlgorithm:
    def __init__(self, setting):  # n is number of users
        self.alpha = setting.alpha
        self.item_num = setting.item_num
        self.lambda_ = setting.lambda_
        self.item_observed_dim = setting.item_observed_dimension
        self.latent_dim = setting.item_latent_dimension
        self.user_observed_dim = setting.user_observed_dimension
        self.ItemStruct_total = []
        for item_id in range(self.item_num):
            self.ItemStruct_total.append(
                BaseItemStruct(item_id, self.item_observed_dim, self.latent_dim, self.user_observed_dim, self.lambda_, "zero"))
            
    def decide(self, pool_items_info, user_info, Theta):
        maxPTA = float('-inf')
        item_picked_info = None
        #为每个item 赋值
        for item_info in pool_items_info:
            observed_feature = item_info.observed_feature[:self.item_observed_dim]
            self.ItemStruct_total[item_info.id].observed_feature = observed_feature
             #同步更新一下contex_feature
            self.ItemStruct_total[item_info.id].update_contex_feature(observed_feature = observed_feature)
        items_pta = self.getProb(self.alpha, pool_items_info, user_info, Theta)
        maxid = np.argmax(items_pta)
        item_picked_info = pool_items_info[maxid]       
        return item_picked_info

    def getProb(self, alpha, pool_items_info, user_info, Theta): 
        items_context = []
        Psis = []
        Psis_inv = []
        for i in range(len(pool_items_info)):
            ItemStruct_current = self.ItemStruct_total[pool_items_info[i].id]
            item_context_feature = ItemStruct_current.get_context_feature()
            items_context.append(item_context_feature)

            Psi = ItemStruct_current.get_Psi()
            Psi_inv = np.linalg.inv(Psi)
            Psis_inv.append(Psi_inv) # armpool size(25) dl dl

        user_feature = user_info.observed_feature[:self.user_observed_dim]# 截取合适长度

        # 计算整个armpool的mean
        temps = np.matmul(items_context, Theta)
        means = np.matmul(temps, user_feature)

        ThetaX = Theta[self.item_observed_dim:, :] # dl * du(user dimension)
        P_line_temp = np.dot(ThetaX, user_feature).T  # 1 * dl

        vars = alpha * np.sqrt(np.matmul(np.matmul(P_line_temp, Psis_inv), P_line_temp.T))
        return means + vars

    def updateParameters(self, item_info_picked, reward, user_info, Theta):
        ItemStruct_current = self.ItemStruct_total[item_info_picked.id]
        users_info = []
        rewards = []
        users_info.append(user_info)
        rewards.append(reward)
        ItemStruct_current.updateParameters(users_info, rewards, Theta)


