
import numpy as np
import config
import util


class BaseUserStruct:
    def __init__(self, user_id, usertheta_dimension, lambda_, init="random"):
        self.user_id = user_id
        self.d = usertheta_dimension
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        if (init == "random"):
            self.UserTheta = np.random.rand(self.d)
        else:
            self.UserTheta = np.zeros(self.d)
        self.time = 0
    def updateParameters(self, item_info_picked, reward):
        articlePicked_FeatureVector = item_info_picked.observed_feature[:self.d]
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * reward
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1
    def get_user_parameters(self):
        return self.UserTheta, self.AInv, self.time


#---------------LinUCB UserBased  更新用户theta ---------------
class LinUCBAlgorithm_UserBased:
    def __init__(self):
        self.user_num = config.general_setting.user_num
        self.user_dim = config.LinUCB_setting.user_dimension
        self.item_dim = config.LinUCB_setting.item_dimension
        self.alpha = config.LinUCB_setting.alpha
        self.lambda_ = config.LinUCB_setting.lambda_
        self.UserStruct_total = []
        self.count = 0
        self.youxiaocount = 0
        for user_id in range(self.user_num):
            self.UserStruct_total.append(
                BaseUserStruct(user_id, self.user_dim, self.lambda_, "random"))

    def decide(self, pool_items_info, user_info, Theta):# Theta没用
        maxPTA = float('-inf')
        item_picked_info = None
        for item_info in pool_items_info:
            x_pta = self.getProb(self.alpha, item_info, user_info)
            if maxPTA < x_pta:
                item_picked_info = item_info
                maxPTA = x_pta
        return item_picked_info

    def getProb(self, alpha, item_info, user_info):
        UserStruct_current = self.UserStruct_total[user_info.id]
        UserTheta, AInv, time = UserStruct_current.get_user_parameters()
        if alpha == -1:
            alpha = 0.1 * np.sqrt(np.log(time + 1))
        item_observed_feature = item_info.observed_feature[:self.item_dim]
        mean = np.dot(UserTheta, item_observed_feature)
        var = np.sqrt(np.dot(np.dot(item_observed_feature, AInv), item_observed_feature))
        pta = mean + alpha * var
        return pta

    def updateParameters(self, item_info_picked, reward, user_info, Theta):
        UserStruct_current = self.UserStruct_total[user_info.id]
        UserStruct_current.updateParameters(item_info_picked, reward)
        self.count += 1
        # if reward == 1:
        #     self.youxiaocount +=1
        #     print("usertheta有效更新次数", self.youxiaocount)
        #     usertheta, _, _ = UserStruct_current.get_user_parameters()
        #     print("事件数目", self.count)
        #     print("usertheta", usertheta)


class BaseItemStruct:
    def __init__(self, item_id, itemtheta_dimension, lambda_, init="random"):
        self.item_id = item_id
        self.d = itemtheta_dimension
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        if (init == "random"):
            self.ItemTheta = np.random.rand(self.d)
        else:
            self.ItemTheta = np.zeros(self.d)
        self.time = 0
    # def updateParameters(self, item_info_picked, reward):
    #     articlePicked_FeatureVector = item_info_picked.observed_feature[:config.LinUCB_setting.item_dimension]
    #     self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
    #     self.b += articlePicked_FeatureVector * reward
    #     self.AInv = np.linalg.inv(self.A)
    #     self.UserTheta = np.dot(self.AInv, self.b)
    #     self.time += 1

    def updateParameters(self, user_info, click):
        self.time += 1
        # if userID in self.count:
        #     self.count[userID] += 1
        # else:
        #     self.count[userID] = 1
        self.A += np.outer(user_info.observed_feature[:self.d],
                            user_info.observed_feature[:self.d])
        self.b += user_info.observed_feature[:self.d] * click
        self.AInv = np.linalg.inv(self.A)
        self.ItemTheta = np.dot(self.AInv, self.b)

    def get_item_parameters(self):
        return self.ItemTheta, self.AInv, self.time




#---------------LinUCB ItemBased 更新item algorithm---------------
class LinUCBAlgorithm_ItemBased:
    def __init__(self):
        self.item_num = config.general_setting.item_num
        self.item_dim = config.LinUCB_setting.item_dimension

        self.alpha = config.LinUCB_setting.alpha
        self.lambda_ = config.LinUCB_setting.lambda_
        self.ItemStruct_total = []
        self.count = 0
        self.youxiaocount=0
        for item_id in range(self.item_num):
            self.ItemStruct_total.append(
                BaseItemStruct(item_id, self.item_dim, self.lambda_, "random"))

    def decide(self, pool_items_info, user_info, Theta):# Theta没用
        maxPTA = float('-inf')
        item_picked_info = None
        for item_info in pool_items_info:
            x_pta = self.getProb(self.alpha, item_info, user_info)
            if maxPTA < x_pta:
                item_picked_info = item_info
                maxPTA = x_pta
        return item_picked_info

    def getProb(self, alpha, item_info, user_info):
        ItemStruct_current = self.ItemStruct_total[item_info.id]
        ItemTheta, AInv, time = ItemStruct_current.get_item_parameters()
        if alpha == -1:
            alpha = 0.1 * np.sqrt(np.log(time + 1))
        user_observed_feature = user_info.observed_feature[:self.item_dim]
        mean = np.dot(user_observed_feature, ItemTheta)
        var = np.sqrt(np.dot(np.dot(user_observed_feature, AInv), user_observed_feature))
        pta = mean + alpha * var
        return pta

    def updateParameters(self, item_info_picked, reward, user_info, Theta):
        ItemStruct_current = self.ItemStruct_total[item_info_picked.id]
        ItemStruct_current.updateParameters(user_info, reward)
        # self.count += 1
        # if reward == 1:
        #     self.youxiaocount += 1
        #     print("itemtheta有效更新次数", self.youxiaocount)
        #     itemtheta, _, _ = ItemStruct_current.get_item_parameters()
        #     print("事件数目", self.count)
        #     print("itemtheta", itemtheta)










# import torch
# import numpy as np
# import config
# import util
# class BaseUserStruct:
#     def __init__(self, user_id, usertheta_dimension, lambda_, init="zero"):
#         self.user_id = user_id
#         self.d = usertheta_dimension
#         self.lambda_ = lambda_
#         self.A = lambda_ * torch.eye(self.d).cuda()
#         self.b = torch.zeros(self.d).cuda()
#         self.AInv = torch.inverse(self.A).cuda()
#         if (init == "random"):
#             self.UserTheta = torch.rand(self.d).cuda()
#         else:
#             self.UserTheta = torch.zeros(self.d).cuda()
#         self.time = 0
#     def updateParameters(self, item_info_picked, reward):
#         articlePicked_FeatureVector = item_info_picked.observed_feature[:config.LinUCB_setting.item_dimension]
#         articlePicked_FeatureVector_t = torch.from_numpy(articlePicked_FeatureVector).float().cuda()
#         reward_t = torch.tensor(reward, dtype=torch.float32).cuda()
#         self.A += torch.ger(articlePicked_FeatureVector_t, articlePicked_FeatureVector_t).cuda()
#         self.b += articlePicked_FeatureVector_t * reward_t
#         self.AInv = torch.inverse(self.A).cuda()
#         self.UserTheta = torch.matmul(self.AInv, self.b).cuda()
#         self.time += 1
#
#     def get_user_parameters(self):
#         return self.UserTheta, self.AInv, self.time
#     # 给 UserTheta 赋初值
#     def set_user_parameters(self, UserTheta):
#         self.b = self.lambda_ * UserTheta[:self.d]
#
# #---------------LinUCB(fixed user order) algorithm---------------
# class LinUCBAlgorithm:
#     def __init__(self):
#         self.user_num = config.general_setting.user_num
#         self.user_dim = config.LinUCB_setting.user_dimension
#         self.item_dim = config.LinUCB_setting.item_dimension
#         self.alpha = config.LinUCB_setting.alpha
#         self.lambda_ = config.LinUCB_setting.lambda_
#         self.UserStruct_total = []
#         UserFeatureVectors, _, _ = util.read_in_observed_feature(config.general_setting.dataset)
#         for user_id in range(self.user_num):
#             CurrentUserStruct = BaseUserStruct(user_id, self.user_dim, self.lambda_, "zero")
#             User_init_feature = torch.from_numpy(UserFeatureVectors[user_id]).float().cuda()
#             CurrentUserStruct.set_user_parameters(User_init_feature)
#             self.UserStruct_total.append(CurrentUserStruct)
#
#     def decide(self, pool_items_info, user_info, Theta):# Theta没用
#         maxPTA = float('-inf')
#         item_picked_info = None
#         for item_info in pool_items_info:
#             x_pta = self.getProb(self.alpha, item_info, user_info)
#             if maxPTA < x_pta:
#                 item_picked_info = item_info
#                 maxPTA = x_pta
#         return item_picked_info
#
#     def getProb(self, alpha, item_info, user_info):
#         UserStruct_current = self.UserStruct_total[user_info.id]
#         UserTheta, AInv, time = UserStruct_current.get_user_parameters()
#         if alpha == -1:
#             alpha = 0.1 * np.sqrt(np.log(time + 1))
#         item_observed_feature = item_info.observed_feature[:self.item_dim]
#         item_observed_feature_t = torch.from_numpy(item_observed_feature).float().cuda()
#         mean = torch.dot(UserTheta, item_observed_feature_t)
#         var = torch.sqrt(torch.dot(torch.matmul(item_observed_feature_t, AInv), item_observed_feature_t))
#         pta = mean + alpha * var
#         return pta.cpu().numpy()
#
#     def updateParameters(self, item_info_picked, reward, user_info, Theta):
#         UserStruct_current = self.UserStruct_total[user_info.id]
#         UserStruct_current.updateParameters(item_info_picked, reward)
#


