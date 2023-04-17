import numpy as np
import config


class BaseAlg:
    def __init__(self):
        pass

    def decide(self, pool_items_info, user_info, k, Theta):
        pass


    def update(self, articlePicked, click, userID):
        pass

    # def getV(self, articleID):
    #     if self.dimension == 0:
    #         return np.zeros(self.context_dimension + self.hidden_dimension)
    #     else:
    #         return np.zeros(self.dimension)
    #
    # def getW(self, userID):
    #     return np.identity(n=self.n_users)


class randomStruct:
    def __init__(self):
        self.accumulated_reward = 0
        self.accumulated_regret = 0

class Item:
    def __init__(self, id: int, feature: np.ndarray):
        self.id = id
        self.observed_feature = feature
        self.contextFeatureVector = feature
class User:
    def __init__(self, id: int, feature: np.ndarray):
        self.id = id
        self.observed_feature = feature