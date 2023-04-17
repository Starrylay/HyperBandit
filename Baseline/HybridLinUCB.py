
import numpy as np
import util
import config

#------------------------Hybrid_LinUCB_____________________________________
class LinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, init="zero"):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        if (init == "random"):
            self.UserTheta = np.random.rand(self.d)
        else:
            self.UserTheta = np.zeros(self.d)
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        change = np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector*click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1
    # def getTheta(self):
    # 	return self.UserTheta

    # def getA(self):
    # 	return self.A

    # def getProb(self, alpha, article_FeatureVector):
    # 	if alpha == -1:
    # 		alpha = alpha = 0.1*np.sqrt(np.log(self.time+1))
    # 	mean = np.dot(self.UserTheta,  article_FeatureVector)
    # 	var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
    # 	pta = mean + alpha * var
    # 	return pta
    # def getProb_plot(self, alpha, article_FeatureVector):
    # 	mean = np.dot(self.UserTheta,  article_FeatureVector)
    # 	var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
    # 	pta = mean + alpha * var
    # 	return pta, mean, alpha * var



class Hybrid_LinUCB_singleUserStruct(LinUCBUserStruct):
    def __init__(self, userFeature, lambda_, userID):
        LinUCBUserStruct.__init__(self, len(userFeature), lambda_)
        self.d = len(userFeature)

        self.B = np.zeros([self.d, self.d**2])
        self.userFeature = userFeature
    def updateParameters(self, articlePicked_FeatureVector, click):
        additionalFeatureVector = util.vectorize(np.outer(self.userFeature, articlePicked_FeatureVector))
        LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
        self.B +=np.outer(articlePicked_FeatureVector, additionalFeatureVector)
    def updateTheta(self, beta):
        self.UserTheta = np.dot(self.AInv, (self.b- np.dot(self.B, beta)))



class Hybrid_LinUCBUserStruct:
    def __init__(self, featureDimension,  lambda_, UserFeatureVectors):

        self.k = featureDimension**2
        self.A_z = lambda_*np.identity(n = self.k)
        self.b_z = np.zeros(self.k)
        self.A_zInv = np.linalg.inv(self.A_z)
        self.beta = np.dot(self.A_zInv, self.b_z)
        self.users = {}
        for uid in UserFeatureVectors:
            self.users[uid] = Hybrid_LinUCB_singleUserStruct(UserFeatureVectors[uid][:featureDimension], lambda_ , uid)

    def updateParameters(self, articlePicked_FeatureVector, click, userID):

        z = util.vectorize( np.outer(self.users[userID].userFeature, articlePicked_FeatureVector))

        temp = np.dot(np.transpose(self.users[userID].B), self.users[userID].AInv)

        self.A_z += np.dot(temp, self.users[userID].B)
        self.b_z +=np.dot(temp, self.users[userID].b)

        self.users[userID].updateParameters(articlePicked_FeatureVector, click)

        temp = np.dot(np.transpose(self.users[userID].B), self.users[userID].AInv)

        self.A_z = self.A_z + np.outer(z,z) - np.dot(temp, self.users[userID].B)
        self.b_z =self.b_z+ click*z - np.dot(temp, self.users[userID].b)
        self.A_zInv = np.linalg.inv(self.A_z)

        self.beta =np.dot(self.A_zInv, self.b_z)
        self.users[userID].updateTheta(self.beta)

    def getProb(self, alpha, article_FeatureVector,userID):
        x = article_FeatureVector
        z = util.vectorize(np.outer(self.users[userID].userFeature, article_FeatureVector))
        temp =np.dot(np.dot(np.dot( self.A_zInv , np.transpose( self.users[userID].B)) , self.users[userID].AInv), x )
        mean = np.dot(self.users[userID].UserTheta,  x)+ np.dot(self.beta, z)
        s_t = np.dot(np.dot(z, self.A_zInv),  z) + np.dot(np.dot(x, self.users[userID].AInv),  x)
        -2* np.dot(z, temp)+ np.dot(np.dot( np.dot(x, self.users[userID].AInv) ,  self.users[userID].B ) ,temp)

        var = np.sqrt(s_t)
        pta = mean + alpha * var
        return pta


class Hybrid_LinUCBAlgorithm:

    def __init__(self):
        self.dimension = config.HybridLinUCB_setting.user_dimension
        self.alpha = config.HybridLinUCB_setting.alpha
        self.lambda_ = config.HybridLinUCB_setting.lambda_
        UserFeatureVectors, _, _ = util.read_in_observed_feature(config.general_setting.dataset)
        self.USER = Hybrid_LinUCBUserStruct(self.dimension , self.lambda_, UserFeatureVectors)
    def decide(self, pool_items_info, user_info,Theta):
        userID = user_info.id
        maxPTA = float('-inf')
        articlePicked = None
        for x in pool_items_info:
            x_pta = self.USER.getProb(self.alpha, x.contextFeatureVector[:self.dimension], userID)
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta
        return articlePicked
    def updateParameters(self, item_info_picked, reward, user_info,Theta):
        userID = user_info.id
        self.USER.updateParameters(item_info_picked.contextFeatureVector[:self.dimension], reward, userID)

    def getCoTheta(self, userID):
        return self.USER.users[userID].UserTheta
