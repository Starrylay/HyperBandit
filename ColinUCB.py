import numpy as np
# from scipy.linalg import sqrtm
from util import vectorize, matrixize
import util
import warnings
import config
import json

class CoLinUCBUserSharedStruct(object):
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.currentW = np.identity(n = userNum)
		self.featureDimension = featureDimension
		self.W = W
		# print ("W: ", self.W)
		self.userNum = userNum
		self.A = np.identity(n = featureDimension*userNum)#lambda_*
		self.b = np.zeros(featureDimension*userNum)
		self.AInv =  np.linalg.inv(self.A)  # 1/lambda_ * np.identity(n = featureDimension*userNum)#la

		
		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.CoTheta = np.zeros(shape = (featureDimension, userNum))

		self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))
		# print ("Big W: ", self.BigW)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))
		self.alpha_t = 0.0
		self.sigma = 1.e-200   #Used in the high probability bound, i.e, with probability at least (1 - sigma) the confidence bound. So sigma should be very small
		self.lambda_ = lambda_
	def updateParameters(self, articlePicked, click,  user_info, update ='Inv'):
		X = vectorize(np.outer(articlePicked.observed_feature[:self.featureDimension], self.W.T[user_info.id]))
		#print "X: " + str(X)
		change = np.outer(X, X)
		self.A += change
		self.b += click*X
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked.observed_feature[:self.featureDimension]))
		self.CoTheta = np.dot(self.UserTheta, self.W) 
		self.CCA = np.dot(np.dot(self.BigW, self.AInv), np.transpose(self.BigW))
	
	# #计算单个item
	def getProb(self, alpha, article, userID):
		warnings.filterwarnings('error')# 将任何警告转化为error终止程序
		TempFeatureM = np.zeros(shape =(len(article.observed_feature[:self.featureDimension]), self.userNum))
		TempFeatureM.T[userID] = article.observed_feature[:self.featureDimension]
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.observed_feature[:self.featureDimension])
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))

		#self.alpha_t = 0.01*np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		# try:
		# 	self.alpha_t = 0.01*np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		# except:
		# 	self.alpha_t = 0.0
		#pta = mean + alpha * var    # use emprically tuned alpha
		pta = mean + alpha *var   # use the theoretically computed alpha_t
		return pta

	def getUserCoTheta(self, user_id):
		return self.CoTheta.T[user_id]

	def getCCA(self):
		return self.CCA	
	def calculateAlphaT(self):
		warnings.filterwarnings('error')
		try:
			self.alpha_t = 0.01*np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		except:
			self.alpha_t = 0.0
		return self.alpha_t

#---------------CoLinUCB(fixed user order) algorithms: Asynisized version and Synchorized version		
class CoLinUCBAlgorithm:
	def __init__(self, W_type='None'):
		
		self.alpha = config.ColinUCB_setting.alpha
		self.lambda_ = config.ColinUCB_setting.lambda_
		self.update = 'inv' #default is inverse. Could be 'rankone' or 'inv'

		self.cluster_num = config.ColinUCB_setting.cluster_num
		self.user_dimension = config.ColinUCB_setting.user_dimension
		if W_type == "Have":
			if config.general_setting.dataset == "NYC":
				W = util.read_json('./dataset/foursquare/NYC/SparseW.json')
			if config.general_setting.dataset == "TKY":
				W = util.read_json('./dataset/foursquare/TKY/SparseW.json')
			if config.general_setting.dataset == "lastfm":
				W = util.read_json('./dataset/lastfm/lastfm_SparseW.json')
		elif W_type == "None":
			W = np.identity(self.cluster_num)
		# print("W: ", W)
		self.W = np.array(W) #用户类别关系矩阵
		self.USERS = CoLinUCBUserSharedStruct(self.user_dimension, self.lambda_, self.cluster_num, self.W)

	# def decide_old(self, pool_articles, userID):
	# 	maxPTA = float('-inf')
	# 	articlePicked = None

	# 	for x in pool_articles:
	# 		x_pta = self.USERS.getProb(self.alpha, x, userID)
	# 		# pick article with highest Prob
	# 		if maxPTA < x_pta:
	# 			articlePicked = x
	# 			maxPTA = x_pta

	# 	return [articlePicked]

	

	def decide(self, pool_articles, user_info, Theta, k = 1):
		# MEAN
		art_features = np.empty([len(pool_articles), len(pool_articles[0].observed_feature[:self.user_dimension])]) #pool  d
		for i in range(len(pool_articles)):
			art_features[i, :] = pool_articles[i].observed_feature[:self.user_dimension]
		user_features = self.USERS.CoTheta.T[user_info.id]  # d 1
		mean_matrix = np.matmul(art_features, user_features) # pool 1

		# VARIANCE
		art_temp_features = np.empty([len(pool_articles), len(pool_articles[0].observed_feature[:self.user_dimension])*self.cluster_num])
		for i in range(len(pool_articles)):
			TempFeatureM = np.zeros(shape =(len(pool_articles[0].observed_feature[:self.user_dimension]), self.cluster_num))
			TempFeatureM.T[user_info.id] = pool_articles[i].observed_feature[:self.user_dimension]
			art_temp_features[i, :] = vectorize(TempFeatureM)  # pool  d*N

		
		var_matrix = self.alpha * np.sqrt(np.diag(np.matmul(np.matmul(art_temp_features, self.USERS.CCA), art_temp_features.T)))
		# maxPTA = float('-inf')
		# articlePicked = None

		# for x in pool_articles:
		# 	x_pta = self.USERS.getProb(self.alpha, x, user_info.id)
		# 	# pick article with highest Prob
		# 	if maxPTA < x_pta:
		# 		articlePicked = x
		# 		maxPTA = x_pta

		# return articlePicked







		#self.USERS.calculateAlphaT()
		# if self.use_alpha_t:
		# 	self.USERS.calculateAlphaT()
		# 	pta_matrix = mean_matrix + self.USERS.alpha_t*np.diag(var_matrix)
		# else:
		# a = self.alpha*np.diag(var_matrix)
		pta_matrix = mean_matrix + var_matrix
		# print(pta_matrix)
		pool_positions = np.argsort(pta_matrix)[-k:]

		articles = []
		for i in range(k):
			articles.append(pool_articles[pool_positions[i]])

		return articles[0]

		#return pool_articles[pool_position]

	def updateParameters(self, articlePicked, click, user_info, Theta):
		self.USERS.updateParameters(articlePicked, click, user_info, update = self.update)
		
	def getLearntParameters(self, user_info):
		return self.USERS.UserTheta.T[user_info.id]

	def getTheta(self, user_info):
		return self.USERS.UserTheta.T[user_info.id]

	def getCoTheta(self, user_info):
		return self.USERS.CoTheta.T[user_info.id]

	def getA(self):
		return self.USERS.A

	def getW(self, user_info):
		
		return self.USERS.W



	
