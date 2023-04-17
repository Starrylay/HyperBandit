import numpy as np
import config
import json
import time
import torch
import pandas as pd
import datetime

def vectorize(Matrix):
    """
    将矩阵转置并向量化,按行打开
    """
    return np.reshape(Matrix.T, Matrix.shape[0] * Matrix.shape[1])
def matrixize(V, C_dimension):
    """
    (len(V) / C_dimension)行， C_dimension列
    """
    return np.transpose(np.reshape(V, (int(len(V) / C_dimension), C_dimension)))

def read_json(filename):
    """
    读取json文件
    Args:
        filename(str)
    return:
        dict
    """
    with open(filename, 'r') as f:
        data = json.load(f)
        if type(data) == dict:
            data = {int(key): value for key, value in data.items()}
        if type(data) == str:
            data = np.array(data)
        return data

def ReadFeatureFile(FeatureVectorsFileName):
    """
    读取lastfm的arm feature
    返回一个字典，key是 item的id，value是float型的feature
    """
    FeatureVectors = {}
    if FeatureVectorsFileName[-3:] =="dat":
        with open(FeatureVectorsFileName, "r") as f:
            for line in f:
                line = line.split("\t")
                vec = line[1].strip("[]").strip("\n").split(";") #[0] 是item id [1]是arm feature
                FeatureVectors[int(line[0])] = np.array(vec).astype(np.float)

    elif FeatureVectorsFileName[-3:] =="csv":
        # 读取没有列名的CSV文件
        data = pd.read_csv(FeatureVectorsFileName, header=0,sep=' ')
        for index, row in data.iterrows():
            key = int(row[0])
            value = row[1:].tolist()
            FeatureVectors[key] = np.array(value)

    return FeatureVectors

def ParseLine(line):
    """
    解析lastfm的交互历史的每一行
    Args:
        line(str)
    return:
        user_id(int)，
        time(str)，
        pool_items(array(int))
    """
    if config.general_setting.dataset == "lastfm":
        

        line_list = line.split("\t")
        user_id, pool_articles, dayofweek, blockofday = line_list[0], line_list[1], line_list[2], line_list[3].strip()
        user_id= int(user_id)
        pool_articles = np.array(pool_articles.strip("[").strip("\n").strip("]").split(",")).astype(int)
        # time = time.apply(datetime.utcfromtimestamp)
        time = dayofweek + ' ' +  blockofday
        # line_list = line.split("\t")
        # user_id, time, pool_articles = line_list[0], line_list[1],line_list[2]
        # user_id= int(user_id)
        # pool_articles = np.array(pool_articles.strip("[").strip("\n").strip("]").split(",")).astype(int)
        return user_id, time, pool_articles

    elif config.general_setting.dataset == "NYC":
        line_list = line.split('\t\t')
        user_id = int(line_list[0])
        pool_items = np.array(line_list[1].strip("\n").split(" ")).astype(int)
        time = line_list[2] + ' ' + line_list[3]
        return user_id, time, pool_items
    
    elif config.general_setting.dataset == "TKY":
        line_list = line.split("\t")
        user_id, pool_articles, dayofweek, blockofday = line_list[0], line_list[1], line_list[2], line_list[3].strip()
        user_id = int(user_id)
        pool_items = np.array(pool_articles.strip("[").strip("\n").strip("]").split(",")).astype(int)
        time = dayofweek + ' ' +  blockofday
        return user_id, time, pool_items

def get_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

#读取feature
def read_in_observed_feature(dataset):
    """
    read in observed user feature and observed item feature corresponding to the dataset.
    Args:
        dataset(str):dataset name
    Return:
        UserFeatureVectors(dict): {user id : observed user feature}
        ItemFeatureVectors(dict): {item id : observed item feature}
    """

    UserFeatureVectors = {}
    ItemFeatureVectors = {}
    ClusterFeatureVectors = {}
    if dataset == "NYC":
        # UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/User_FeatureVectors.dat")  # 读入userr feature
        UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        # ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare//NYC/POI_Cat_FeatureVectors.dat")  # 读入itemfeature
        ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/POI_Glove_FeatureVectors.csv")
        ClusterFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/40clusters_FeatureVectors.dat")  #cluste
    if dataset == "TKY":
        UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/TKY/User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        # UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        #ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/POI_Cat_FeatureVectors.dat")  # 读入itemfeature
        ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/TKY/POI_Glove_FeatureVectors.csv")
        ClusterFeatureVectors = ReadFeatureFile("./dataset/foursquare/TKY/40clusters_FeatureVectors.dat")  #cluste

    if dataset == "lastfm":
        UserFeatureVectors = ReadFeatureFile("./dataset/lastfm/lastfm_User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        ItemFeatureVectors = ReadFeatureFile("./dataset/lastfm/Arm_FeatureVectors_2.dat")  # 读入itemfeature
        ClusterFeatureVectors = ReadFeatureFile("./dataset/lastfm/lastfm_40clusters.dat")
    return UserFeatureVectors, ItemFeatureVectors, ClusterFeatureVectors

#读取label
def read_label(dataset):
    if dataset == "NYC":
        filename = "./dataset/foursquare/NYC/40clusters_labels.json"
     
        label = read_json(filename)#用户id对应的类别
    if dataset == "TKY":
        filename = "./dataset/foursquare/TKY/40clusters_labels.json"
        label = read_json(filename)#用户id对应的类别
    if dataset == "lastfm":
        filename = "./dataset/lastfm/lastfm_40clusters_labels.json"
        label = read_json(filename)
        
    return label

