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
        data = pd.read_csv(FeatureVectorsFileName, header=0,sep=' ')
        for index, row in data.iterrows():
            try:
                key = int(row[0])
            except:
                key = row[0]
            value = row[1:].tolist()
            FeatureVectors[key] = np.array(value)

    return FeatureVectors

def ParseLine(line,dataset):
    """
    解析lastfm的交互历史的每一行
    Args:
        line(str)
    return:
        user_id(int)，
        time(str)，
        pool_items(array(int))
    """
    if dataset == "lastfm":
        

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

    elif dataset == "NYC":
        line_list = line.split('\t\t')
        user_id = int(line_list[0])
        pool_items = np.array(line_list[1].strip("\n").split(" ")).astype(int)
        time = line_list[2] + ' ' + line_list[3]
        return user_id, time, pool_items
    
    elif dataset == "TKY":
        line_list = line.split('\t\t')
        user_id = int(line_list[0])
        pool_items = np.array(line_list[1].strip("\n").split(" ")).astype(int)
        time = line_list[2] + ' ' + line_list[3]
        return user_id, time, pool_items
    
        # line_list = line.split("\t")
        # user_id, pool_articles, dayofweek, blockofday = line_list[0], line_list[1], line_list[2], line_list[3].strip()
        # user_id = int(user_id)
        # pool_items = np.array(pool_articles.strip("[").strip("\n").strip("]").split(",")).astype(int)
        # time = dayofweek + ' ' +  blockofday
        # return user_id, time, pool_items
    
    elif dataset == "kuai":

        line_list = line.split('\t\t')
        user_id = int(line_list[0])
        pool_items = np.array(line_list[1].strip("\n").split(" ")).astype(int)
        time = line_list[2] + ' ' + line_list[3]
        return user_id, time, pool_items

        # line_list = line.split("\t")
        # user_id, pool_articles, dayofweek, blockofday = line_list[0], line_list[1], line_list[2], line_list[3].strip()
        # user_id = int(user_id)
        # pool_items = np.array(pool_articles.strip("[").strip("\n").strip("]").split(",")).astype(int)
        # time = dayofweek + ' ' +  blockofday
        # return user_id, time, pool_items


def get_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def item_read(path):
    new_data = {}
    with open(path, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            new_key = int(key) 
            new_data[new_key] = np.array(value)
          
    return new_data

def user_read(path):
    new_data = {}
    with open(path, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            new_key = int(key) 
            new_data[new_key] = np.array(value)
    return new_data


#读取feature
def read_in_observed_feature(dataset, feature_type):
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
    Timevectors = ReadFeatureFile("./dataset/foursquare/NYC/timeVectors_30.csv") 
    if feature_type == "LLM_wo_attribute_pca":
        if dataset == "NYC":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_NYC_LLM_wo_attribute_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_NYC_LLM_wo_attribute_pca.json")
            # UserFeatureVectors = user_read("user_embedding_mean_item_embedding_NYC_LLM_wo_attribute_pca.json")
            # ItemFeatureVectors = item_read("item_embedding_NYC_LLM_wo_attribute_pca.json")
        if dataset == "TKY":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_TKY_LLM_wo_attribute_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_TKY_LLM_wo_attribute_pca.json")
        if dataset == "kuai":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_kuai_LLM_wo_attribute_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_kuai_LLM_wo_attribute_pca.json")
            
    if feature_type == "LLM_with_attribute_pca":
        if dataset == "NYC":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_NYC_LLM_with_attribute_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_NYC_LLM_with_attribute_pca.json")
            # UserFeatureVectors = user_read("user_embedding_mean_item_embedding_NYC_LLM_wo_attribute_pca.json")
            # ItemFeatureVectors = item_read("item_embedding_NYC_LLM_wo_attribute_pca.json")
        if dataset == "TKY":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_TKY_LLM_with_attribute_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_TKY_LLM_with_attribute_pca.json")
        if dataset == "kuai":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_kuai_LLM_with_attribute_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_kuai_LLM_with_attribute_pca.json")

    if feature_type == "glove_pca":
        if dataset == "NYC":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_NYC_glove_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_NYC_glove_pca.json")
        if dataset == "TKY":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_TKY_glove_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_TKY_glove_pca.json")
        if dataset == "kuai":
            UserFeatureVectors = user_read("embedding/user_embedding_mean_item_embedding_kuai_bert_pca.json")
            ItemFeatureVectors = item_read("embedding/item_embedding_kuai_bert_pca.json")


    if dataset == "NYC":
        # UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/User_FeatureVectors.dat")  # 读入userr feature
        # UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        # # ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare//NYC/POI_Cat_FeatureVectors.dat")  # 读入itemfeature
        # ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/POI_Glove_FeatureVectors.csv")
        ClusterFeatureVectors = ReadFeatureFile("./dataset/foursquare/NYC/40clusters_FeatureVectors.dat")  #cluste
    if dataset == "TKY":
        # UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/TKY/User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        # # UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        # #ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/POI_Cat_FeatureVectors.dat")  # 读入itemfeature
        # ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/TKY/POI_Glove_FeatureVectors.csv")
        ClusterFeatureVectors = ReadFeatureFile("./dataset/foursquare/TKY/40clusters_FeatureVectors.dat")  #cluste
    if dataset == "kuai":
        # UserFeatureVectors = ReadFeatureFile("./dataset/kuairec/user_feature_pca.csv")  # 读入userr feature
        # # UserFeatureVectors = ReadFeatureFile("./dataset/foursquare/User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        # # ItemFeatureVectors = ReadFeatureFile("./dataset/foursquare/POI_Cat_FeatureVectors.dat")  # 读入itemfeature
        # # ItemFeatureVectors = ReadFeatureFile("./dataset/kuairec/item_feature_pca_meanday.csv")
        # ItemFeatureVectors = ReadFeatureFile("./dataset/kuairec/cat_feature_pca.csv")
        ClusterFeatureVectors = ReadFeatureFile("./dataset/kuairec/40clusters_FeatureVectors.dat")  #cluste

    if dataset == "lastfm":
        # UserFeatureVectors = ReadFeatureFile("./dataset/lastfm/lastfm_User_Glovemean_FeatureVectors.csv")  # 读入userr feature
        # ItemFeatureVectors = ReadFeatureFile("./dataset/lastfm/Arm_FeatureVectors_2.dat")  # 读入itemfeature
        ClusterFeatureVectors = ReadFeatureFile("./dataset/lastfm/lastfm_40clusters.dat")

    return UserFeatureVectors, ItemFeatureVectors, ClusterFeatureVectors, Timevectors

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
    if dataset == "kuai":
        filename = "./dataset/kuairec/40clusters_labels.json"
        label = read_json(filename)
        
    return label

def output_transform(output,hypernet_setting):
    if hypernet_setting.rank == -1:
        if torch.is_tensor(output):
            Theta_Matrix = output.reshape(-1, hypernet_setting.item_dimension, hypernet_setting.user_observed_dimension)

            # x = output[:,:hypernet_setting.rank*hypernet_setting.user_observed_dimension]
            # x = x.reshape(-1, hypernet_setting.user_observed_dimension, hypernet_setting.rank)
            # y = output[:,-hypernet_setting.rank*hypernet_setting.item_dimension:]
            # y = y.reshape(-1,hypernet_setting.rank,hypernet_setting.item_dimension)
            # Theta_Matrix = torch.matmul(x,y)
            # Theta_Matrix = torch.einsum('bi, bj -> bij',output[:,:config.FactorUCB_setting.user_observed_dimension],output[:,-config.FactorUCB_setting.item_dimension:])
        else:
            Theta_Matrix = output.reshape(hypernet_setting.item_dimension, hypernet_setting.user_observed_dimension)
            # x = output[:hypernet_setting.rank*hypernet_setting.user_observed_dimension]
            # x = x.reshape(hypernet_setting.user_observed_dimension,hypernet_setting.rank)
            # y = output[-hypernet_setting.rank*hypernet_setting.item_dimension:]
            # y = y.reshape(hypernet_setting.rank,hypernet_setting.item_dimension)
            # Theta_Matrix = np.matmul(x, y)


    else:
        if torch.is_tensor(output):
            # Theta_Matrix = output.reshape(config.FactorUCB_setting.item_dimension,config.FactorUCB_setting.user_observed_dimension)
            x = output[:,:hypernet_setting.rank*hypernet_setting.user_observed_dimension]
            x = x.reshape(-1,hypernet_setting.user_observed_dimension,hypernet_setting.rank)
            y = output[:,-hypernet_setting.rank*hypernet_setting.item_dimension:]
            y = y.reshape(-1,hypernet_setting.rank,hypernet_setting.item_dimension)
            Theta_Matrix = torch.matmul(x,y)
            # Theta_Matrix = torch.einsum('bi, bj -> bij',output[:,:config.FactorUCB_setting.user_observed_dimension],output[:,-config.FactorUCB_setting.item_dimension:])
        else:
            x = output[:hypernet_setting.rank*hypernet_setting.user_observed_dimension]
            x = x.reshape(hypernet_setting.user_observed_dimension,hypernet_setting.rank)
            y = output[-hypernet_setting.rank*hypernet_setting.item_dimension:]
            y = y.reshape(hypernet_setting.rank,hypernet_setting.item_dimension)
            Theta_Matrix = np.matmul(x, y)
    return Theta_Matrix   


