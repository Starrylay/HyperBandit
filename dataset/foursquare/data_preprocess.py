from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from scipy.sparse import csr_matrix
import csv
import random
import pickle
import json
import pandas as pd
def data_process_user(event_file: str ):
    user_documents_dict = {}
    with open(event_file, "r", encoding="latin-1") as f:
        f.readline()
        count = 0
        for line in f:
            count += 1
            line = line.split("\t")
            times_plit = line[7].split(' ')
            if times_plit[1] == 'Apr' and times_plit[2] >= '10': #Apr03到Apr09是我需要的预处理数据
                break
            if int(line[0]) not in user_documents_dict:
                user_documents_dict[int(line[0])] = [line[2]] # 用户去过的地点的类别
            else:
                user_documents_dict[int(line[0])].append(line[2])
        user_documents_list = []
        user_id_list =[]

        for key in user_documents_dict:
            user_id_list.append(key)
            tempstr = ' '.join(user_documents_dict[key])
            user_documents_list.append(tempstr)

    return user_id_list, user_documents_list, user_documents_dict

def time_process(time:list):
    hour = list(map(int, time[3].split(':')))[0]
    hour_shift = int(time[4]) / 60
    hour_true = hour + hour_shift
    if hour_true >= 8 and hour_true < 12:
        time_zone = "morning"
    elif hour_true >= 12 and hour_true < 14:
        time_zone = "noon"
    elif hour_true >= 14 and hour_true < 16:
        time_zone = "afternoon"
    elif hour_true >= 16 and hour_true < 22:
        time_zone = "night"
    else:
        time_zone = "rest"
    return  time_zone
def data_process_item(event_file: str ):
    item_documents = {}
    with open(event_file, "r", encoding="latin-1") as f:
        f.readline()
        count = 0
        for line in f:
            count += 1
            line = line.split("\t")
            times_plit = line[7].split(' ')
            if times_plit[1] == 'Apr' and times_plit[2] >= '10': #Apr03到Apr09是我需要的预处理数据
                break
            time_zone = time_process(times_plit)
            if line[2] not in item_documents:
                item_documents[line[2]] = [line[2]]# 地点的类别
                item_documents[line[2]].append(time_zone)
            else:
                item_documents[line[2]].append(line[2])
                item_documents[line[2]].append(time_zone)
        item_documents_list = []
        item_id_list =[]

        for key in item_documents:
            item_id_list.append(key)
            tempstr = ' '.join(item_documents[key])
            item_documents_list.append(tempstr)

    return item_id_list, item_documents_list

def data_process_event(event_file: str):
    user_id_list = []
    cat_id_list = []
    week_list = []
    time_list = []
    events = {}
    with open(event_file, "r", encoding="latin-1") as f:
        f.readline()
        count = 0
        for line in f:
            line = line.split("\t")
            time_split = line[7].split(' ')
            
            if time_split[1] == 'Apr' and time_split[2] < '10': #Apr03到Apr09是可观测数据,测试时不要
                count += 1
                continue

            time_zone = time_process(time_split)# 输出当前时间段
            time_list.append(time_zone)
            user_id_list.append(line[0])
            cat_id_list.append(line[2])
            week_list.append(time_split[0])

            events["user_ids"] = user_id_list
            events["cat_ids"] = cat_id_list
            events["week_periods"] = week_list
            events["time_periods"] = time_list

    return events

def cat_id_map(event_file: str):
    with open(event_file, "r", encoding="latin-1") as f:
        f.readline()
        cat_ids_list = []
        cat_ids_dict = {}
        for line in f:
            line = line.split("\t")
            cat_ids_list.append(line[2])
        unique_cat_ids = list(set(cat_ids_list))
        for i in range(len(unique_cat_ids)):
            cat_ids_dict[unique_cat_ids[i]] = str(i) #id : 新id
        # 将字典存储为文件
        with open('./cat_id_map_dict.pkl', 'wb') as f:
            pickle.dump(cat_ids_dict, f)


if __name__ == '__main__':

    #数据读入预处理
    event_file = r"./2 NYC and Tokyo Check-in Datase/dataset_TSMC2014_TKY.txt"

    # #用户观测向量生成
    # import os
    # for dirname, _, filenames in os.walk('./dataset/foursquare/'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))

    user_id_list, user_documents_list, user_documents_dict = data_process_user(event_file) # 返回前一周的user列表，元素是venue id

    # # 初始化 TfidfVectorizer
    # vectorizer = TfidfVectorizer()
    # # 计算 TF-IDF
    # tfidf_matrix = vectorizer.fit_transform(user_documents_list) # 每一行代表一个文章，每一列代表一个单词
    # #将稀疏矩阵转化为密集矩阵再转化为array
    # dense_matrix = np.asarray(csr_matrix.todense(tfidf_matrix))
    # # 创建 PCA 对象，指定降维后的维度为 25
    # pca = PCA(n_components=25)
    # # 对数据矩阵进行拟合和转换
    # X_projected = pca.fit_transform(dense_matrix)
    # #将降维后的向量与user id 匹配
    # filename = 'User_FeatureVectors.dat'
    # with open(filename, 'w', newline='') as f:
    #     for i in range(X_projected.shape[0]):
    #         new_line = str(user_id_list[i]) + '\t' + ';'.join(str(X_projected[i, :]).strip('[').strip(']').split()) + '\n'
    #         f.write(new_line)
    # 读取没有列名的CSV文件

    # data = pd.read_csv("./POI_Glove_FeatureVectors.csv", header=0, sep=' ')
    # FeatureVectors = {}
    # for index, row in data.iterrows():
    #     key = int(row[0])
    #     value = row[1:].tolist()
    #     FeatureVectors[key] = np.array(value)
    # # cat_id_map_dict = {}

    # # with open("./dataset/foursquare/cat_id_map_dict.json", 'r') as f:
    # #     cat_id_map_dict = json.load(f)
    # with open("./cat_id_map_dict.json", 'r') as f:
    #     cat_id_map_dict = json.load(f)
    # users_glove_feature = {}
    # for user in user_documents_dict:
    #     list = user_documents_dict[user]
    #     newlist =  [cat_id_map_dict.get(item) for item in list]
    #     featurelist = [FeatureVectors.get(item) for item in newlist]
    #     featuremean = np.mean(featurelist,axis = 0)
    #     users_glove_feature[user] = featuremean



# POI 类别id映射
   # cat_id_map(event_file)


# POI 类别观测向量生成

    # item_id_list, item_documents_list = data_process_item(event_file)
    # # 重新映射cat id
    # with open('./cat_id_map_dict.pkl', 'rb') as f:
    #     cat_id_map_dict = pickle.load(f)
    # for i in range(len(item_id_list)):
    #     item_id_list[i] = cat_id_map_dict[item_id_list[i]]
    # # 初始化 TfidfVectorizer
    # vectorizer = TfidfVectorizer()
    # # 计算 TF-IDF
    # tfidf_matrix = vectorizer.fit_transform(item_documents_list) # 每一行代表一个文章，每一列代表一个单词
    # #将稀疏矩阵转化为密集矩阵再转化为array
    # dense_matrix = np.asarray(csr_matrix.todense(tfidf_matrix))
    # # 创建 PCA 对象，指定降维后的维度为 25
    # pca = PCA(n_components=10)
    # # 对数据矩阵进行拟合和转换
    # X_projected = pca.fit_transform(dense_matrix)
    # #将降维后的向量与user id 匹配
    # filename = 'POI_Cat_FeatureVectors.dat'
    # with open(filename, 'w', newline='') as f:
    #     for i in range(X_projected.shape[0]):
    #         new_line = str(item_id_list[i]) + '\t' + ';'.join(str(X_projected[i, :]).strip('[').strip(']').split()) + '\n'
    #         f.write(new_line)



#  生成事件流包含
    """
        events["user_ids"] = user_id_list
        events["cat_ids"] = cat_id_list
        events["week_periods"] = week_list
        events["time_periods"] = time_list
    """
    events = data_process_event(event_file)
    #生成armpool
    #给出cat(类别)字典
    unique_cat_ids = list(set(events["cat_ids"]))
    
    #将catid从0重新映射
    with open('./TKY/cat_id_map_dict.json', 'r') as f:
        cat_id_map_dict = json.load(f)
    for i in range(len(events["cat_ids"])):
        events["cat_ids"][i] = cat_id_map_dict[events["cat_ids"][i]]
    for i in range(len(unique_cat_ids)):
        unique_cat_ids[i] = cat_id_map_dict[unique_cat_ids[i]]
    
    #采样
    num_samples = 24
    # 从my_list中随机选择num_samples个元素，并从中排除exclude_list中的元素
    armpool = []
    
    for i in range(len( events["user_ids"])):
        #随机选出armpool剩余元素，注意避开用户实际选择元素
        arms_list = random.sample([x for x in unique_cat_ids if x != events["cat_ids"][i]], num_samples)
        #添加用户选择元素到第一个位置
        arms_list.insert(0, events["cat_ids"][i])
        arms_list = list(map(str, arms_list))  # 使用map函数将int换为str
        armpool.append(arms_list)
    
    filename = './TKY/Events_NYC.dat'
    with open(filename, 'w', newline='') as f:
        for i in range(len(events["user_ids"])):
            new_line = str(events["user_ids"][i]) + '\t\t' + ' '.join(armpool[i]) + '\t\t' + events["week_periods"][i] + '\t\t' + events["time_periods"][i] + '\n'
            f.write(new_line)








