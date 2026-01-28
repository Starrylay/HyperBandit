import RUN 
from result import AlgResult
import FactorUCB
import LinUCB
import ColinUCB
import Baseline.dLinUCB as dLinUCB
import Baseline.HybridLinUCB as HybridLinUCB
import Baseline.factorUCB as factorUCB
import Baseline.ADTS as ADTS
import config
import time
import argparse
import csv
import pandas as pd
import datetime
import torch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='NYC',help='kuai,NYC,TKY')
    parser.add_argument('--baseline', default='LinUCB')
    parser.add_argument('--time_embedding', type=str ,default="learn",help='"glove" "onehot" "polar"')
    parser.add_argument('--train',type=bool, default=True)
    parser.add_argument('--user_input',type=bool, default=False)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--rank',type=int, default=-1,help='-1, 10')
    parser.add_argument('--feature',type=str, default="glove_pca",help='LLM_pca,glove')
    parser.add_argument('--gpu',type=int, default=2)
    parser.add_argument('--train_window',type=int, default=5000)
    parser.add_argument('--result_path',type=str, default=f"./TOIS_result/log_temp.csv")
    parser.add_argument('--warm_start', action='store_true', help='warm start the data.')

    args = parser.parse_args()
        # 检查CUDA是否可用
    args.dataset = "NYC"
    args.time_embedding = "polar" #"polar"
    args.train = True
    args.user_input = False
    args.sample_rate= 0.3
    args.rank = 2
    args.feature = "glove_pca"#"LLM_full_pca"
    args.warm_start = True
    args.baseline = "FactorUCB"

    print(args)
    #global 实例
    # LinUCB_setting_item = config.LinUCB_setting()
    # algresult.algorithms["LinUCB_ItemBased"] = LinUCB.LinUCBAlgorithm_ItemBased(LinUCB_setting_item)
    # LinUCB_setting = config.LinUCB_setting()
    # algresult.algorithms["LinUCB"] = LinUCB.LinUCBAlgorithm_UserBased(LinUCB_setting)
    # algresult.algorithms["HybridLinUCB"] = HybridLinUCB.Hybrid_LinUCBAlgorithm()
    # algresult.algorithms["HyperBandit fixed Theta"] = FactorUCB.FactorUCBAlgorithm() # 实例1
    # HyperBandit_fixed_mlp_setting = config.FactorUCB_setting(rank= -1)
    # algresult.algorithms["HyperBandit fixed mlp"] = FactorUCB.FactorUCBAlgorithm(HyperBandit_fixed_mlp_setting) 
    # algresult.algorithms["FactorUCB w/o W"] = factorUCB.FactorUCBAlgorithm(W_type = "None")
    # algresult.algorithms["FactorUCB"] = factorUCB.FactorUCBAlgorithm(W_type = "Have")
    # algresult.algorithms["DLinUCB"] = dLinUCB.DLinUCBAlgorithm()
    # # algresult.algorithms["ColinUCB w/o W"] = ColinUCB.CoLinUCBAlgorithm(W_type = "None")
    # # algresult.algorithms["ColinUCB"] = ColinUCB.CoLinUCBAlgorithm(W_type = "Have")
    # algresult.algorithms["ADTS"] = ADTS.AdaptiveThompson() 0.5347747802734375 train time: 0.6230242252349854
   
    for i in [5]:# 测试次数 2,3
        datasetlist = [args.dataset] #,,'NYC','TKY'
        for dataset in datasetlist:
            reward_result = []
            time_result = []
            epoch_result = []
            rectime_result = []
            count = 0
            alglist = [args.baseline]
            # ["HyperBandit"]#,"LinUCB","DLinUCB","ADTS","HybridLinUCB",]#["LinUCB","DLinUCB"] "FactorUCB"
                
            algresult = AlgResult()
            for algname in alglist :
                # HyperBandit_fixed_theta_setting = config.FactorUCB_setting()
                # algresult.algorithms["HyperBandit fixed Theta"] = FactorUCB.FactorUCBAlgorithm(HyperBandit_fixed_theta_setting) # 实例1
                # LinUCB_setting_item = config.LinUCB_setting()
                # algresult.algorithms["LinUCB_ItemBased"] = LinUCB.LinUCBAlgorithm_ItemBased(LinUCB_setting_item)
                # algresult.algorithms["FactorUCB w/o W"] = factorUCB.FactorUCBAlgorithm(W_type = "None")
                if  algname == "HyperBandit fixed mlp":
                    HyperBandit_fixed_mlp_setting = config.FactorUCB_setting(args)
                    algresult.algorithms["HyperBandit fixed mlp"] = FactorUCB.FactorUCBAlgorithm(HyperBandit_fixed_mlp_setting) 
                if algname == "HyperBandit":
                    HyperBandit_setting = config.FactorUCB_setting(args)
                    algresult.algorithms["HyperBandit"] = FactorUCB.FactorUCBAlgorithm(HyperBandit_setting) # 实例2  
                if algname == "FactorUCB":
                    Factor_setting = config.FactorUCB_setting(args)
                    algresult.algorithms["FactorUCB"] = factorUCB.FactorUCBAlgorithm(W_type = "Have",setting = Factor_setting)
                if algname == "HybridLinUCB":
                    HybridLinUCB_setting = config.HybridLinUCB_setting(args)
                    algresult.algorithms["HybridLinUCB"] = HybridLinUCB.Hybrid_LinUCBAlgorithm(HybridLinUCB_setting)
                if algname == "LinUCB":
                    LinUCB_setting = config.LinUCB_setting(args)
                    algresult.algorithms["LinUCB"] = LinUCB.LinUCBAlgorithm_UserBased(LinUCB_setting)
                if algname == "DLinUCB":
                    dlinUCB_setting = config.dLinUCB_setting(args)
                    algresult.algorithms["DLinUCB"] = dLinUCB.DLinUCBAlgorithm(dlinUCB_setting)
                if algname == "ADTS":
                    ADTS_setting  = config.ADTS_setting(args)
                    algresult.algorithms["ADTS"] = ADTS.AdaptiveThompson(ADTS_setting)

                # for alg_name, alg in algresult.algorithms.items():
                algresult.AlgReward[algname] = []
                algresult.AlgPicked[algname] = []
                algresult.AlgRegret[algname] = []
                algresult.BatchCumlateRegret[algname] = []
                algresult.BatchCumlateReward[algname] = []
                algresult.AlgRewardRatio_vsRandom[algname] = []
                algresult.traintime = 0  

            hypernet_setting = config.hypernet_setting(args)
            general_setting = config.general_setting(dataset)
            reward, traintime, epoch, rectime, count = RUN.run_rec(args, algresult, general_setting, hypernet_setting)  
            time_result.append(traintime)
            epoch_result.append(epoch)
            reward_result.append(reward)
            rectime_result.append(rectime)#--time_embedding=%%e --train=True --user_input=False --sample=%%s --rank=%%i --dataset="NYC" --feature=%%f
                
            # 将时间信息还有收益信息写入excel中   
            # print("payoff:", reward_result , "run_time:", time_result, "epoch:", epoch_result)
            # data = {"{}_{}_payoff_{}:".format(i,dataset,args.rank):reward_result, "{}_{}_traintime_{}:".format(i, dataset,args.rank):time_result, "{}_{}_epoch_{}:".format(i,dataset,args.rank):epoch_result, "{}_{}_rectime_{}:".format(i,dataset,args.rank):rectime_result, "count":count}
            # 指定 CSV 文件路径
                # 获取当前日期
            current_date = datetime.datetime.now()
            # 获取天部分，并确保它是两位数的格式
            day_formatted = current_date.strftime("%d")

            data = {
                "reward":reward,
                "time_embedding":args.time_embedding,
                "feature":args.feature,
                "rank":args.rank,
                "dataset":args.dataset,
                "train_window":args.train_window,
                "sample_rate":args.sample_rate,
                "warm_start":args.warm_start
                }
            

            # csv_file = "./result_log/static_data.csv"
            csv_file = args.result_path #f"./result_log/{day_formatted}_log_HyperBandit.csv"
            # 将数据转换为DataFrame
            df = pd.DataFrame(data,index = alglist)
            # 将DataFrame写入CSV文件，使用追加模式
            with open(csv_file, "a", newline="") as f:
                df.to_csv(f, header=True, index=True,sep=',')