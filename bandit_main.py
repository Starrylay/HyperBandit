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
import os


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
    parser.add_argument('--result_path',type=str, default=f"./TOIS_result_new/")
    parser.add_argument('--warm_start', action='store_true', help='warm start the data.')
    parser.add_argument('--isupdate', action='store_true', help='update the bandit policy.')
    parser.add_argument('--is_hypernet', action='store_true', help='use hypernet.')
    parser.add_argument('--pool_size', default=25,type=int, help='10,15,20,25')
    parser.add_argument('--llm_dada_size',type=int, default=10000, help='debug mode.')

    args = parser.parse_args()

    print(args)

    for i in [1]:# 测试次数 2,3
        datasetlist = [args.dataset] #,,'NYC','TKY'
        for dataset in datasetlist:
            reward_result = []
            time_result = []
            epoch_result = []
            rectime_result = []
            count = 0
            alglist = [args.baseline]
            # ["HyperBandit"]#,"LinUCB","DLinUCB","ADTS","HybridLinUCB",]#["LinUCB","DLinUCB"] "FactorUCB"  NeuralLinear 
            algresult = AlgResult()
            # init setting
            hypernet_setting = config.hypernet_setting(args)
            general_setting = config.general_setting(dataset, args)
            config.test_setting.from_args(args)
            # 打印类变量
            print("类变量:", vars(config.test_setting)) 

            for algname in alglist :
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

            
            rewards, traintime, epoch, rectime, count = RUN.run_rec(args, algresult, general_setting, hypernet_setting)  
            time_result.append(traintime)
            epoch_result.append(epoch)
            # reward_result.append(reward )
            rectime_result.append(rectime) #--time_embedding=%%e --train=True --user_input=False --sample=%%s --rank=%%i --dataset="NYC" --feature=%%f
            current_date = datetime.datetime.now()
            time_name = current_date.strftime("%Y-%m-%d")

            csv_file = args.result_path + f"{time_name}.csv "
            
            for alg_name, _ in algresult.AlgRewardRatio_vsRandom.items():
                reward = algresult.AlgRewardRatio_vsRandom[alg_name][-1]
                if alg_name == "HyperBandit":
                    data = {
                        'algorithm':alg_name,
                        "reward":reward,
                        "time_embedding":args.time_embedding,
                        "feature":args.feature,
                        "rank":args.rank,
                        "dataset":args.dataset,
                        "warm_start":args.warm_start,
                        "isupdate":args.isupdate,
                        "is_hypernet":args.is_hypernet,
                        "pool_size":args.pool_size
                        }
                else:
                    data = {
                        'algorithm':alg_name,
                        "reward":reward,
                        "time_embedding":"None",
                        "feature":args.feature,
                        "rank":"None",
                        "dataset":args.dataset,
                        "warm_start":args.warm_start
                        }

                df = pd.DataFrame(data, index=[0])
                # 将DataFrame写入CSV文件，使用追加模式


                os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
                with open(csv_file, "a", newline="") as f:
                    df.to_csv(f, header=True, index=True,sep=',')