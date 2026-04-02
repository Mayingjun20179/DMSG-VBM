import numpy as np
import tensorly as tl
# tl.set_backend('numpy')
import torch
from utils import *
from DATA1.data import GetData as GetData1
from DATA2.data import GetData as GetData2
from DMSGVBM_model import Model,set_seed
from Evaluate import cv_tensor_model_evaluate,get_metrics,cal_recall_ndcg
import pandas as pd


class Experiments(object):
    def __init__(self, GHW_data, model_name='DMSGVBM',**kwargs):
        super().__init__()
        self.GHW_data = GHW_data
        self.model_name = model_name
        self.parameters = kwargs

    def CV_triplet(self,args,prop):
        k_folds = 5
        index_matrix = np.array(np.where(self.GHW_data.adj_tensor == 1))
        positive_num = index_matrix.shape[1]
        sample_num_per_fold = int(positive_num / k_folds)

        np.random.shuffle(index_matrix.T)

        metrics_tensor = np.zeros((1, 7))
        result_100 = []
        for k in range(k_folds):

            train_tensor = np.array(self.GHW_data.adj_tensor, copy=True)
            if k != k_folds - 1:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_tensor[test_index] = 0
            train_tensor = torch.tensor(train_tensor,dtype=torch.float32)

            ###Constructing Hybrid graphs
            args = Const_hyper(args,self.GHW_data.mic_sim,self.GHW_data.dis_sim,train_tensor)

            #Predictions of the DMSGVBM model
            self.model = Model(args, self.model_name)
            predict_tensor = self.model.DMSGVBM(train_tensor, args)

            #Randomized Generation of Test Negatives and Model Evaluation
            for i in range(20):
                jieguo = cv_tensor_model_evaluate(self.GHW_data, predict_tensor, test_index, i, prop)
                metrics_tensor = metrics_tensor + jieguo
                result_100.append(jieguo)

        # Get result
        result = pd.DataFrame(np.around(metrics_tensor / 100, decimals=4)[:,0:3],columns=['AUPR','AUC','F1'])
        result_100 = pd.DataFrame(np.array(result_100)[:,0:3],columns=['AUPR','AUC','F1'])
        print(result)
        return result, result_100

    def CV_drug(self,args):
        k_folds = 5
        hw_matrix = self.GHW_data.adj_tensor.sum(0)
        ind_matrix = np.array(np.where(hw_matrix > 0))
        pair_num = ind_matrix.shape[1]
        sample_num_per_fold = int(pair_num / k_folds)
        np.random.shuffle(ind_matrix.T)
        metrics = 0
        for k in range(k_folds):

            train_tensor = np.array(self.GHW_data.adj_tensor, copy=True)
            if k != k_folds - 1:
                test_index = tuple(ind_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                test_index = tuple(ind_matrix[:, k * sample_num_per_fold:])

            train_tensor[:,test_index[0],test_index[1]] = 0
            train_tensor1 =  np.array(self.GHW_data.adj_tensor, copy=True)
            for ss in range(test_index[0].shape[0]):
                train_tensor1[:,test_index[0][ss],test_index[1][ss]]=0
            print(np.abs(train_tensor-train_tensor1).sum())
            train_tensor = torch.tensor(train_tensor,dtype=torch.float32)
            ###
            args = Const_hyper(args,self.GHW_data.mic_sim,self.GHW_data.dis_sim,train_tensor)
            self.model = Model(args, self.model_name)
            predict_tensor = self.model.DMSGVBM(train_tensor, args).numpy()
            adj_tensor = self.GHW_data.adj_tensor.numpy()
            result_g = 0
            N_test = np.array(test_index).shape[1]
            for t in range(N_test):
                predict_score = predict_tensor[:,test_index[0][t], test_index[1][t]]
                predict_matrix = np.mat(predict_score)
                real_score = adj_tensor[:,test_index[0][t], test_index[1][t]]
                real_matrix = np.mat(real_score)
                # result_g += np.array(get_metrics(real_matrix, predict_matrix))
                result_g1 = np.array(get_metrics(real_matrix, predict_matrix))[0:3]  #只保留aupr,auc,f1
                result_g2 = cal_recall_ndcg(real_score, predict_score,args.topK) #保留recall,ndcg
                result_g += np.concatenate((result_g1,result_g2))
            metrics += result_g/N_test
            print(result_g/N_test)

        result = pd.DataFrame(np.around(metrics[np.newaxis, :] / k_folds, decimals=4)[:,3:],
                               columns=['HR@1', 'HR@5', 'HR@10', 'NDCG@1', 'NDCG@5', 'NDCG@10'])
        return result

    def CV_mic(self,args):
        k_folds = 5
        gw_matrix = self.GHW_data.adj_tensor.sum(1)
        ind_matrix = np.array(np.where(gw_matrix > 0))
        pair_num = ind_matrix.shape[1]
        sample_num_per_fold = int(pair_num / k_folds)
        np.random.shuffle(ind_matrix.T)
        metrics = 0
        for k in range(k_folds):

            train_tensor = np.array(self.GHW_data.adj_tensor, copy=True)
            if k != k_folds - 1:
                test_index = tuple(ind_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                test_index = tuple(ind_matrix[:, k * sample_num_per_fold:])

            train_tensor[test_index[0],:,test_index[1]] = 0
            train_tensor1 = np.array(self.GHW_data.adj_tensor, copy=True)
            for ss in range(test_index[0].shape[0]):
                train_tensor1[test_index[0][ss],:,test_index[1][ss]]=0
            print(np.abs(train_tensor-train_tensor1).sum())
            train_tensor = torch.tensor(train_tensor,dtype=torch.float32)
            ###
            args = Const_hyper(args,self.GHW_data.mic_sim,self.GHW_data.dis_sim,train_tensor)
            self.model = Model(args, self.model_name)
            predict_tensor = self.model.DMSGVBM(train_tensor, args).numpy()
            adj_tensor = self.GHW_data.adj_tensor.numpy()
            result_h = 0
            N_test = np.array(test_index).shape[1]
            for t in range(N_test):
                predict_score = predict_tensor[test_index[0][t],:, test_index[1][t]]
                predict_matrix = np.mat(predict_score)
                real_score = adj_tensor[test_index[0][t],:, test_index[1][t]]
                real_matrix = np.mat(real_score)
                result_h1 = np.array(get_metrics(real_matrix, predict_matrix))[0:3]
                result_h2 = cal_recall_ndcg(real_score, predict_score,args.topK)
                result_h += np.concatenate((result_h1,result_h2))
            metrics += result_h/N_test
            print(result_h/N_test)

        result = pd.DataFrame(np.around(metrics[np.newaxis, :] / k_folds, decimals=4)[:, 3:],
                              columns=['HR@1', 'HR@5', 'HR@10', 'NDCG@1', 'NDCG@5', 'NDCG@10'])
        return result

    def CV_dis(self,args):
        k_folds = 5
        gh_matrix = self.GHW_data.adj_tensor.sum(2)
        ind_matrix = np.array(np.where(gh_matrix > 0))
        pair_num = ind_matrix.shape[1]
        sample_num_per_fold = int(pair_num / k_folds)
        np.random.shuffle(ind_matrix.T)
        metrics = 0
        for k in range(k_folds):
            train_tensor = np.array(self.GHW_data.adj_tensor, copy=True)
            if k != k_folds - 1:
                test_index = tuple(ind_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                test_index = tuple(ind_matrix[:, k * sample_num_per_fold:])

            train_tensor[test_index[0],test_index[1],:] = 0
            train_tensor1 = np.array(self.GHW_data.adj_tensor, copy=True)
            for ss in range(test_index[0].shape[0]):
                train_tensor1[test_index[0][ss],test_index[1][ss],:]=0
            print(np.abs(train_tensor-train_tensor1).sum())
            train_tensor = torch.tensor(train_tensor,dtype=torch.float32)
            ##
            args = Const_hyper(args,self.GHW_data.mic_sim,self.GHW_data.dis_sim,train_tensor)
            self.model = Model(args, self.model_name)
            predict_tensor = self.model.DMSGVBM(train_tensor, args).numpy()
            adj_tensor = self.GHW_data.adj_tensor.numpy()
            result_w = 0
            N_test = np.array(test_index).shape[1]
            for t in range(N_test):
                predict_score = predict_tensor[test_index[0][t], test_index[1][t],:]
                predict_matrix = np.mat(predict_score)
                real_score = adj_tensor[test_index[0][t], test_index[1][t],:]
                real_matrix = np.mat(real_score)
                result_w1 = np.array(get_metrics(real_matrix, predict_matrix))[0:3]
                result_w2 = cal_recall_ndcg(real_score, predict_score,args.topK)
                result_w += np.concatenate((result_w1,result_w2))
            metrics += result_w/N_test
            print(result_w/N_test)

        result = pd.DataFrame(np.around(metrics[np.newaxis, :] / k_folds, decimals=4)[:, 3:],
                              columns=['HR@1', 'HR@5', 'HR@10', 'NDCG@1', 'NDCG@5', 'NDCG@10'])
        return result


if __name__ == '__main__':
    #
    args = parse()
    args.device = torch.device('cuda:0')
    ##### data1
    seed = 1
    set_seed(seed)
    root = './DATA1'
    GHW_data = GetData1(root)
    args.durg_inf = GHW_data.batch_drug.to(args.device)
    args.use_GMP = True
    args.G_num,args.H_num,args.W_num = GHW_data.N_drug,GHW_data.N_mic,GHW_data.N_dis
    experiment = Experiments(GHW_data, model_name='DMSGVBM')

    # # CV_triplet
    prop = [1,10,100]   #1/ρ
    for kk in prop:
        args.triple = True
        result_CV_triplet, result_CV_triplet100 = experiment.CV_triplet(args,kk)
        file_path = './result1/DMSGVBM_triplet' +'_prop_'+str(kk)+ '.txt'
        result_CV_triplet.to_csv(file_path, index=False, sep='\t')
        file_path = './result1/DMSGVBM_triplet_100' +'_prop_'+str(kk)+ '.txt'
        result_CV_triplet100.to_csv(file_path,sep='\t')
        print(result_CV_triplet)
    # #
    #CV_drug
    args.triple = False
    args.topK = [1,5,10]
    result_CV_drug = experiment.CV_drug(args)
    file_path = './result1/DMSGVBM_drug' + '.txt'
    result_CV_drug.to_csv(file_path, index=False, sep='\t')
    print(result_CV_drug)

    # CV_mic
    args.triple = False
    args.topK = [1, 5, 10]
    result_CV_mic = experiment.CV_mic(args)
    file_path = './result1/DMSGVBM_mic' + '.txt'
    result_CV_mic.to_csv(file_path,index=False, sep='\t')
    print(result_CV_mic)

    #CV_dis
    args.triple = False
    args.topK = [1,5,10]
    result_CV_dis = experiment.CV_dis(args)
    file_path = './result1/DMSGVBM_dis' + '.txt'
    result_CV_dis.to_csv(file_path,index=False, sep='\t')
    print(result_CV_dis)


    #####data2
    seed = 1
    set_seed(seed)
    root = './DATA2'
    GHW_data = GetData2(root)
    args.durg_inf = GHW_data.batch_drug.to(args.device)
    args.use_GMP = True
    args.G_num,args.H_num,args.W_num = GHW_data.N_drug,GHW_data.N_mic,GHW_data.N_dis
    experiment = Experiments(GHW_data, model_name='DMSGVBM')
    # CV_triplet
    prop = [1,10,100]   #1/ρ
    for kk in prop:
        args.triple = True
        result_CV_triplet, result_CV_triplet100 = experiment.CV_triplet(args,kk)
        file_path = './result2/DMSGVBM_triplet' +'_prop_'+str(kk)+ '.txt'
        result_CV_triplet.to_csv(file_path,index=False,sep='\t')
        file_path = './result2/DMSGVBM_triplet_100' +'_prop_'+str(kk)+ '.txt'
        result_CV_triplet100.to_csv(file_path,sep='\t')
        print(result_CV_triplet)

    #CV_drug
    args.triple = False
    args.topK = [1,5,10]
    result_CV_drug = experiment.CV_drug(args)
    file_path = './result2/DMSGVBM_drug' + '.txt'
    result_CV_drug.to_csv(file_path,index=False,sep='\t')
    print(result_CV_drug)
    #
    #CV_mic
    args.triple = False
    result_CV_mic = experiment.CV_mic(args)
    file_path = './result2/DMSGVBM_mic' + '.txt'
    result_CV_mic.to_csv(file_path,index=False,sep='\t')
    print(result_CV_mic)

    #CV_dis
    args.triple = False
    result_CV_dis = experiment.CV_dis(args)
    file_path = './result2/DMSGVBM_dis' + '.txt'
    result_CV_dis.to_csv(file_path,index=False,sep='\t')
    print(result_CV_dis)

