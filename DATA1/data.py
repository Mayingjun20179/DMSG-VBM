import csv
import os.path as osp
import numpy as np
import scipy.io as sio
import torch

from DATA1.process_smiles import *


class GetData(object):
    def __init__(self, root):
        super().__init__()
        # self.root = osp.join(root, 'data_v2.mat')
        self.root = root
        self.batch_drug, self.dis_sim, self.mic_sim,self.adj_tensor,self.index_0,self.N_0 = self.__get_data__()
        self.N_drug, self.N_mic, self.N_dis = self.adj_tensor.shape


    def __get_data__(self):
        smiles_file = osp.join(self.root,'drug_smiles_270.csv')
        batch_drug = drug_fea_process(smiles_file)
        dis_file = osp.join(self.root,'dis_sim.txt')
        dis_sim = np.loadtxt(dis_file, delimiter='\t')
        mic_file = osp.join(self.root,'mic_sim_NinimHMDA.txt')
        mic_sim = np.loadtxt(mic_file, delimiter='\t')
        dis_sim = torch.from_numpy(dis_sim).type(torch.float32)
        mic_sim = torch.from_numpy(mic_sim).type(torch.float32)
        #
        adj_file = osp.join(self.root,'adj_del_4mic_myid.txt')
        adj_ind = np.loadtxt(adj_file)
        adj_ind = np.array(adj_ind,dtype=np.int64)
        adj_ind = torch.from_numpy(adj_ind)
        N_drug,N_mic,N_dis,_ = adj_ind.max(dim=0).values+1
        adj_tensor = torch.zeros(N_drug,N_mic,N_dis)
        adj_tensor[adj_ind[:,0],adj_ind[:,1],adj_ind[:,2]]=1

        #
        index_0 = np.array(np.where(adj_tensor.numpy() == 0)).T
        N_0 = index_0.shape[0]

        return batch_drug,dis_sim,mic_sim,adj_tensor,index_0,N_0

def drug_fea_process(smiles_file):
    reader = csv.reader(open(smiles_file,encoding='utf-8'))
    smile_graph = []
    for item in reader:
        smile = item[1]
        g = smile_to_graph(smile) #
        smile_graph.append(g)
    drug_num = len(smile_graph)
    dru_data = GraphDataset_v(xc=smile_graph, cid=[i for i in range(drug_num + 1)])
    dru_data = torch.utils.data.DataLoader(dataset=dru_data, batch_size=drug_num, shuffle=False,
                                           collate_fn=collate)
    for step, batch_drug in enumerate(dru_data):
        drug_data = batch_drug
    return drug_data