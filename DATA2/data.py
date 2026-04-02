import csv
import os.path as osp
import numpy as np
import scipy.io as sio
import torch
import pandas as pd


from DATA2.process_smiles import *


class GetData(object):
    def __init__(self, root):
        super().__init__()
        # self.root = osp.join(root, 'data_v2.mat')
        self.root = root
        self.batch_drug, self.dis_sim, self.mic_sim,self.adj_tensor,self.index_0,self.N_0 = self.__get_data__()
        self.N_drug, self.N_mic, self.N_dis = self.adj_tensor.shape


    def __get_data__(self):

        #drug
        drug_file = osp.join(self.root,'drug_inf.csv')
        batch_drug,drug_dict = drug_fea_process(drug_file)

        #disease
        dis_file = osp.join(self.root,'dis_sim.csv')
        dis_sim = pd.read_csv(dis_file, index_col=0)
        dis_sim_value = torch.from_numpy(dis_sim.to_numpy()).type(torch.float32)
        dis_dict = {index: value for value, index in enumerate(dis_sim.index)}

        #micro
        mic_file = osp.join(self.root,'micro_sim.csv')
        mic_sim = pd.read_csv(mic_file,index_col=0)
        mic_sim_value = torch.from_numpy(mic_sim.to_numpy()).type(torch.float32)
        mic_dict = {index:value for value,index in enumerate(mic_sim.index)}


        #
        adj_file = osp.join(self.root,'drug_micro_dis_triple.csv')
        adj_data = pd.read_csv(adj_file)
        adj_ind = [(drug_dict[adj_data.loc[i,'pubchem_id']],mic_dict[adj_data.loc[i,'micro_tid']],dis_dict[adj_data.loc[i,'dis_MESH']]) for i in range(adj_data.shape[0])]
        adj_ind = np.array(adj_ind, dtype=np.int64)


        adj_ind = torch.from_numpy(adj_ind)
        N_drug,N_mic,N_dis = adj_ind.max(dim=0).values+1
        adj_tensor = torch.zeros(N_drug,N_mic,N_dis)
        adj_tensor[adj_ind[:,0],adj_ind[:,1],adj_ind[:,2]]=1

        #
        index_0 = np.array(np.where(adj_tensor.numpy() == 0)).T
        N_0 = index_0.shape[0]

        return batch_drug,dis_sim_value,mic_sim_value,adj_tensor,index_0,N_0

def drug_fea_process(smiles_file):
    drug_inf = pd.read_csv(smiles_file)
    drug_dict = {index: value for value, index in enumerate(drug_inf['pubchem_id'])}

    smile_graph = []
    for index,row in drug_inf.iterrows():
        smile = row['smile']
        g = smile_to_graph(smile) 
        smile_graph.append(g)
    drug_num = len(smile_graph)
    dru_data = GraphDataset_v(xc=smile_graph, cid=[i for i in range(drug_num + 1)])
    dru_data = torch.utils.data.DataLoader(dataset=dru_data, batch_size=drug_num, shuffle=False,
                                           collate_fn=collate)
    for step, batch_drug in enumerate(dru_data):
        drug_data = batch_drug
    return drug_data,drug_dict