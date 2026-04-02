import torch
import numpy as np
import argparse


def parse():
    p = argparse.ArgumentParser("BLTDMGDL: Variational Bayesian Inference with Hybrid Graph Deep Learning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=str, default='DATA1_zhang', help='data name ')
    p.add_argument('--dataset', type=str, default='./DATA', help='dataset name')
    p.add_argument('--model-name', type=str, default='BLTDMGDL', help='BLTDMGDL')
    p.add_argument('--activation', type=str, default='tanh', help='activation layer between MGConvs')
    p.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')
    p.add_argument('--rank', type=int, default=10, help='rank')
    p.add_argument('--lr', type=float, default=0.001, help='learning rate')
    p.add_argument('--L2',type=float,default=1e-4,help = 'weight_decay')
    p.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    return p.parse_args()



def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def Loss_fun_opt(FGHW, GHW,lambdas,args):
    lambdas = lambdas.diag().to(args.device)
    Ng, Nh, Nw = args.G_num, args.H_num, args.W_num
    G_emb, H_emb, W_emb = GHW[:Ng], GHW[Ng:Ng + Nh], GHW[Ng + Nh:]
    FG_emb, FH_emb, FW_emb = FGHW[:Ng], FGHW[Ng:Ng + Nh], FGHW[Ng + Nh:]
    loss_g = torch.trace(lambdas @ (G_emb-FG_emb).t() @ (G_emb-FG_emb))
    loss_h = torch.trace(lambdas @ (H_emb - FH_emb).t() @ (H_emb - FH_emb))
    loss_w = torch.trace(lambdas @ (W_emb - FW_emb).t() @ (W_emb - FW_emb))
    return (loss_h+loss_g+loss_w)/(Ng+Nh+Nw)



def Const_hyper(args,sim_H,sim_W,train_tensor):
    #sg
    for i in range(len(sim_H)):
        sim_H[i,i]=1
    sim_H = (sim_H+sim_H.t())/2
    ind_H = torch.nonzero(sim_H)
    edge_H = ind_H.t().to(args.device)
    w_H = sim_H[ind_H[:,0],ind_H[:,1]]
    args.edge_H = edge_H.to(args.device)
    args.w_H = w_H.to(args.device)

    #sh
    for i in range(len(sim_W)):
        sim_W[i,i]=1
    sim_W = (sim_W + sim_W.t()) / 2
    ind_W = torch.nonzero(sim_W)
    edge_W = ind_W.t().to(args.device)
    w_W = sim_W[ind_W[:,0],ind_W[:,1]]
    args.edge_W = edge_W.to(args.device)
    args.w_W = w_W.to(args.device)

    #sghw
    Ng,Nh,Nw = train_tensor.shape
    indice_T = torch.nonzero(train_tensor)
    indice_T[:,1] += Ng
    indice_T[:, 2] += Ng+Nh
    NT = indice_T.shape[0]
    edges_T = torch.arange(NT).unsqueeze(1).expand(NT, 3)+Ng+Nh
    hyper_3 = torch.cat((indice_T.reshape(1, NT * 3), edges_T.reshape(1, NT * 3)), dim=0)
    wghw = torch.cat([torch.ones(1,3) for idx in indice_T],dim=1).squeeze()
    synergy_graph = hyper_3
    args.edge_HGW = synergy_graph.to(args.device)
    args.weight = wghw.to(args.device)
    return args


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)  
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)  
    # the degree of the node
    DV = np.sum(H * W, axis=1)  
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  

    invDE = np.mat(np.diag(np.power(DE, -1)))  
    invDE[np.isinf(invDE)] = 0
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2[np.isinf(DV2)] = 0
    W = np.mat(np.diag(W)) 
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G