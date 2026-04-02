import tensorly as tl
import numpy as np
import torch
from tensorly.tenalg import khatri_rao
tl.set_backend('pytorch')
import os
import random
from F_mlayer_model import *
from utils import Loss_fun_opt


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)




class Model(object):
    def __init__(self,args,name='DMSGVBM',**kwargs):
        super().__init__()
        self.name = name
        self.model = Hybridgraphattention(
            GcnEncoder(dim_H=args.rank, dim_W=args.rank,output=args.rank,args = args),
            HgnnEncoder(in_channels = args.rank,out_channels=args.rank,args = args),args).to(args.device)
        self.paramater = kwargs
    def DMSGVBM(self,Y,args):
        I, J, K = Y.shape
        R = args.rank
        G_mu = torch.randn(I, R)
        G_sigma = torch.tile(torch.eye(R).unsqueeze(0), (I, 1, 1))

        H_mu = torch.randn(J, R)
        H_sigma = torch.tile(torch.eye(R).unsqueeze(0), (J, 1, 1))

        W_mu = torch.randn(K, R)
        W_sigma = torch.tile(torch.eye(R).unsqueeze(0), (K, 1, 1))

        #
        lambdas = torch.ones(R)
        GHW = torch.cat((G_mu,H_mu,W_mu),dim=0).to(args.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.L2)

        for epoch in range(args.epochs):
            self.model.train()
            loss_train = 0
            optimizer.zero_grad()

            FGHW = self.model(args)
            loss = Loss_fun_opt(FGHW, GHW,lambdas,args)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        FGHW = FGHW.detach().to('cpu')
        Fg, Fh, Fw = FGHW[:I], FGHW[I:I + J], FGHW[I + J:]
        print(FGHW.norm())
        #
        Bg = G_sigma.reshape(I, R * R)
        Bh = H_sigma.reshape(J, R * R)
        Bw = W_sigma.reshape(K, R * R)

        kx = torch.ones((I, J, K))
        c = 5
        Aijk = (c * Y - 1 + Y) / 2
        lam_kx = self.jisuan_lamb(kx)  
        Bijk = (c * Y + 1 - Y) * lam_kx
        weight = torch.ones(R)
        P0 = tl.cp_to_tensor((weight, [G_mu, H_mu, W_mu]))
        tol = 5e-4

        # Model learning
        for it in range(1,501):
            if it % 20 == 0:
                print(it)

            #############E_step
            #update G
            ENZZT = (tl.unfold(Bijk, 0) @ khatri_rao([Bh, Bw])).reshape(I, R, R)
            FslashY = khatri_rao([H_mu, W_mu]).T @ tl.unfold(Aijk, 0).T
            for i in range(I):
                G_sigma[i, :, :] = torch.linalg.inv(2 * ENZZT[i,:, :] + lambdas.diag())
                G_mu[i, :] = G_sigma[i,:, :] @ (FslashY[:, i] + lambdas.diag() @ Fg[i,:])
            Bg = G_sigma.reshape(I, R * R) + khatri_rao([G_mu.T, G_mu.T]).T

            #update H
            ENZZT = (tl.unfold(Bijk, 1) @ khatri_rao([Bg, Bw])).reshape(J, R, R)
            FslashY = khatri_rao([G_mu, W_mu]).T @ tl.unfold(Aijk, 1).T
            for j in range(J):
                H_sigma[j,:, :] = torch.linalg.inv(2 * ENZZT[j,:, :] + lambdas.diag())  # Posterior covariance matrix
                H_mu[j, :] = H_sigma[j,:, :] @ (FslashY[:, j] + lambdas.diag() @ Fh[j,:])  # Posterior expectation
            Bh = H_sigma.reshape(J,R * R) + khatri_rao([H_mu.T, H_mu.T]).T

            #update W
            ENZZT = (tl.unfold(Bijk, 2) @ khatri_rao([Bg, Bh])).reshape(K, R, R)
            FslashY = khatri_rao([G_mu, H_mu]).T @ tl.unfold(Aijk, 2).T
            for k in range(K):
                W_sigma[k,:, :] = torch.linalg.inv(2 * ENZZT[k,:, :] + lambdas.diag())  # Posterior covariance matrix
                W_mu[k, :] = W_sigma[k,:, :] @ (FslashY[:, k] + lambdas.diag() @ Fw[k,:])  # Posterior expectation
            Bw = W_sigma.reshape(K,R * R) + khatri_rao([W_mu.T, W_mu.T]).T

            ##
            P1 = tl.cp_to_tensor((weight, [G_mu, H_mu, W_mu]))
            error_itr = torch.norm(P0-P1)/torch.norm(P0)
            if error_itr<tol:
                break
            P0 = P1


            # Update ξ(kx)
            kx2 = (khatri_rao([Bg, Bh, Bw]) @ torch.ones(R * R).unsqueeze(1)).reshape(I, J, K)

            if torch.any(kx2 < 0):
                print(it)
                raise ValueError('error')

            kx = torch.sqrt(kx2)

            lam_kx = self.jisuan_lamb(kx)  # Assuming jisuan_lamb is defined
            Bijk = (c * Y + 1 - Y) * lam_kx


            ###############M-step
            #update lambda
            Cg = (G_mu-Fg).t() @ (G_mu-Fg) + G_sigma.sum(dim=0)
            Ch = (H_mu-Fh).t() @ (H_mu-Fh) + H_sigma.sum(dim=0)
            Cw = (W_mu-Fw).t() @ (W_mu-Fw) + W_sigma.sum(dim=0)
            lambdas = (I+J+K)/(Cg+Ch+Cw).diag()

            # update {Fg,Fh,Fw}
            if it % 20 ==0:
                GHW = torch.cat((G_mu, H_mu, W_mu), dim=0).to(args.device)
                for epoch in range(args.epochs):
                    self.model.train()
                    loss_train = 0
                    optimizer.zero_grad()
                    FGHW = self.model(args)
                    loss = Loss_fun_opt(FGHW, GHW,lambdas,args)
                    loss.backward()
                    optimizer.step()
                    loss_train += loss.item()
                FGHW = FGHW.detach().to('cpu')
                Fg, Fh, Fw = FGHW[:I], FGHW[I:I + J], FGHW[I + J:]
                # print(FGHW.norm())

        # Prepare the results
        P0 = tl.cp_to_tensor((weight, [G_mu, H_mu, W_mu]))
        P = P0.sigmoid()

        return P


    #
    def jisuan_lamb(self,kx):
        sig_kx = kx.sigmoid()
        lam_kx = 1.0 / (2 * kx) * (sig_kx - 0.5)
        return lam_kx



