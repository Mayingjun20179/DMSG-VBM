import numpy as np


def cv_tensor_model_evaluate(GHW_data, predict_tensor, test_index, seed,prop):
#Referred from F. Huang, X. Yue, Z. Xiong et al., “Tensor decomposition with relational constraints for predicting multiple types of microRNA-disease associations,” Brief Bioinform, vol. 22, no. 3, May 20, 2021.
    test_po_num = np.array(test_index).shape[1]   
    test_index_0 = GHW_data.index_0.T   
    np.random.seed(seed)
    random_ind = np.random.randint(0, GHW_data.N_0, size=prop*test_po_num) 
    test_ne_index = tuple(test_index_0[:, random_ind])
    real_score = np.column_stack(
        (np.mat(GHW_data.adj_tensor[test_ne_index].flatten()), np.mat(GHW_data.adj_tensor[test_index].flatten())))
    predict_score = np.column_stack(
        (np.mat(predict_tensor[test_ne_index].flatten()), np.mat(predict_tensor[test_index].flatten())))
    # real_score and predict_score are array
    return get_metrics(real_score, predict_score)


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix * real_score.T
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T

    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
    PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    # return [aupr[0, 0], auc[0, 0], f1_score]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]




def cal_recall_ndcg(real_score, predict_score,topK):
#Referred from C. Chen, M. Zhang, Y. Zhang et al., “Efficient Neural Matrix Factorization without Sampling for Recommendation,” ACM Transactions on Information Systems, vol. 38, no. 2, pp. 1-28, 2020.
    real_score = np.array(real_score).squeeze()
    predict_score = np.array(predict_score).squeeze()

    recall = []
    true_bin = np.zeros_like(predict_score, dtype=bool)
    true_bin[real_score.nonzero()] = True  

    for kj in topK:
        idx_topk_part = np.argpartition(-predict_score, kj)  

        pre_bin = np.zeros_like(predict_score, dtype=bool)
        pre_bin[idx_topk_part[:kj]] = True
        # pre_bin[:,idx_topk_part[:,:kj]] = True

        tmp = (np.logical_and(true_bin, pre_bin).sum()).astype(np.float32)
        recall.append(tmp / np.minimum(kj, true_bin.sum()))  

    ndcg = []

    for kj in topK:
        idx_topk_part = np.argpartition(-predict_score, kj)

        topk_part = predict_score[idx_topk_part[:kj]]
        idx_part = np.argsort(-topk_part)
        idx_topk = idx_topk_part[idx_part] 

        tp = np.log(2) / np.log(np.arange(2, kj + 2))

        DCG = (real_score[idx_topk] * tp).sum()

        IDCG = (tp[:min(true_bin.sum(), kj)]).sum()  
        ndcg.append(DCG / IDCG)


    result = np.concatenate((np.array(recall),np.array(ndcg)))
    return result
