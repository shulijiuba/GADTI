# -*- coding: utf-8 -*-

import dgl
import time
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.model import GRDTI


def loda_data():
    network_path = '../data/'

    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    true_drug = 708
    drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
    drug_chemical = drug_chemical[:true_drug, :true_drug]
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')

    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')

    num_drug = len(drug_drug)
    num_protein = len(protein_protein)

    # Removed the self-loop
    drug_chemical = drug_chemical - np.identity(num_drug)
    protein_sequence = protein_sequence / 100.
    protein_sequence = protein_sequence - np.identity(num_protein)

    drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')

    # Removed DTIs with similar drugs or proteins
    #drug_protein = np.loadtxt(network_path + 'mat_drug_protein_homo_protein_drug.txt')

    print("Load data finished.")

    return drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, \
           protein_disease, drug_protein


def ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                   protein_disease, drug_protein):
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = len(drug_disease.T)
    num_sideeffect = len(drug_sideeffect.T)

    list_drug = []
    for i in range(num_drug):
        list_drug.append((i, i))

    list_protein = []
    for i in range(num_protein):
        list_protein.append((i, i))

    list_disease = []
    for i in range(num_disease):
        list_disease.append((i, i))

    list_sideeffect = []
    for i in range(num_sideeffect):
        list_sideeffect.append((i, i))

    list_DDI = []
    for row in range(num_drug):
        for col in range(num_drug):
            if drug_drug[row, col] > 0:
                list_DDI.append((row, col))

    list_PPI = []
    for row in range(num_protein):
        for col in range(num_protein):
            if protein_protein[row, col] > 0:
                list_PPI.append((row, col))

    list_drug_protein = []
    list_protein_drug = []
    for row in range(num_drug):
        for col in range(num_protein):
            if drug_protein[row, col] > 0:
                list_drug_protein.append((row, col))
                list_protein_drug.append((col, row))

    list_drug_sideeffect = []
    list_sideeffect_drug = []
    for row in range(num_drug):
        for col in range(num_sideeffect):
            if drug_sideeffect[row, col] > 0:
                list_drug_sideeffect.append((row, col))
                list_sideeffect_drug.append((col, row))

    list_drug_disease = []
    list_disease_drug = []
    for row in range(num_drug):
        for col in range(num_disease):
            if drug_disease[row, col] > 0:
                list_drug_disease.append((row, col))
                list_disease_drug.append((col, row))

    list_protein_disease = []
    list_disease_protein = []
    for row in range(num_protein):
        for col in range(num_disease):
            if protein_disease[row, col] > 0:
                list_protein_disease.append((row, col))
                list_disease_protein.append((col, row))

    g_HIN = dgl.heterograph({('disease', 'disease_disease virtual', 'disease'): list_disease,
                             ('drug', 'drug_drug virtual', 'drug'): list_drug,
                             ('protein', 'protein_protein virtual', 'protein'): list_protein,
                             ('sideeffect', 'sideeffect_sideeffect virtual', 'sideeffect'): list_sideeffect,
                             ('drug', 'drug_drug interaction', 'drug'): list_DDI, \
                             ('protein', 'protein_protein interaction', 'protein'): list_PPI, \
                             ('drug', 'drug_protein interaction', 'protein'): list_drug_protein, \
                             ('protein', 'protein_drug interaction', 'drug'): list_protein_drug, \
                             ('drug', 'drug_sideeffect association', 'sideeffect'): list_drug_sideeffect, \
                             ('sideeffect', 'sideeffect_drug association', 'drug'): list_sideeffect_drug, \
                             ('drug', 'drug_disease association', 'disease'): list_drug_disease, \
                             ('disease', 'disease_drug association', 'drug'): list_disease_drug, \
                             ('protein', 'protein_disease association', 'disease'): list_protein_disease, \
                             ('disease', 'disease_protein association', 'protein'): list_disease_protein})

    g = g_HIN.edge_type_subgraph(['drug_drug interaction', 'protein_protein interaction',
                                  'drug_protein interaction', 'protein_drug interaction',
                                  'drug_sideeffect association', 'sideeffect_drug association',
                                  'drug_disease association', 'disease_drug association',
                                  'protein_disease association', 'disease_protein association'
                                  ])

    return g


def TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_drug, drug_chemical, drug_disease,
                     drug_sideeffect, protein_protein, protein_sequence, protein_disease):
    device = th.device(args.device)

    # Numbers of different nodes
    num_disease = len(drug_disease.T)
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_sideeffect = len(drug_sideeffect.T)

    drug_protein = th.zeros((num_drug, num_protein))
    mask = th.zeros((num_drug, num_protein)).to(device)
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    best_valid_aupr = 0.
    # best_valid_auc = 0
    test_aupr = 0.
    test_auc = 0.
    patience = 0.

    pos = np.count_nonzero(DTItest[:, 2])
    neg = np.size(DTItest[:, 2]) - pos
    xy_roc_sampling = []
    xy_pr_sampling = []

    g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                       protein_disease, drug_protein)

    drug_drug = th.tensor(drug_drug).to(device)
    drug_chemical = th.tensor(drug_chemical).to(device)
    drug_disease = th.tensor(drug_disease).to(device)
    drug_sideeffect = th.tensor(drug_sideeffect).to(device)
    protein_protein = th.tensor(protein_protein).to(device)
    protein_sequence = th.tensor(protein_sequence).to(device)
    protein_disease = th.tensor(protein_disease).to(device)
    drug_protein = drug_protein.to(device)

    model = GRDTI(g, num_disease, num_drug, num_protein, num_sideeffect, args)
    model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.epochs):

        model.train()
        tloss, dtiloss, l2loss, dp_re, DTI_p = model(drug_drug, drug_chemical, drug_disease, drug_sideeffect,
                                                     protein_protein, protein_sequence, protein_disease,
                                                     drug_protein, mask)

        results = dp_re.detach().cpu()
        optimizer.zero_grad()
        loss = tloss
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()

        if i % 25 == 0:
            with th.no_grad():
                print("step", i, ":", "Total_loss & DTIloss & L2_loss:", loss.cpu().data.numpy(), ",", dtiloss.item(),
                      ",", l2loss.item())

                pred_list = []
                ground_truth = []

                for ele in DTIvalid:
                    pred_list.append(results[ele[0], ele[1]])
                    ground_truth.append(ele[2])

                valid_auc = roc_auc_score(ground_truth, pred_list)
                valid_aupr = average_precision_score(ground_truth, pred_list)

                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    # best_valid_auc = valid_auc
                    best_DTI_potential = DTI_p
                    patience = 0

                    # Calculating AUC & AUPR (pos:neg=1:10)
                    db = []
                    xy_roc = []
                    xy_pr = []
                    for ele in DTItest:
                        db.append([results[ele[0], ele[1]], ele[2]])

                    db = sorted(db, key=lambda x: x[0], reverse=True)

                    tp, fp = 0., 0.
                    for i_db in range(len(db)):
                        if db[i_db][0]:
                            if db[i_db][1]:
                                tp = tp + 1
                            else:
                                fp = fp + 1
                            xy_roc.append([fp / neg, tp / pos])
                            xy_pr.append([tp / pos, tp / (tp + fp)])

                    test_auc = 0.
                    prev_x = 0.
                    for x, y in xy_roc:
                        if x != prev_x:
                            test_auc += (x - prev_x) * y
                            prev_x = x

                    test_aupr = 0.
                    prev_x = 0.
                    for x, y in xy_pr:
                        if x != prev_x:
                            test_aupr += (x - prev_x) * y
                            prev_x = x

                    # All unknown DTI pairs all treated as negative examples
                    '''pred_list = []
                    ground_truth = []
                    for ele in DTItest:
                        pred_list.append(results[ele[0], ele[1]])
                        ground_truth.append(ele[2])
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)'''

                else:
                    patience += 1
                    if patience > args.patience:
                        print("Early Stopping")

                        # sampling (pos:neg=1:10) for averaging and plotting
                        xy_roc_sampling = []
                        xy_pr_sampling = []
                        for i_xy in range(len(xy_roc)):
                            if i_xy % 10 == 0:
                                xy_roc_sampling.append(xy_roc[i_xy])
                                xy_pr_sampling.append(xy_pr[i_xy])

                        # Record data for sampling, averaging and plotting.
                        # All unknown DTI pairs all treated as negative examples
                        '''t1 = time.localtime()
                        time_creat_txt = str(t1.tm_year) + '_' + str(t1.tm_mon) + '_' + str(t1.tm_mday) + '_' + str(
                            t1.tm_hour) + '_' + str(t1.tm_min)
                        fpr, tpr, threshold = roc_curve(ground_truth, pred_list)
                        print("len(fpr):", len(fpr))
                        np.savetxt('fpr_' + time_creat_txt + '.csv', fpr)
                        np.savetxt('tpr_' + time_creat_txt + '.csv', tpr)
                        np.savetxt('ROC_threshold_' + time_creat_txt + '.csv', threshold)

                        precision, recall, threshold = precision_recall_curve(ground_truth, pred_list)
                        print("len(recall):", len(recall))
                        np.savetxt('precision_' + time_creat_txt + '.csv', precision)
                        np.savetxt('recall_' + time_creat_txt + '.csv', recall)
                        np.savetxt('PRC_threshold_' + time_creat_txt + '.csv', threshold)'''

                        break

                print('Valid auc & aupr:', valid_auc, valid_aupr, ";  ", 'Test auc & aupr:', test_auc, test_aupr)

    return test_auc, test_aupr, xy_roc_sampling, xy_pr_sampling, best_DTI_potential


def main(args):
    drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq, protein_di, dti_original = loda_data()

    # sampling
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_original[i][j]) == 0:
                whole_negative_index.append([i, j])

    # pos:neg=1:10
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=10 * len(whole_positive_index), replace=False)

    # All unknown DTI pairs all treated as negative examples
    '''negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(whole_negative_index), replace=False)'''

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    test_auc_round = []
    test_aupr_round = []
    tpr_mean = []
    fpr = []
    precision_mean = []
    recall = []

    rounds = args.rounds
    for r in range(rounds):
        print("----------------------------------------")

        test_auc_fold = []
        test_aupr_fold = []

        kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        k_fold = 0

        for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
            train = data_set[train_index]
            DTItest = data_set[test_index]
            DTItrain, DTIvalid = train_test_split(train, test_size=0.05, random_state=None)

            k_fold += 1
            print("--------------------------------------------------------------")
            print("round ", r + 1, " of ", rounds, ":", "KFold ", k_fold, " of 10")
            print("--------------------------------------------------------------")

            time_roundStart = time.time()

            t_auc, t_aupr, xy_roc, xy_pr, DTI_potential = TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_d,
                                                                           drug_ch, drug_di, drug_side, protein_p,
                                                                           protein_seq, protein_di)

            time_roundEnd = time.time()
            print("Time spent in this fold:", time_roundEnd - time_roundStart)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)

            order_txt1 = 'DTI_potential_' + 'r' + str(r + 1) + '_f' + str(k_fold) + '.csv'
            np.savetxt(order_txt1, DTI_potential.detach().cpu().numpy(), fmt='%-.4f', delimiter=',')
            top_values, top_indices = th.topk(DTI_potential, 40)
            order_txt2 = 'top40_' + 'r' + str(r + 1) + '_f' + str(k_fold) + '.csv'
            np.savetxt(order_txt2, top_indices.detach().cpu().numpy(), fmt='%d', delimiter=',')

            # pos:neg=1:10
            if not fpr:
                fpr = [_v[0] for _v in xy_roc]
            if not recall:
                recall = [_v[0] for _v in xy_pr]

            temp = [_v[1] for _v in xy_roc]
            tpr_mean.append(temp)
            temp = [_v[1] for _v in xy_pr]
            precision_mean.append(temp)

        print("Training and evaluation is OK.")

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))

    t1 = time.localtime()
    time_creat_txt = str(t1.tm_year) + '_' + str(t1.tm_mon) + '_' + str(t1.tm_mday) + '_' + str(t1.tm_hour) + '_' + str(
        t1.tm_min)
    np.savetxt('test_auc_' + time_creat_txt, test_auc_round)
    np.savetxt('test_aupr_' + time_creat_txt, test_aupr_round)

    # pos:neg=1:10
    tpr = (np.mean(np.array(tpr_mean), axis=0)).tolist()
    precision = (np.mean(np.array(precision_mean), axis=0)).tolist()

    np.savetxt('fpr.csv', fpr, fmt='%-.4f', delimiter=',')
    np.savetxt('tpr.csv', tpr, fmt='%-.4f', delimiter=',')
    np.savetxt('recall.csv', recall, fmt='%-.4f', delimiter=',')
    np.savetxt('precision.csv', precision, fmt='%-.4f', delimiter=',')


if __name__ == "__main__":
    args = parse_args()
    print(args)

    start = time.time()
    main(args)
    end = time.time()
    print("Total time:", end - start)
