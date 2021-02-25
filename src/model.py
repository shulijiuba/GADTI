import dgl
import torch.nn.functional as F
import torch as th
import torch.nn as nn
from src.utils import *
from src.propagation import Propagation


class GRDTI(nn.Module):
    def __init__(self, g, n_disease, n_drug, n_protein, n_sideeffect, args):
        super(GRDTI, self).__init__()
        self.g = g
        self.device = th.device(args.device)
        self.dim_embedding = args.dim_embedding

        self.activation = F.elu
        self.reg_lambda = args.reg_lambda

        self.num_disease = n_disease
        self.num_drug = n_drug
        self.num_protein = n_protein
        self.num_sideeffect = n_sideeffect

        self.drug_feat = nn.Parameter(th.FloatTensor(self.num_drug, self.dim_embedding))
        nn.init.normal_(self.drug_feat, mean=0, std=0.1)
        self.protein_feat = nn.Parameter(th.FloatTensor(self.num_protein, self.dim_embedding))
        nn.init.normal_(self.protein_feat, mean=0, std=0.1)
        self.disease_feat = nn.Parameter(th.FloatTensor(self.num_disease, self.dim_embedding))
        nn.init.normal_(self.disease_feat, mean=0, std=0.1)
        self.sideeffect_feat = nn.Parameter(th.FloatTensor(self.num_sideeffect, self.dim_embedding))
        nn.init.normal_(self.sideeffect_feat, mean=0, std=0.1)

        # 邻居信息的权重矩阵，对应论文公式（1）中的Wr、br
        self.fc_DDI = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_ch = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_Di = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_Side = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_PPI = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_seq = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_Di = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Di_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Di_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Side_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()

        self.propagation = Propagation(args.k, args.alpha, args.edge_drop)

        # Linear transformation for reconstruction
        tmp = th.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_ch = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Side = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_P = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_PPI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_seq = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        #self.re_P_D = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        #self.re_Di_P = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        #self.re_Di_D = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        #self.re_Side_D = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))

        self.reset_parameters()

    def reset_parameters(self):
        for m in GRDTI.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                protein_sequence, protein_disease, drug_protein, drug_protein_mask):

        disease_feat = th.mean(th.stack((th.mm(row_normalize(drug_disease.T).float(),
                                               F.relu(self.fc_Di_D(self.drug_feat))),
                                         th.mm(row_normalize(protein_disease.T).float(),
                                               F.relu(self.fc_Di_P(self.protein_feat))),
                                         self.disease_feat), dim=1), dim=1)

        drug_feat = th.mean(th.stack((th.mm(row_normalize(drug_drug).float(),
                                            F.relu(self.fc_DDI(self.drug_feat))),
                                      th.mm(row_normalize(drug_chemical).float(),
                                            F.relu(self.fc_D_ch(self.drug_feat))),
                                      th.mm(row_normalize(drug_disease).float(),
                                            F.relu(self.fc_D_Di(self.disease_feat))),
                                      th.mm(row_normalize(drug_sideeffect).float(),
                                            F.relu(self.fc_D_Side(self.sideeffect_feat))),
                                      th.mm(row_normalize(drug_protein).float(),
                                            F.relu(self.fc_D_P(self.protein_feat))),
                                      self.drug_feat), dim=1), dim=1)

        protein_feat = th.mean(th.stack((th.mm(row_normalize(protein_protein).float(),
                                               F.relu(self.fc_PPI(self.protein_feat))),
                                         th.mm(row_normalize(protein_sequence).float(),
                                               F.relu(self.fc_P_seq(self.protein_feat))),
                                         th.mm(row_normalize(protein_disease).float(),
                                               F.relu(self.fc_P_Di(self.disease_feat))),
                                         th.mm(row_normalize(drug_protein.T).float(),
                                               F.relu(self.fc_P_D(self.drug_feat))),
                                         self.protein_feat), dim=1), dim=1)

        sideeffect_feat = th.mean(th.stack((th.mm(row_normalize(drug_sideeffect.T).float(),
                                                  F.relu(self.fc_Side_D(self.drug_feat))),
                                            self.sideeffect_feat), dim=1), dim=1)

        node_feat = th.cat((disease_feat, drug_feat, protein_feat, sideeffect_feat), dim=0)

        node_feat = self.propagation(dgl.to_homogeneous(self.g), node_feat)

        disease_embedding = node_feat[:self.num_disease].to(self.device)
        drug_embedding = node_feat[self.num_disease:self.num_disease + self.num_drug].to(self.device)
        protein_embedding = node_feat[self.num_disease + self.num_drug:self.num_disease + self.num_drug +
                                                                       self.num_protein].to(self.device)
        sideeffect_embedding = node_feat[-self.num_sideeffect:].to(self.device)

        disease_vector = l2_norm(disease_embedding)
        drug_vector = l2_norm(drug_embedding)
        protein_vector = l2_norm(protein_embedding)
        sideeffect_vector = l2_norm(sideeffect_embedding)

        drug_drug_reconstruct = th.mm(th.mm(drug_vector, self.re_DDI), drug_vector.t())
        drug_drug_reconstruct_loss = th.sum(
            (drug_drug_reconstruct - drug_drug.float()) ** 2)

        drug_chemical_reconstruct = th.mm(th.mm(drug_vector, self.re_D_ch), drug_vector.t())
        drug_chemical_reconstruct_loss = th.sum(
            (drug_chemical_reconstruct - drug_chemical.float()) ** 2)

        drug_disease_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Di), disease_vector.t())
        drug_disease_reconstruct_loss = th.sum(
            (drug_disease_reconstruct - drug_disease.float()) ** 2)

        drug_sideeffect_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Side), sideeffect_vector.t())
        drug_sideeffect_reconstruct_loss = th.sum(
            (drug_sideeffect_reconstruct - drug_sideeffect.float()) ** 2)

        protein_protein_reconstruct = th.mm(th.mm(protein_vector, self.re_PPI), protein_vector.t())
        protein_protein_reconstruct_loss = th.sum(
            (protein_protein_reconstruct - protein_protein.float()) ** 2)

        protein_sequence_reconstruct = th.mm(th.mm(protein_vector, self.re_P_seq), protein_vector.t())
        protein_sequence_reconstruct_loss = th.sum(
            (protein_sequence_reconstruct - protein_sequence.float()) ** 2)

        protein_disease_reconstruct = th.mm(th.mm(protein_vector, self.re_P_Di), disease_vector.t())
        protein_disease_reconstruct_loss = th.sum(
            (protein_disease_reconstruct - protein_disease.float()) ** 2)

        drug_protein_reconstruct = th.mm(th.mm(drug_vector, self.re_D_P), protein_vector.t())
        tmp = th.mul(drug_protein_mask.float(), (drug_protein_reconstruct - drug_protein.float()))
        DTI_potential = drug_protein_reconstruct - drug_protein.float()
        drug_protein_reconstruct_loss = th.sum(tmp ** 2)

        other_loss = drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss + drug_disease_reconstruct_loss + \
                     drug_sideeffect_reconstruct_loss + protein_protein_reconstruct_loss + \
                     protein_sequence_reconstruct_loss + protein_disease_reconstruct_loss

        L2_loss = 0.
        for name, param in GRDTI.named_parameters(self):
            if 'bias' not in name:
                L2_loss = L2_loss + th.sum(param.pow(2))
        L2_loss = L2_loss * 0.5

        tloss = drug_protein_reconstruct_loss + 1.0 * other_loss + self.reg_lambda * L2_loss

        return tloss, drug_protein_reconstruct_loss, L2_loss, drug_protein_reconstruct, DTI_potential