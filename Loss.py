
import torch
import torch.nn.functional as F

def Contrastive_Loss(Z1, Z2, tau, num_samples=None):

    N = Z1.size(0)

    if num_samples is None or num_samples > N:
        num_samples = N

    # Cosine Similarity Calculation
    Z1 = F.normalize(Z1, dim=1)
    Z2 = F.normalize(Z2, dim=1)

    indices = torch.randperm(N)[:num_samples]
    Z1_sample = Z1[indices]
    Z2_sample = Z2[indices]

    sim_matrix = torch.mm(Z1_sample, Z2_sample.t()) / tau
    pos_sim = torch.diag(sim_matrix)

    neg_sim_matrix = torch.exp(sim_matrix)
    neg_exp_sim = (neg_sim_matrix.sum(dim=1) - torch.exp(pos_sim))

    pos_exp_sim = torch.exp(pos_sim)
    loss = -torch.log(pos_exp_sim / (pos_exp_sim + neg_exp_sim)).mean()

    return loss

