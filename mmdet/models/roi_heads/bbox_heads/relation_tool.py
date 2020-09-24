import torch
from torch import nn
import numpy as np


def PositionalEmbedding(f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    size = delta_h.size()
    # print(size)
    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding


class RelationModule(nn.Module):
    def __init__(self, n_relations=16, appearance_feature_dim=1024, key_feature_dim=64, geo_feature_dim=64,
                 isDuplication=False):
        super(RelationModule, self).__init__()
        self.isDuplication = isDuplication
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))

    def forward(self, input_data):
        if (self.isDuplication):
            f_a, embedding_f_a, position_embedding = input_data
        else:
            f_a, position_embedding = input_data
        isFirst = True
        for N in range(self.Nr):
            if (isFirst):
                if (self.isDuplication):
                    concat = self.relation[N](embedding_f_a, position_embedding)
                else:
                    concat = self.relation[N](f_a, position_embedding)
                isFirst = False
            else:
                if (self.isDuplication):
                    concat = torch.cat((concat, self.relation[N](embedding_f_a, position_embedding)), -1)
                else:
                    concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return concat + f_a


class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024, key_feature_dim=64, geo_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_a, position_embedding):
        N, _ = f_a.size()

        position_embedding = position_embedding.view(-1, self.dim_g)

        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a)
        w_k = w_k.view(N, 1, self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1, N, self.dim_k)

        scaled_dot = torch.sum((w_k * w_q), -1)
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N, N)
        w_a = scaled_dot.view(N, N)

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N, N, 1)
        w_v = w_v.view(N, 1, -1)

        output = w_mn * w_v

        output = torch.sum(output, -2)
        return output
