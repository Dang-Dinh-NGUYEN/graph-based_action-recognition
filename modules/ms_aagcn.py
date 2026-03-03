import math
from typing import Mapping
import torch.nn as nn
import torch_geometric_temporal as tgp
from data_generator.ntu_data import EDGE_INDEX
from modules import register_model

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0.0)

@register_model("ms_aagcn")
class ms_aagcn(nn.Module):
    def __init__(self, num_joint=25, num_skeleton=2, edge_index=EDGE_INDEX, in_channels=3, dropout=0, adaptive=True, attention=True, num_class=128, **kwargs):
        super(ms_aagcn, self).__init__()

        self.graph = tgp.tsagcn.GraphAAGCN(edge_index=edge_index, num_nodes=num_joint)

        self.data_bn = nn.BatchNorm1d(num_skeleton * in_channels * num_joint)

        self.l1 = tgp.tsagcn.AAGCN(in_channels=in_channels, out_channels=16, edge_index=edge_index, num_nodes=num_joint, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = tgp.tsagcn.AAGCN(in_channels=16, out_channels=16, edge_index=edge_index, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l3 = tgp.tsagcn.AAGCN(in_channels=16, out_channels=16, edge_index=edge_index, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l4 = tgp.tsagcn.AAGCN(in_channels=16, out_channels=16, edge_index=edge_index, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l5 = tgp.tsagcn.AAGCN(in_channels=16, out_channels=32, edge_index=edge_index, num_nodes=num_joint, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = tgp.tsagcn.AAGCN(in_channels=32, out_channels=32, edge_index=edge_index, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l7 = tgp.tsagcn.AAGCN(in_channels=32, out_channels=32, edge_index=edge_index, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l8 = tgp.tsagcn.AAGCN(in_channels=32, out_channels=64, edge_index=edge_index, num_nodes=num_joint, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = tgp.tsagcn.AAGCN(in_channels=64, out_channels=64, edge_index=edge_index, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l10 = tgp.tsagcn.AAGCN(in_channels=64, out_channels=256, edge_index=edge_index, num_nodes=num_joint, adaptive=adaptive, attention=attention)

        bn_init(self.data_bn, 1.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))


    def forward(self, x, **kwargs):
        if  isinstance(x, Mapping):
           x = x["raw_inputs"]
           
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        representation = x
        x = self.dropout(x)
        x = self.fc(x)

        return {"features": x, "representation": representation}
