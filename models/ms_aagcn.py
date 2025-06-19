import math
import torch.nn as nn
import torch_geometric_temporal as tgp
from data_generator.ntu_data import EDGE_INDEX


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class ms_aagcn(nn.Module):
    def __init__(self, num_class=60, num_joint=25, num_skeleton=2, edge_index=EDGE_INDEX, in_channels=3, drop_out=0, adaptive=True, attention=True):
        super(ms_aagcn, self).__init__()

        self.graph = tgp.tsagcn.GraphAAGCN(edge_index=EDGE_INDEX, num_nodes=num_joint)

        self.data_bn = nn.BatchNorm1d(num_skeleton * in_channels * num_joint)

        self.l1 = tgp.tsagcn.AAGCN(in_channels=3, out_channels=64, edge_index=EDGE_INDEX, num_nodes=num_joint, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = tgp.tsagcn.AAGCN(in_channels=64, out_channels=64, edge_index=EDGE_INDEX, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l3 = tgp.tsagcn.AAGCN(in_channels=64, out_channels=64, edge_index=EDGE_INDEX, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l4 = tgp.tsagcn.AAGCN(in_channels=64, out_channels=64, edge_index=EDGE_INDEX, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l5 = tgp.tsagcn.AAGCN(in_channels=64, out_channels=128, edge_index=EDGE_INDEX, num_nodes=num_joint, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = tgp.tsagcn.AAGCN(in_channels=128, out_channels=128, edge_index=EDGE_INDEX, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l7 = tgp.tsagcn.AAGCN(in_channels=128, out_channels=128, edge_index=EDGE_INDEX, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l8 = tgp.tsagcn.AAGCN(in_channels=128, out_channels=256, edge_index=EDGE_INDEX, num_nodes=num_joint, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = tgp.tsagcn.AAGCN(in_channels=256, out_channels=256, edge_index=EDGE_INDEX, num_nodes=num_joint, adaptive=adaptive, attention=attention)
        self.l10 = tgp.tsagcn.AAGCN(in_channels=256, out_channels=256, edge_index=EDGE_INDEX, num_nodes=num_joint, adaptive=adaptive, attention=attention)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
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
        x = self.drop_out(x)

        return self.fc(x)

