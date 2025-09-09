# 2023.03.16-Changed for building NexToU
#            Harbin Institute of Technology (Shenzhen), <pcshi@stu.hit.edu.cn>

# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin

##############################
#    Basic layers
##############################
def act_layer(act, inplace=True, neg_slope=1e-2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc, conv_op):
    # normalization layer
    norm = norm.lower()
    if norm == 'batch':
        if conv_op == nn.Conv2d:
            layer = nn.BatchNorm2d(nc, affine=True)
        elif conv_op == nn.Conv3d:
            layer = nn.BatchNorm3d(nc, affine=True)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)
    elif norm == 'instance':
        if conv_op == nn.Conv2d:
            layer = nn.InstanceNorm2d(nc, affine=True)
        elif conv_op == nn.Conv3d:
            layer = nn.InstanceNorm3d(nc, affine=True)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, conv_op=nn.Conv3d):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1], conv_op))
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0., conv_op=nn.Conv3d, dropout_op=None):
        m = []
        self.conv_op = conv_op
        if self.conv_op == nn.Conv2d:
            self.batch_norm = nn.BatchNorm2d
            self.instance_norm = nn.InstanceNorm2d
            self.groups_num = 4
        elif self.conv_op == nn.Conv3d:
            self.batch_norm = nn.BatchNorm3d
            self.instance_norm = nn.InstanceNorm3d
            self.groups_num = 8
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        dropout_op = dropout_op
        dropout_op_kwargs = {}
        dropout_op_kwargs['p'] = drop
        for i in range(1, len(channels)):
            m.append(conv_op(channels[i - 1], channels[i], 1, bias=bias, groups=self.groups_num))
            
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1], conv_op))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            
        super(BasicConv, self).__init__(*m)

def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times k}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature
