import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


# def _extract_patches(x, kernel_size, stride, padding):
    # """
    # :param x: The input feature maps.  (batch_size, in_c, h, w)
    # :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    # :param stride: the stride of conv operation  (tuple of two elements)
    # :param padding: number of paddings. be a tuple of two elements
    # :return: (batch_size, out_h, out_w, in_c*kh*kw)
    # """
    # if padding[0] + padding[1] > 0:
        # x = F.pad(x, (padding[1], padding[1], padding[0],
                      # padding[0])).data  # Actually check dims
    # x = x.unfold(2, kernel_size[0], stride[0])
    # x = x.unfold(3, kernel_size[1], stride[1])
    # x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    # x = x.view(
        # x.size(0), x.size(1), x.size(2),
        # x.size(3) * x.size(4) * x.size(5))
    # return x

def _extract_patches(x, kernel_size, stride, padding, groups):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3) #.contiguous()
    return torch.mean(x.reshape((x.size(0), x.size(1), x.size(2), groups, -1, x.size(4),x.size(5))), 3).view(
            x.size(0), x.size(1), x.size(2), -1)



def update_running_stat(aa, m_aa, stat_decay):
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)



class ComputeCovA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a = None

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding, layer.groups)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    # @staticmethod
    # def linear(a, layer):
        # # a: batch_size * in_dim
        # batch_size = a.size(0)
        # if layer.bias is not None:
            # a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        # return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer, mode='expand'):
        wt = 1.0
        batch_size = a.size(0)
        if a.ndim == 2:
            b = a
        elif a.ndim == 3:
            if mode=='reduce':
                b = a.mean(dim=1)#reduce case
            else:
                wt = 1.0/np.sqrt(a.size(1)) #expand case
                b = a.reshape(-1, a.size(-1))/np.sqrt(a.size(1)) #expand case
        else:
            if mode=='reduce':
                raise NotImplementedError
            else:
                wt = 1.0/np.sqrt(np.prod(a.shape[1:-1])) #expand case
                b = a.reshape(-1, a.size(-1))/np.sqrt(np.prod(a.shape[1:-1])) #expand case

        if layer.bias is not None:
            b = torch.cat([b, b.new(b.size(0), 1).fill_(wt)], 1)

        b = b / np.sqrt(batch_size)
        return b.t() @ b




class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    # @staticmethod
    # def linear(g, layer, batch_averaged):
        # # g: batch_size * out_dim
        # batch_size = g.size(0)

        # if batch_averaged:
            # cov_g = g.t() @ (g * batch_size)
        # else:
            # cov_g = g.t() @ (g / batch_size)
        # return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged, mode='expand', scaling=1.0):
        batch_size = g.size(0)

        if g.ndim == 2:
            b = g
        elif g.ndim == 3:
            if mode=='reduce':
                b = g.sum(dim=1) #reduce case
                # b = g.sum(dim=1)/np.sqrt(g.size(1)) #modified reduce case
            else:
                b = g.reshape(-1, g.size(-1)) #expand
        else:
            if mode=='reduce':
                raise NotImplementedError
            else:
                b = g.reshape(-1, g.size(-1)) #expand

        if batch_averaged:
            b = b * (scaling * np.sqrt(batch_size))
        else:
            b = b * (scaling / np.sqrt(batch_size))
        return b.t() @ b



if __name__ == '__main__':
    def test_ComputeCovA():
        pass

    def test_ComputeCovG():
        pass






