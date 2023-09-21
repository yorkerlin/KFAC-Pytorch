import math

import torch
import torch.optim as optim

from .org_kfac_utils import (ComputeCovA, ComputeCovG)
from .org_kfac_utils import update_running_stat


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 cast_dtype = torch.float32,
                 use_eign = True,
                 using_adamw = False,
                 adamw_eps = 1e-8,
                 adamw_beta1 = 0.9,
                 adamw_beta2 = 0.999,
                 ):
        print('org kfac v2')
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model

        self.steps = 0
        self.cast_dtype = cast_dtype
        self.use_eign = use_eign
        self.grad_scale = 1.0
        self.using_adamw = using_adamw
        self.org_lr= lr
        self.org_wt = weight_decay
        self.adamw_eps = adamw_eps
        self.adamw_beta1 = adamw_beta1
        self.adamw_beta2 = adamw_beta2

        self._prepare_model()


        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            gg /= (self.grad_scale**2)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    # def _prepare_model(self):
        # count = 0
        # # print(self.model)
        # print("=> We keep following layers in KFAC. ")
        # for module in self.model.modules():
            # classname = module.__class__.__name__
            # # print('=> We keep following layers in KFAC. <=')
            # if classname in self.known_modules:
                # self.modules.append(module)
                # module.register_forward_pre_hook(self._save_input)
                # module.register_backward_hook(self._save_grad_output)
                # # print('(%s): %s' % (count, module))
                # count += 1


    def _prepare_model(self): #ok
        count = 0
        self.param_keys={}
        self.get_name = {}
        # print(self.model)
        print("=> We keep following layers in. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers. <=')
            if classname in self.known_modules:

                for name, param in module.named_parameters():
                    if param.requires_grad:
                        self.param_keys.setdefault(param.data_ptr(), module)

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                # print('(%s): %s' % (count, module))
                count += 1

        has_decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                match=False
                if param.data_ptr() in self.param_keys:
                    full_name = '%s-%s'%(name, self.param_keys[param.data_ptr()].__class__.__name__) 
                    print('using kfac', name, full_name, param.size())
                    self.get_name.setdefault(self.param_keys[param.data_ptr()], full_name)
                else:
                    if self.using_adamw:
                        print('using adamw', name, param.size())
                    else:
                        print('using sgd', name, param.size())

                    has_decay.append(param)

        if self.using_adamw:
            param_others = [{'params': has_decay},]
            self.opt_others = optim.AdamW(param_others, eps=self.adamw_eps,
                    betas=(self.adamw_beta1, self.adamw_beta2),
                    lr=self.org_lr, weight_decay=self.org_wt)

            print('adamw eps:', self.adamw_eps)
            print('adamw beta1:', self.adamw_beta1)
            print('adamw beta2:', self.adamw_beta2)





    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.linalg.eigh(
            self.m_aa[m].to(dtype=torch.float32), UPLO='U')
        self.d_g[m], self.Q_g[m] = torch.linalg.eigh(
            self.m_gg[m].to(dtype=torch.float32), UPLO='U')

        self.d_a[m].mul_((self.d_a[m] > eps).float()).to(dtype=self.cast_dtype)
        self.d_g[m].mul_((self.d_g[m] > eps).float()).to(dtype=self.cast_dtype)
        self.Q_a[m] = self.Q_a[m].to(dtype=self.cast_dtype)
        self.Q_g[m] = self.Q_g[m].to(dtype=self.cast_dtype)


    def _inv_covs(self, m, damping):
        """Inverses the covariances."""
        eps = damping
        diag_aat = self.m_aa[m].new(self.m_aa[m].shape[0]).fill_(eps)
        diag_ggt = self.m_gg[m].new(self.m_gg[m].shape[0]).fill_(eps)

        self.Q_a[m] = (self.m_aa[m] +
                torch.diag(diag_aat)).to(dtype=torch.float32).inverse().to(dtype=self.cast_dtype)
        self.Q_g[m] = (self.m_gg[m] +
                torch.diag(diag_ggt)).to(dtype=torch.float32).inverse().to(dtype=self.cast_dtype)



    @staticmethod
    def _get_matrix_form_grad(m, classname, cast_dtype):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat.to(dtype=cast_dtype)

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        if self.use_eign:
            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        else:
            v = self.Q_g[m] @ p_grad_mat @ self.Q_a[m]

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        # nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))
        nu = 1.0

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                if self.using_adamw and  (p.data_ptr() not in self.param_keys):
                    #using opt_others
                    continue

                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        if self.using_adamw:
            self.opt_others.step()

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        # print('lr_cov:', 1.0 - self.stat_decay)
        # print('grad_scale:',  self.grad_scale)
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                if self.use_eign:
                    self._update_inv(m)
                else:
                    self._inv_covs(m, damping)
            p_grad_mat = self._get_matrix_form_grad(m, classname, self.cast_dtype)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
        self.grad_scale = 1.0
