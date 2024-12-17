import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List
import math


# optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)
# optimizer = MARS(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, lr_1d=args.adamw_lr)

__all__ = ['SophiaG', 'MARS']


def exists(val):
    return val is not None


def update_fn(p, grad, exp_avg, exp_avg_sq, lr, wd, beta1, beta2, last_grad, eps, amsgrad, max_exp_avg_sq, step, gamma,
              mars_type, is_grad_2d, optimize_1d, lr_1d_factor, betas_1d, weight_decay_1d):
    # optimize_1d: use MARS for 1d para, not: use AdamW for 1d para
    if optimize_1d or is_grad_2d:
        c_t = (grad - last_grad).mul(gamma * (beta1 / (1. - beta1))).add(grad)
        c_t_norm = torch.norm(c_t)
        if c_t_norm > 1.:
            c_t = c_t / c_t_norm
        exp_avg.mul_(beta1).add_(c_t, alpha=1. - beta1)
        if (mars_type == "mars-adamw") or (mars_type == "mars-shampoo" and not is_grad_2d):
            exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1. - beta2)
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            if amsgrad:
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
            else:
                denom = exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
            real_update_tmp = -lr * torch.mul(p.data, wd).add(exp_avg.div(denom))
        elif mars_type == "mars-lion":
            real_update_tmp = -lr * torch.mul(p.data, wd).add(exp_avg.sign())
        elif mars_type == "mars-shampoo" and is_grad_2d:
            factor = max(1, grad.size(0) / grad.size(1)) ** 0.5
            real_update_tmp = NewtonSchulz(exp_avg.mul(1. / (1. - beta1)), eps=eps).mul(factor).add(wd, p.data).mul(-lr)
        p.data.add_(real_update_tmp)
    else:
        beta1_1d, beta2_1d = betas_1d
        exp_avg.mul_(beta1_1d).add_(grad, alpha=1. - beta1_1d)
        exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1. - beta2_1d)
        bias_correction1 = 1.0 - beta1_1d ** step
        bias_correction2 = 1.0 - beta2_1d ** step
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
        else:
            denom = exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
        real_update_tmp = -lr * lr_1d_factor * torch.mul(p.data, weight_decay_1d).add(exp_avg.div(denom))
        p.data.add_(real_update_tmp)
    return exp_avg, exp_avg_sq


class MARS(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=0., amsgrad=False, gamma=0.025,
                 is_approx=True, mars_type="mars-adamw", optimize_1d=False, lr_1d=3e-3, betas_1d=(0.9, 0.95),
                 weight_decay_1d=0.1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        assert mars_type in ["mars-adamw", "mars-lion", "mars-shampoo"], "MARS type not supported"
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        mars_type=mars_type, gamma=gamma,
                        optimize_1d=optimize_1d, weight_decay_1d=weight_decay_1d)
        super(MARS, self).__init__(params, defaults)
        self.eps = eps
        self.update_fn = update_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.step_num = 0
        self.is_approx = is_approx
        self.gamma = gamma
        self.mars_type = mars_type
        self.optimize_1d = optimize_1d
        self.lr_1d_factor = lr_1d / lr
        self.weight_decay_1d = weight_decay_1d
        self.betas_1d = betas_1d

    @torch.no_grad()
    def update_last_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if "last_grad" not in state:
                        state["last_grad"] = torch.zeros_like(p)
                    state["last_grad"].zero_().add_(state["previous_grad"], alpha=1.0)

    @torch.no_grad()
    def update_previous_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                # print ("para name", len(group['params']), len(group['names']), group['names'])
                for p in group['params']:
                    # import pdb
                    # pdb.set_trace()
                    if p.grad is None:
                        print(p, "grad is none")
                        continue
                    state = self.state[p]
                    if "previous_grad" not in state:
                        state['previous_grad'] = torch.zeros_like(p)
                    state['previous_grad'].zero_().add_(p.grad, alpha=1.0)

    def __setstate__(self, state):
        super(MARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        If using exact version, the example usage is as follows:
            previous_X, previous_Y = None, None
            for epoch in range(epochs):
                for X, Y in data_loader:
                    if previous_X:
                        logits, loss = model(X, Y)
                        loss.backward()
                        optimizer.update_previous_grad()
                        optimizer.zero_grad(set_to_none=True)
                    logits, loss = model(X, Y)
                    loss.backward()
                    optimizer.step(bs=bs)
                    optimizer.zero_grad(set_to_none=True)
                    optimizer.update_last_grad()
                    iter_num += 1
                    previous_X, previous_Y = X.clone(), Y.clone()
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                'FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        real_update = 0
        real_update_wo_lr = 0
        gamma = self.gamma
        # import pdb
        # pdb.set_trace()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # ('----- starting a parameter state', state.keys(), 'Length of state', len(state))
                # State initialization
                if len(state) <= 1:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Last Gradient
                    state['last_grad'] = torch.zeros_like(p)
                    # state['previous_grad'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                # import pdb
                # pdb.set_trace()
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                last_grad = state['last_grad']
                lr, wd, beta1, beta2 = group['lr'], group['weight_decay'], *group['betas']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                else:
                    max_exp_avg_sq = 0

                if 'step' in state:
                    state['step'] += 1
                else:
                    state['step'] = 1
                step = state['step']
                is_grad_2d = (len(grad.shape) == 2)
                exp_avg, exp_avg_sq = self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    last_grad,
                    self.eps,
                    amsgrad,
                    max_exp_avg_sq,
                    step,
                    gamma,
                    mars_type=self.mars_type,
                    is_grad_2d=is_grad_2d,
                    optimize_1d=self.optimize_1d,
                    lr_1d_factor=self.lr_1d_factor,
                    betas_1d=self.betas_1d,
                    weight_decay_1d=self.weight_decay if self.optimize_1d else self.weight_decay_1d
                )
                if self.is_approx:
                    state['last_grad'] = grad
        self.step_num = step

        return loss


@torch.compile
def NewtonSchulz(M, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M.bfloat16() / (M.norm() + eps)
    if M.size(0) > M.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if M.size(0) > M.size(1):
        X = X.T
    return X.to(M.dtype)


class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(params_with_grad,
                    grads,
                    exp_avgs,
                    hessian,
                    state_steps,
                    bs=bs,
                    beta1=beta1,
                    beta2=beta2,
                    rho=group['rho'],
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    maximize=group['maximize'],
                    capturable=group['capturable'])

        return loss


def sophiag(params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            hessian: List[Tensor],
            state_steps: List[Tensor],
            capturable: bool = False,
            *,
            bs: int,
            beta1: float,
            beta2: float,
            rho: float,
            lr: float,
            weight_decay: float,
            maximize: bool):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_sophiag

    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)


def _single_tensor_sophiag(params: List[Tensor],
                           grads: List[Tensor],
                           exp_avgs: List[Tensor],
                           hessian: List[Tensor],
                           state_steps: List[Tensor],
                           *,
                           bs: int,
                           beta1: float,
                           beta2: float,
                           rho: float,
                           lr: float,
                           weight_decay: float,
                           maximize: bool,
                           capturable: bool):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step = step_t
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)