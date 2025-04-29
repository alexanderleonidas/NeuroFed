import torch

class PerturbationOptimizer(torch.optim.Optimizer):
    """Custom optimizer for weight-perturbation-based learning.
    """
    def __init__(self, params, lr=0.0001, sigma=0.00001):
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive.")
        if sigma <= 0.0:
            raise ValueError("Perturbation size must be positive.")
        defaults = dict(lr=lr, sigma=sigma)
        super(PerturbationOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError("Gradient-free optimization requires a closure to compute loss.")

        loss_before = closure()
        noise = {}
        for group_idx, group in enumerate(self.param_groups):
            noise[group_idx] = []
            sigma = group['sigma']
            for param_idx, param in enumerate(group['params']):
                if param.requires_grad:
                    noise[group_idx].append(torch.normal(mean=0, std=sigma, size=param.data.shape).to(param.device))
                    param.add_(noise[group_idx][param_idx])

        noisy_loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            sigma = group['sigma']
            for param_idx, param in enumerate(group['params']):
                if param.requires_grad:
                    update = lr * (noisy_loss - loss_before) / (sigma ** 2)
                    param.sub_(update * noise[group_idx][param_idx])

        return loss_before