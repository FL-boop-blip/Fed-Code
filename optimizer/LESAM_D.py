import torch
import torch.nn.functional as F
import numpy as np

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)


class LESAM_D(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM_D, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, g_update):
        # If no update direction is provided, skip the SAM perturbation
        if g_update is None:
            return

        # Convert g_update (which may be a numpy array or tensor, possibly 2D)
        # into a flat 1D torch tensor on the same device as the model params.
        if isinstance(g_update, np.ndarray):
            g_vec = torch.from_numpy(g_update).float()
        elif isinstance(g_update, torch.Tensor):
            g_vec = g_update.detach().float()
        else:
            # Fallback for Python lists or other array-likes
            g_vec = torch.tensor(g_update, dtype=torch.float32)

        g_vec = g_vec.view(-1)

        # Determine total number of parameters and reference device
        first_param = None
        total_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                total_params += p.numel()
                if first_param is None:
                    first_param = p

        if first_param is None or total_params == 0:
            return

        device = first_param.device
        g_vec = g_vec.to(device)

        if g_vec.numel() != total_params:
            raise ValueError(
                f"g_update has {g_vec.numel()} elements, but model parameters require {total_params} elements."
            )

        # First pass: compute aggregate norm (sum of per-parameter L2 norms)
        grad_norm = torch.tensor(0.0, device=device)
        idx_flat = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.requires_grad = True
                length = p.numel()
                g_p = g_vec[idx_flat:idx_flat + length].view_as(p)
                grad_norm = grad_norm + g_p.norm(p=2)
                idx_flat += length

        # Second pass: apply perturbation proportional to g_update
        idx_flat = 0
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                p.requires_grad = True
                length = p.numel()
                g_p = g_vec[idx_flat:idx_flat + length].view_as(p)
                idx_flat += length

                # original SAM
                # e_w = p.grad * scale.to(p)
                # ASAM
                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)

                # Direction based on provided global update g_update
                e_w = -g_p * scale.to(p)

                # climb to the local maximum "w + e(w)"
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, g_update):
        # Unpack the stored context for the current step.
        if self.paras is None:
            raise ValueError("LESAM_D.paras is not set before calling step().")

        if len(self.paras) == 4:
            # Standard loss: loss_fn(predictions, labels)
            inputs, labels, loss_func, model = self.paras
            delta_list, lamb = None, None
        elif len(self.paras) == 6:
            # Dynamic loss: loss_fn(predictions, labels, param_list, delta_list, lamb)
            inputs, labels, loss_func, model, delta_list, lamb = self.paras
        else:
            raise ValueError(
                f"LESAM_D.paras has unexpected length {len(self.paras)} (expected 4 or 6)."
            )

        self.zero_grad()

        self.first_step(g_update)

        param_list = param_to_vector(model)
        predictions = model(inputs)
        if delta_list is not None and lamb is not None:
            loss = loss_func(predictions, labels, param_list, delta_list, lamb)
        else:
            loss = loss_func(predictions, labels)

        self.zero_grad()
        loss.backward()

        self.second_step()
