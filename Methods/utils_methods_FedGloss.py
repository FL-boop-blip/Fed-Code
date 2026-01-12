import numpy as np
from utils_libs import *
from utils_general import *
from Model.utils_models import *
from tensorboardX import SummaryWriter
from collections import Counter


def train_FedGloss(
    data_obj,
    act_prob,
    learning_rate,
    batch_size,
    epoch,
    com_amount,
    print_per,
    weight_decay,
    model_func,
    init_model,
    rho,          # server-side SAM radius
    rho_l,        # client-side SAM radius
    eta,          # client-side SAM step-size
    beta,         # Lagrangian controller
    sch_step,
    sch_gamma,
    save_period,
    rho0=0.001,
    T_s=0,
    server_lr=None,
    server_momentum=0.0,
    suffix='',
    trial=True,
    data_path='',
    rand_seed=0,
    lr_decay_per_round=1.0,
):
    """
    FedGloSS-style training loop in FL_BENCH.
    Returns:
        avg_models: list of global models saved every save_period rounds
        trn_perf:   [com_amount, 2]  (loss, acc) on central train data
        tst_perf:   [com_amount, 2]  (loss, acc) on test data
    """
    suffix = 'FedGloss_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_rho%f_rhol%f_beta%f' % (
        save_period,
        act_prob,
        learning_rate,
        sch_step,
        sch_gamma,
        batch_size,
        epoch,
        weight_decay,
        rho,
        rho_l,
        beta,
    )
    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_client = data_obj.n_client
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    if server_lr is None:
        server_lr = learning_rate

    # server/client SAM schedules
    rhos_l, server_rhos = _build_rho_schedules(rho, rho_l, rho0, T_s, com_amount)

    # Global model on server
    server_model = model_func().to(device)
    server_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    # Server optimizer (FedOpt-style SGD)
    server_optimizer = torch.optim.SGD(
        params=server_model.parameters(),
        lr=server_lr,
        momentum=server_momentum,
    )

    # Server state sigma and last pseudo-gradient
    sigma = OrderedDict()
    for name, p in server_model.named_parameters():
        sigma[name] = torch.zeros_like(p.data)
    delta_w_tilde = None
    epsilon_tilde = None

    # Per-client dual states
    client_states = []
    for _ in range(n_client):
        st = OrderedDict()
        for name, p in server_model.named_parameters():
            st[name] = torch.zeros_like(p.data)
        client_states.append(st)

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_models = list(range(n_save_instances))

    trn_perf = np.zeros((com_amount, 2))
    tst_perf = np.zeros((com_amount, 2))

    writer = SummaryWriter('%sRuns_FedGloss/%s/%s' % (data_path, data_obj.name, suffix))

    for t in range(com_amount):
        # ----- server ascent step (SAM on server) -----
        if delta_w_tilde is not None:
            grads = []
            for k in delta_w_tilde:
                grads.append(torch.norm(delta_w_tilde[k], p=2))
            grad_norm = torch.norm(torch.stack(grads), p=2) + 1e-16

            epsilon_tilde = OrderedDict()
            for name, p in server_model.named_parameters():
                if name not in delta_w_tilde:
                    continue
                e = delta_w_tilde[name] * (server_rhos[t] / grad_norm)
                epsilon_tilde[name] = e
                with torch.no_grad():
                    p.add_(e)

        # ----- select active clients -----
        inc_seed = 0
        while True:
            np.random.seed(t + rand_seed + inc_seed)
            act_list = np.random.uniform(size=n_client)
            act_clients = act_list <= act_prob
            selected_clnts = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clnts) != 0:
                break
        print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

        # ----- train selected clients -----
        updates = []
        server_state_before = copy.deepcopy(server_model.state_dict())

        for clnt in selected_clnts:
            print('---- Training client %d' % clnt)
            trn_x = clnt_x[clnt]
            trn_y = clnt_y[clnt]

            local_model = model_func().to(device)
            local_model.load_state_dict(copy.deepcopy(server_state_before))
            for p in local_model.parameters():
                p.requires_grad = True

            initial_state = copy.deepcopy(server_state_before)
            local_state = client_states[clnt]

            local_lr = learning_rate * (lr_decay_per_round ** t)

            local_model, updated_state, final_state = train_model_FedGloss(
                local_model,
                trn_x,
                trn_y,
                initial_state,
                local_state,
                beta,
                rhos_l[t],
                eta,
                local_lr,
                batch_size,
                epoch,
                print_per,
                weight_decay,
                data_obj.dataset,
                sch_step,
                sch_gamma,
            )
            client_states[clnt] = updated_state

            # compute client delta (FedOpt-style)
            delta = OrderedDict()
            for k, x in server_state_before.items():
                y = final_state[k]
                if ('running' in k) or ('num_batches_tracked' in k):
                    delta[k] = y.clone().detach()
                else:
                    delta[k] = (y - x).clone().detach()

            num_samples = len(trn_y)
            updates.append((num_samples, delta))

        # ----- update server sigma -----
        if updates:
            for name in sigma:
                deviation = torch.zeros_like(sigma[name])
                for _, delta in updates:
                    deviation += delta[name]
                sigma[name] -= deviation / (beta * float(n_client) * float(len(updates)))

        # ----- remove server perturbation -----
        if epsilon_tilde is not None:
            for name, p in server_model.named_parameters():
                if name in epsilon_tilde:
                    with torch.no_grad():
                        p.sub_(epsilon_tilde[name])

        # ----- FedOpt-style aggregation -----
        server_optimizer.zero_grad()
        pseudo_grad = _average_deltas(updates, device)

        # set gradients and take optimizer step
        for name, p in server_model.named_parameters():
            if name in pseudo_grad:
                p.grad = -1.0 * pseudo_grad[name]
        server_optimizer.step()

        # update BN running stats if they exist
        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_grad.items() if ("running" in k) or ("num_batches_tracked" in k)}
        )
        if len(bn_layers) > 0:
            server_model.load_state_dict(bn_layers, strict=False)

        # ----- save pseudo-gradient for next SAM step -----
        delta_w_tilde = OrderedDict()
        for name, p in server_model.named_parameters():
            if p.grad is not None:
                delta_w_tilde[name] = p.grad.detach().clone()

        # ----- solve Lagrangian on server -----
        with torch.no_grad():
            for name, p in server_model.named_parameters():
                if name in sigma:
                    p.sub_(beta * sigma[name])

        # ----- evaluation -----
        loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, server_model, data_obj.dataset, 0)
        tst_perf[t] = [loss_tst, acc_tst]
        print("**** FedGloss Round %3d, Test Accuracy: %.4f, Loss: %.4f" % (t + 1, acc_tst, loss_tst))

        loss_trn, acc_trn = get_acc_loss(cent_x, cent_y, server_model, data_obj.dataset, weight_decay)
        trn_perf[t] = [loss_trn, acc_trn]
        print("**** FedGloss Round %3d, Train Accuracy: %.4f, Loss: %.4f" % (t + 1, acc_trn, loss_trn))

        writer.add_scalars(
            'Loss/test',
            {'FedGloss': tst_perf[t][0]},
            t,
        )
        writer.add_scalars(
            'Accuracy/test',
            {'FedGloss': tst_perf[t][1]},
            t,
        )

        if (not trial) and ((t + 1) % save_period == 0):
            torch.save(
                server_model.state_dict(),
                '%sModel/%s/%s/%dcom_global.pt' % (data_path, data_obj.name, suffix, (t + 1)),
            )
            np.save(
                '%sModel/%s/%s/%d_com_tst_perf.npy' % (data_path, data_obj.name, suffix, (t + 1)),
                tst_perf[: t + 1],
            )

            if (t + 1) > save_period:
                prev = t + 1 - save_period
                prev_path = '%sModel/%s/%s/%d_com_trn_perf.npy' % (data_path, data_obj.name, suffix, prev)
                if os.path.exists(prev_path):
                    os.remove(
                        '%sModel/%s/%s/%d_com_tst_perf.npy' % (data_path, data_obj.name, suffix, prev)
                    )

            avg_models[t // save_period] = copy.deepcopy(server_model)

    return avg_models, trn_perf, tst_perf




def _build_rho_schedules(rho_server, rho_client, rho0, T_s, T):
    """
    Build constant (or warmup) schedules for server/client SAM radii.
    For now we keep them constant after an optional short warmup.
    """
    if T_s is None or T_s <= 0:
        rhos_l = [rho_client] * T
        server_rhos = [rho_server] * T
        return rhos_l, server_rhos

    T_s = min(T_s, T)

    rhos_l = [rho0]
    delta_l = (rho_client - rho0) / float(T_s)
    for _ in range(T_s):
        rhos_l.append(rhos_l[-1] + delta_l)
    if len(rhos_l) < T:
        rhos_l += [rho_client] * (T - len(rhos_l))
    else:
        rhos_l = rhos_l[:T]

    server_rhos = [rho0]
    delta_s = (rho_server - rho0) / float(T_s)
    for _ in range(T_s):
        server_rhos.append(server_rhos[-1] + delta_s)
    if len(server_rhos) < T:
        server_rhos += [rho_server] * (T - len(server_rhos))
    else:
        server_rhos = server_rhos[:T]

    return rhos_l, server_rhos


def _average_deltas(weighted_deltas, device):
    """
    weighted_deltas: list of (num_samples, delta_state_dict)
    Returns pseudo_gradient: dict name -> tensor
    """
    total_weight = 0.0
    base = {}
    for num_samples, delta in weighted_deltas:
        if num_samples <= 0:
            continue
        total_weight += num_samples
        for k, v in delta.items():
            if v is None:
                continue
            if k in base:
                base[k] += num_samples * v.float()
            else:
                base[k] = num_samples * v.float()

    if total_weight == 0.0:
        # No updates; return zeros
        if not base:
            return {}
        total_weight = 1.0

    pseudo_grad = {}
    for k, v in base.items():
        pseudo_grad[k] = (v.to(device) / total_weight)
    return pseudo_grad




class FedGlossSAM:
    """
    SAM-style optimizer used in FedGloSS.
    This is a lightweight copy of the implementation in fedgloss/codebase.
    """

    def __init__(self, optimizer, model, rho, eta):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        if not grads:
            return
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.0e-16
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def remove_perturbation(self):
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if "eps" in self.state[p]:
                p.sub_(self.state[p]["eps"])


class FedGloSSOpt(FedGlossSAM):
    """
    FedGloSS optimizer: SAM ascent plus Lagrangian descent modulation.
    """

    def __init__(self, optimizer, model, rho, eta, beta):
        super().__init__(optimizer, model, rho, eta)
        self.beta = beta

    @torch.no_grad()
    def descent_step(self, initial_model_state, local_state):
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            # remove the perturbation applied in ascent_step
            if "eps" in self.state[p]:
                p.sub_(self.state[p]["eps"])

            # add components to solve the Lagrangian function
            if name in local_state:
                p.grad.sub_(local_state[name])
            if name in initial_model_state:
                p.grad.add_((p - initial_model_state[name]) / self.beta)

        self.optimizer.step()
        self.optimizer.zero_grad()


def train_model_FedGloss(
    model,
    trn_x,
    trn_y,
    initial_state,
    local_state,
    beta,
    rho_l,
    eta,
    learning_rate,
    batch_size,
    epoch,
    print_per,
    weight_decay,
    dataset_name,
    sch_step,
    sch_gamma,
    momentum=0.0,
):
    """
    Local client training for FedGloSS using SAM + Lagrangian terms.
    """
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(
        Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
        batch_size=batch_size,
        shuffle=True,
    )
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    minimizer = FedGloSSOpt(optimizer, model, rho_l, eta, beta)

    running_loss = 0.0
    i = 0

    for e in range(epoch):
        trn_gen_iter = trn_gen.__iter__()
        for _ in range(int(np.ceil(n_trn / batch_size))):
            batch = next(trn_gen_iter, None)
            if batch is None:
                break
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()

            # ascent step
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            minimizer.ascent_step()

            # descent step
            outputs = model(batch_x)
            loss2 = criterion(outputs, batch_y)
            loss2.backward()
            minimizer.descent_step(initial_state, local_state)

            with torch.no_grad():
                running_loss += loss.item()
            i += 1

        if (e + 1) % max(1, print_per) == 0 and i > 0:
            avg_loss = running_loss / float(i)
            print("Epoch %3d, FedGloss local loss: %.4f, LR: %.4f" % (e + 1, avg_loss, scheduler.get_lr()[0]))

        scheduler.step()

    final_state = copy.deepcopy(model.state_dict())

    # update local dual state
    with torch.no_grad():
        for name in local_state.keys():
            if name in final_state and name in initial_state:
                local_state[name] -= (final_state[name] - initial_state[name]) / beta

    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model, local_state, final_state