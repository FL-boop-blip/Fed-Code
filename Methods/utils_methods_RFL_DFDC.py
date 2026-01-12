import numpy as np
from utils_libs import *
from utils_general import *
from Model.utils_models import *
from tensorboardX import SummaryWriter
from collections import Counter


def train_RFL_DFDC(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef, beta,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                 sim_gamma=0.1):
    suffix = 'RFL_DFDC_' + suffix
    suffix += '_a%f_b%f_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_g%f' % (alpha_coef,beta,
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size,
        epoch, weight_decay, sim_gamma)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    adjacency = np.eye(n_clnt).astype('float32')
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedDG/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    # trn_cur_cld_perf[:i + 1] = np.load(
                    #     '%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist    
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))
            clnt_updates = {}
            amp_weights = {clnt: 1 for clnt in selected_clnts}

            for clnt in selected_clnts:
                # Train locally 
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedCVC(model, model_func, alpha_coef_adpt,
                                                       cld_mdl_param_tensor, hist_params_diffs_curr,
                                                       trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, epoch, print_per, weight_decay,
                                                       data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par
                clnt_updates[clnt] = curr_model_par - cld_mdl_param

            for idx_i in range(len(selected_clnts)):
                i_clnt = selected_clnts[idx_i]
                adjacency[i_clnt, i_clnt] = 1.0
                for idx_j in range(idx_i + 1, len(selected_clnts)):
                    j_clnt = selected_clnts[idx_j]
                    sim = _cosine_similarity(clnt_updates[i_clnt], clnt_updates[j_clnt])
                    adjacency[i_clnt, j_clnt] = (1 - sim_gamma) * adjacency[i_clnt, j_clnt] + sim_gamma * sim
                    adjacency[j_clnt, i_clnt] = adjacency[i_clnt, j_clnt]

            for clnt in unselected_clnts:
                neighbor_weights = adjacency[clnt, selected_clnts] if len(selected_clnts) > 0 else np.array([])
                weight_sum = np.sum(neighbor_weights)
                if len(neighbor_weights) == 0 or weight_sum <= 1e-12:
                    continue
                normalized = neighbor_weights / weight_sum
                drift_projection = np.zeros(n_par)
                for idx, active_client in enumerate(selected_clnts):
                    drift_projection += normalized[idx] * clnt_updates[active_client]
                hist_params_diffs[clnt] = hist_params_diffs[clnt] + beta * drift_projection
            

            for clnt in unselected_clnts:
                neighbor_weights = adjacency[clnt, selected_clnts] if len(selected_clnts) > 0 else np.array([])
                if len(neighbor_weights) == 0:
                    continue
                best_idx = np.argmax(neighbor_weights)
                if neighbor_weights[best_idx] <= 0:
                    continue
                best_client = selected_clnts[best_idx]
                amp_weights[best_client] = amp_weights.get(best_client, 1) + 1

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            # dual_correction = np.zeros(n_par)
            # for clnt in selected_clnts:
            #     print('Client %d amplification weight: %.4f' % (clnt, amp_weights[clnt]))
            #     eff = amp_weights[clnt]
            #     # if eff > 50:
            #     #     dual_correction += 1 + 0.1 * 49 * hist_params_diffs[clnt]
            #     # else:
            #     #     dual_correction +=1 + (eff-1) * 0.1 * hist_params_diffs[clnt]
            #     # dual_correction += eff * hist_params_diffs[clnt]
            #     dual_correction += hist_params_diffs[clnt]
            
            # for clnt in unselected_clnts:
            #     dual_correction += hist_params_diffs[clnt]

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)
            # cld_mdl_param = avg_mdl_param_sel + dual_correction / n_clnt

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            # loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
            #                                  cur_cld_model, data_obj.dataset, 0)
            # print("**** Cur cld Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
            #       % (i + 1, acc_tst, loss_tst))
            # trn_cur_cld_perf[i] = [loss_tst, acc_tst]

            # writer.add_scalars('Loss/train',
            #                    {
            #                        'Current cloud': trn_cur_cld_perf[i][0]
            #                    }, i
            #                    )

            # writer.add_scalars('Accuracy/train',
            #                    {
            #                        'Current cloud': trn_cur_cld_perf[i][1]
            #                    }, i
            #                    )

            # writer.add_scalars('Loss/train_wd',
            #                    {
            #                        'Current cloud':
            #                            get_acc_loss(cent_x, cent_y, cur_cld_model, data_obj.dataset, weight_decay)[0]
            #                    }, i
            #                    )

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                # np.save('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                #         trn_cur_cld_perf[:i + 1])
                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    # os.remove('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (
                    # data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf



def _cosine_similarity(delta_a, delta_b):
    """Cosine similarity with [0,1] scaling as defined in RGFL.md."""
    denom = (np.linalg.norm(delta_a) * np.linalg.norm(delta_b)) + 1e-12
    return 0.5 * (np.dot(delta_a, delta_b) / denom + 1.0)



def train_model_FedCVC(model, model_func, alpha, global_model_param, parameter_drifts_curr,
                            trn_x, trn_y,
                            learning_rate, batch_size, epoch, print_per,
                            weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=0.1)

    # model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - parameter_drifts_curr)) * (
                        local_parameter - (global_model_param - parameter_drifts_curr)))

            loss_correct = loss_cp
            loss_correct.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            # model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model