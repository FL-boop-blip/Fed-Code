
import numpy as np
from utils_libs import *
from utils_general import *
from Model.utils_models import *
from tensorboardX import SummaryWriter
from collections import Counter



def train_FedDA2(data_obj, act_prob,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func, model_func1,init_model,init_model1, alpha_coef,beta,
                sch_step, sch_gamma, save_period,T,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedDA2_' + str(epoch) + str(alpha_coef) + str(beta) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_client = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y




    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))


    tst_cur_cld_perf = np.zeros((com_amount, 2))
    residual_metrics = np.zeros((com_amount, 4))

    n_par = len(get_mdl_params([model_func()])[0])
    n_par1 = len(get_mdl_params([model_func1()])[0])

    parameter_drifts = np.zeros((n_client, n_par),dtype='float32')
    parameter_drifts_s = np.zeros((n_client,n_par),dtype='float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_client X n_par
    clnt_models = list(range(n_client))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedDA2/%s/%s' % (data_path, data_obj.name, suffix[:]))
    log_dir = os.path.join(writer.logdir, 'Logs') if hasattr(writer, 'logdir') and writer.logdir else 'Logs'

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt'
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
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cur_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
            cur_cld_model1 = model_func1().to(device)
            cur_cld_model1.load_state_dict(copy.deepcopy(dict(init_model1.named_parameters())))
            cld_mdl_pre_param = get_mdl_params([cur_cld_model1],n_par1)[0]
            client_gradients = []
            normalized_sums = 0.0
            g_gradient = np.zeros_like(cld_mdl_pre_param)
            for clnt in clnt_models:  # 0 - n_client
                print('pre-training for gradient diversity {}'.format(clnt))
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func1().to(device)
                model = clnt_models[clnt]

                model.load_state_dict(copy.deepcopy(
                    dict(cur_cld_model1.named_parameters())))  # use cloud model paramter to init local moedel
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_gradient = compute_client_gradient(model, trn_x, trn_y, batch_size, dataset_name=data_obj.dataset)
                client_gradients.append(clnt_gradient)
                if len(client_gradients) == 50:
                    for i, grad in enumerate(client_gradients):
                        g_gradient += 1/ n_client * client_gradients[i]
                        grad_norm_squared = np.linalg.norm(grad, 2) ** 2
                        normalized_sums += 1/ n_client * weight_list[i] * grad_norm_squared
                    client_gradients.clear()

            total_gradient_norm = np.linalg.norm(g_gradient, 2) ** 2

            diversity = np.sqrt(normalized_sums / total_gradient_norm)
            print('diversity is {}'.format(diversity))
            del client_gradients

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        prev_global_param = cld_mdl_param.copy()
        prev_global_dual = np.zeros_like(prev_global_param)

        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_client))

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                # clnt_models[clnt] = model_func().to(device)
                # model = clnt_models[clnt]
                model = model_func().to(device)
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                alpha = alpha_coef / weight_list[clnt]
                parameter_drifts_curr = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                if i > ( com_amount / 2 ) and T:
                    kd_T = T
                    clnt_models[clnt] = train_model_FedDA2(model, model_func, cur_cld_model,alpha,
                                                      global_mdl, parameter_drifts_curr,
                                                      trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                      batch_size, epoch, kd_T,print_per, weight_decay, data_obj.dataset,
                                                      sch_step, sch_gamma)
                else:
                    clnt_models[clnt] = train_model_FedDA2_wo_T(model, model_func, alpha,
                                                           global_mdl, parameter_drifts_curr,
                                                           trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                           batch_size, epoch, print_per, weight_decay,
                                                           data_obj.dataset,
                                                           sch_step, sch_gamma)
                del parameter_drifts_curr
                del model
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                parameter_drifts[clnt] += diversity * delta_param_curr
                clnt_params_list[clnt] = curr_model_par
                parameter_drifts_s[clnt] += 1 / act_prob * beta * delta_param_curr
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts_s, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)



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
            prev_global_dual = update_residual_metrics(residual_metrics, i, clnt_params_list, selected_clnts,
                                                       cld_mdl_param, prev_global_param, prev_global_dual)
            log_residual_scalars(writer, 'Current cloud', residual_metrics[i], i)
            prev_global_param = cld_mdl_param.copy()
            torch.cuda.empty_cache()

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_parameter_drifts.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    save_last_window_stats(suffix, tst_cur_cld_perf[:, 0], tst_cur_cld_perf[:, 1], output_dir=log_dir)
    register_method_performance(suffix, tst_cur_cld_perf[:, 0], tst_cur_cld_perf[:, 1], residual_metrics)

    return avg_cld_mdls, tst_cur_cld_perf



def train_model_FedDA2(model, model_func, cur_cld_model, alpha, global_model_param, parameter_drifts_curr,
                       trn_x, trn_y,
                       learning_rate, batch_size, epoch, kd_T, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_div = DistillKL(kd_T)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=0.1)

    model.train()
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

            y_pred = model(batch_x)
            y_t = cur_cld_model(batch_x)
            loss_k_i = loss_div(y_pred, y_t)
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

            loss_correct = loss_cp + loss_k_i
            loss_correct.backward()

            # loss = loss_f_i + loss_cp + loss_cg
            # optimizer.zero_grad()
            # loss.backward()
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

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_FedDA2_wo_T(model, model_func, cur_cld_model, alpha, global_model_param, parameter_drifts_curr,
                       trn_x, trn_y,
                       learning_rate, batch_size, epoch, kd_T, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True) 

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=0.1)

    model.train()
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

            # loss = loss_f_i + loss_cp + loss_cg
            # optimizer.zero_grad()
            # loss.backward()
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

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
