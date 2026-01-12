import numpy as np
from utils_libs import *
from utils_general import *
from Model.utils_models import *
from tensorboardX import SummaryWriter
from collections import Counter



def train_FedGamma(data_obj, act_prob, learning_rate, rho,epoch,batch_size, n_minibatch,
                   com_amount, print_per, weight_decay,
                   model_func, init_model, sch_step, sch_gamma,
                   save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                   global_learning_rate=1):
    suffix = 'FedGamma_%d'%epoch + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_client = data_obj.n_client

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client  # normalize it

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params_diffs = np.zeros((n_client + 1, n_par)).astype('float32')  # including cloud state
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_client X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedGamma/%s/%s' % (data_path, data_obj.name, suffix[:]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1))  # Get state_params_diffs
                    state_params_diffs = np.load(
                        '%sModel/%s/%s/%d_state_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_client))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            del clnt_models

            clnt_models = list(range(n_client))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True

                # Scale down c
                state_params_diff_curr = torch.tensor(
                    -state_params_diffs[clnt] + state_params_diffs[-1] / weight_list[clnt], dtype=torch.float32,
                    device=device)

                clnt_models[clnt] = train_model_Fedgamma(clnt_models[clnt], model_func, state_params_diff_curr, trn_x,
                                                       trn_y,
                                                       learning_rate * (lr_decay_per_round ** i),rho, batch_size,
                                                       n_minibatch, print_per,
                                                       weight_decay, data_obj.dataset, sch_step, sch_gamma)

                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_params_diffs[clnt] - state_params_diffs[-1] + 1 / n_minibatch / learning_rate * (
                            prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_params_diffs[clnt]) * weight_list[clnt]
                state_params_diffs[clnt] = new_c

                clnt_params_list[clnt] = curr_model_param

            avg_model_params = global_learning_rate * np.mean(clnt_params_list[selected_clnts], axis=0) + (
                        1 - global_learning_rate) * prev_params

            avg_model = set_client_from_params(model_func().to(device), avg_model_params)

            state_params_diffs[-1] += 1 / n_client * delta_c_sum

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        state_params_diffs)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period)):
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%d_state_params_diffs.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel



def train_model_Fedgamma(model, model_func, state_params_diff, trn_x, trn_y,
                         learning_rate, rho, batch_size, n_minibatch,print_per,
                         weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=rho)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_trn / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    count_step = 0
    is_done = False

    step_loss = 0;
    n_data_step = 0
    for e in range(epoch):
        # Training
        ###
        if is_done:
            break
        ###

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            ####
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            ###
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * state_params_diff)

            loss_algo.backward()

            ###
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()  # Clip gradients to prevent exploding
            step_loss += loss_algo.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay) / 2 * np.sum(params * params)

                print("Step %3d, Training Loss: %.4f, LR: %.5f"
                      % (count_step, step_loss, scheduler.get_lr()[0]))
                step_loss = 0;
                n_data_step = 0

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model