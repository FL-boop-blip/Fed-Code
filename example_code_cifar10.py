import torch

from utils_general import *
from Methods import *

# Dataset initialization
data_path = 'Folder/'  # The folder to save Data & Model

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in data_path + "Data/Raw/" folder.
#     - Change the datase                                                                                             t argument from CIFAR10 to emnist.
########
#      - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, name, crop_amount = 2000)
#########

if __name__ == '__main__':

    n_client = 500
    # Generate IID or Dirichlet distribution
    # IID
    data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, rule='Dirichlet', rule_arg=0.6,
                             unbalanced_sgm=0,
                             data_path=data_path)
    # unbalanced
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)

    # Dirichlet (0.6)
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
    # Dirichlet (0.3)
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)

    model_name = 'cifar10_LeNet'  # Model type
    # model_name = 'Resnet18'
    # model_name_1 = 'cifar10_LeNet'

    ###
    # Common hyperparameters

    com_amount = 1600
    save_period = 1600
    weight_decay = 1e-3
    batch_size = 50
    # act_prob = 1
    act_prob = 0.02
    suffix = model_name
    lr_decay_per_round = 0.998

    # Model function
    model_func = lambda: client_model(model_name)
    init_model = model_func()
    # Initalise the model for all methods with a random seed or load it from a saved initial model
    torch.manual_seed(37)
    if not os.path.exists('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' % (data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' % (data_path, data_obj.name))
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))

    #
    print('FedSpeed')
    
    epoch = 5
    alpha_coef = 0.1
    learning_rate = 0.1
    print_per = epoch // 2
    rho = 0.1

    #
    [avg_cld_mdls, tst_cur_cld_perf] = train_FedSpeed(data_obj=data_obj, act_prob=act_prob,
                                                      learning_rate=learning_rate,
                                                      batch_size=batch_size, epoch=epoch,
                                                      com_amount=com_amount, print_per=print_per,
                                                      weight_decay=weight_decay,
                                                      model_func=model_func, init_model=init_model,
                                                      alpha_coef=alpha_coef, rho=rho,
                                                      sch_step=1, sch_gamma=1,
                                                      save_period=save_period,
                                                      suffix=suffix, trial=False,
                                                      data_path=data_path,
                                                      lr_decay_per_round=lr_decay_per_round)
    # #
    # print('FedSMOO')
    
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedSMOO(data_obj=data_obj, act_prob=act_prob,
    #                                                  learning_rate=learning_rate,
    #                                                  batch_size=batch_size, epoch=epoch,
    #                                                  com_amount=com_amount, print_per=print_per,
    #                                                  weight_decay=weight_decay,
    #                                                  model_func=model_func, init_model=init_model,
    #                                                  alpha_coef=alpha_coef, rho=rho,
    #                                                  sch_step=1, sch_gamma=1,
    #                                                  save_period=save_period,
    #                                                  suffix=suffix, trial=False,
    #                                                  data_path=data_path,
    #                                                  lr_decay_per_round=lr_decay_per_round)
    #
    # print('FedDisco')
    #
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    #
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedDisco(data_obj=data_obj, act_prob=act_prob,
    #                                                   learning_rate=learning_rate,
    #                                                   batch_size=batch_size, epoch=epoch,
    #                                                   com_amount=com_amount, print_per=print_per,
    #                                                   weight_decay=weight_decay,
    #                                                   model_func=model_func, init_model=init_model,
    #                                                   alpha_coef=alpha_coef,
    #                                                   sch_step=1, sch_gamma=1,
    #                                                   save_period=save_period,
    #                                                   suffix=suffix, trial=False,
    #                                                   data_path=data_path,
    #                                                   lr_decay_per_round=lr_decay_per_round)

    # print('FedDC')
    
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    
    # n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    # n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
    # n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
    
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob,
    #                                                n_minibatch=n_minibatch,
    #                                                learning_rate=learning_rate, batch_size=batch_size,
    #                                                epoch=epoch,
    #                                                com_amount=com_amount, print_per=print_per,
    #                                                weight_decay=weight_decay,
    #                                                model_func=model_func, init_model=init_model,
    #                                                alpha_coef=alpha_coef,
    #                                                sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                suffix=suffix, trial=False,
    #                                                data_path=data_path,
    #                                                lr_decay_per_round=lr_decay_per_round)

    # print('FedDyn')
    
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedDyn(data_obj=data_obj, act_prob=act_prob,
    #                                                 learning_rate=learning_rate,
    #                                                 batch_size=batch_size, epoch=epoch,
    #                                                 com_amount=com_amount, print_per=print_per,
    #                                                 weight_decay=weight_decay,
    #                                                 model_func=model_func, init_model=init_model,
    #                                                 alpha_coef=alpha_coef,
    #                                                 sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                 suffix=suffix, trial=False,
    #                                                 data_path=data_path,
    #                                                 lr_decay_per_round=lr_decay_per_round)
    # # # # exit(0)
    # # # # ###
    # # print('SCAFFOLD')
    
    # epoch = 5
    
    # n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    # n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
    
    # n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
    # learning_rate = 0.1
    # print_per = 5
    
    # [fed_mdls_sel, tst_perf_sel] = train_SCAFFOLD(
    #     data_obj=data_obj, act_prob=act_prob,
    #     learning_rate=learning_rate, batch_size=batch_size, n_minibatch=n_minibatch,
    #     com_amount=com_amount, print_per=n_minibatch // 2, weight_decay=weight_decay,
    #     model_func=model_func, init_model=init_model,
    #     sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
    #     trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)
    # #
    # # ####
    # # print('FedAvg')
    
    # epoch = 5
    # learning_rate = 0.1
    # print_per = 5
    
    # [fed_mdls_sel, tst_perf_sel] = train_FedAvg(
    #     data_obj=data_obj, act_prob=act_prob,
    #     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
    #     com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
    #     model_func=model_func, init_model=init_model,
    #     sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
    #     trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)
    # #
    # # # ####
    # print('FedProx')
    
    # epoch = 5
    # learning_rate = 0.1
    # print_per = 5
    # alpha_coef = 0.1
    
    # [fed_mdls_sel, tst_perf_sel] = train_FedProx(
    #     data_obj=data_obj, act_prob=act_prob,
    #     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
    #     com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
    #     model_func=model_func, init_model=init_model, sch_step=1, sch_gamma=1,
    #     save_period=save_period, alpha_coef=alpha_coef, suffix=suffix, trial=False,
    #     data_path=data_path, lr_decay_per_round=lr_decay_per_round)

    # print('FedLESAM_D')
    
    # epoch = 5
    # learning_rate = 0.1
    # print_per = 5
    # alpha_coef = 0.1
    
    # [fed_mdls_sel, tst_perf_sel] = train_FedLESAM_D(
    #     data_obj=data_obj, act_prob=act_prob, rho=0.1,
    #     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
    #     com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
    #     model_func=model_func, init_model=init_model, sch_step=1, sch_gamma=1,
    #     save_period=save_period, alpha_coef=alpha_coef, suffix=suffix, trial=False,
    #     data_path=data_path, lr_decay_per_round=lr_decay_per_round)
    # #
   

    # print('FedCVC')
    
    # epoch = 5
    # alpha_coef = 0.1
    # rho = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    # beta = 0.1
    
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedCVC(data_obj=data_obj, act_prob=act_prob,
    #                                                learning_rate=learning_rate,
    #                                                batch_size=batch_size, epoch=epoch, beta=beta,
    #                                                com_amount=com_amount, print_per=print_per,
    #                                                weight_decay=weight_decay,
    #                                                model_func=model_func, init_model=init_model,
    #                                                alpha_coef=alpha_coef, rho=rho,
    #                                                sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                suffix=suffix, trial=False,
    #                                                data_path=data_path,
    #                                                lr_decay_per_round=lr_decay_per_round)
    # last_100_acc_tst = [tst_cur_cld_perf[i][1] for i in range(com_amount - 100, com_amount)]
    # mean_acc_tst = np.mean(last_100_acc_tst)
    # var_acc_tst = np.var(last_100_acc_tst)
    #
    # # 将结果保存在一个txt文件中
    # with open('Folder/Accuracy/results_FedFC.txt', 'w') as f:
    #     f.write(f"{mean_acc_tst} ± {var_acc_tst}\n")
    #
    # print("Results saved to results.txt")

    # print('A_FedPD')
    
    # epoch = 5
    # alpha_coef = 0.1
    # rho = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    
    # [avg_cld_mdls, tst_cur_cld_perf] = train_A_FedPD(data_obj=data_obj, act_prob=act_prob,
    #                                                  learning_rate=learning_rate,
    #                                                  batch_size=batch_size, epoch=epoch,
    #                                                  com_amount=com_amount, print_per=print_per,
    #                                                  weight_decay=weight_decay,
    #                                                  model_func=model_func, init_model=init_model,
    #                                                  alpha_coef=alpha_coef, rho=rho,
    #                                                  sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                  suffix=suffix, trial=False,
    #                                                  data_path=data_path,
    #                                                  lr_decay_per_round=lr_decay_per_round)

    # print('FedCM')
    #
    # epoch = 5
    #
    # n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    # n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
    #
    # n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
    # learning_rate = 0.1
    # alpha_coef = 0.1
    # print_per = 5
    #
    # [avg_cld_mdls, tst_cur_cld_perf] = train_FedCM(data_obj=data_obj, act_prob=act_prob,
    #                                                learning_rate=learning_rate, batch_size=batch_size,
    #                                                n_minibatch=n_minibatch, epoch=epoch,
    #                                                com_amount=com_amount, print_per=print_per,
    #                                                weight_decay=weight_decay,
    #                                                model_func=model_func, init_model=init_model,
    #                                                alpha_coef=alpha_coef,
    #                                                sch_step=1, sch_gamma=1, save_period=save_period,
    #                                                suffix=suffix, trial=False,
    #                                                data_path=data_path,
    #                                                lr_decay_per_round=lr_decay_per_round)


    # print('FedGamma')
    #
    # epoch = 5
    #
    # n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    # n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
    #
    # n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
    # learning_rate = 0.1
    # print_per = 5
    # rho = 0.1
    #
    # [fed_mdls_sel, tst_perf_sel] = train_FedGamma(data_obj=data_obj, act_prob=act_prob,
    #                                               learning_rate=learning_rate, rho=rho,
    #                                               batch_size=batch_size, n_minibatch=n_minibatch,
    #                                               com_amount=com_amount, print_per=n_minibatch // 2,
    #                                               weight_decay=weight_decay,
    #                                               model_func=model_func, init_model=init_model,
    #                                               sch_step=1, sch_gamma=1, save_period=save_period,
    #                                               suffix=suffix,
    #                                               trial=False, data_path=data_path,
    #                                               lr_decay_per_round=lr_decay_per_round)
    #



    # print('MoFedSAM')

    # epoch = 5

    # n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    # n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)

    # n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
    # learning_rate = 0.1
    # alpha_coef = 0.1
    # print_per = 5
    # rho = 0.1

    # [avg_cld_mdls, tst_cur_cld_perf] = train_MoFedSAM(data_obj=data_obj, act_prob=act_prob,
    #                                                   learning_rate=learning_rate, rho=rho,
    #                                                   batch_size=batch_size, n_minibatch=n_minibatch,
    #                                                   epoch=epoch,
    #                                                   com_amount=com_amount, print_per=print_per,
    #                                                   weight_decay=weight_decay,
    #                                                   model_func=model_func, init_model=init_model,
    #                                                   alpha_coef=alpha_coef,
    #                                                   sch_step=1, sch_gamma=1,
    #                                                   save_period=save_period, suffix=suffix,
    #                                                   trial=False,
    #                                                   data_path=data_path,
    #                                                   lr_decay_per_round=lr_decay_per_round)




    # print('FedGloSS')
    # rho = 0.15              # server-side SAM radius
    # rho_l = 0.15            # client-side SAM radius
    # eta = 0.0               # client-side SAM step size
    # beta = 10.0             # Lagrangian controller
    # rho0 = 0.001            # warmup starting rho
    # T_s = 200               # warmup length (FedGloSS uses 2000; use smaller for example)

    # server_lr = 1.0          # server optimizer learning rate (SGD)
    # server_momentum = 0.0
    # epoch = 5
    # alpha_coef = 0.1
    # learning_rate = 0.1
    # print_per = epoch // 2
    # avg_models, trn_perf, tst_perf = train_FedGloss(
    #     data_obj=data_obj,
    #     act_prob=act_prob,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     epoch=epoch,
    #     com_amount=com_amount,
    #     print_per=print_per,
    #     weight_decay=weight_decay,
    #     model_func=model_func,
    #     init_model=init_model,
    #     rho=rho,
    #     rho_l=rho_l,
    #     eta=eta,
    #     beta=beta,
    #     sch_step=1,
    #     sch_gamma=1.0,
    #     save_period=save_period,
    #     rho0=rho0,
    #     T_s=T_s,
    #     server_lr=server_lr,
    #     server_momentum=server_momentum,
    #     suffix=suffix,
    #     trial=False,
    #     data_path=data_path,
    #     rand_seed=0,
    #     lr_decay_per_round=lr_decay_per_round,
    # )
    print('RFL-DFDC')
#
    epoch = 5
    alpha_coef = 0.1
    learning_rate = 0.1
    print_per = epoch // 2
    beta = 0.1
    train_RFL_DFDC(
        data_obj=data_obj, act_prob=act_prob,
        learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
        com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
        model_func=model_func, init_model=init_model, alpha_coef=alpha_coef, beta=beta,
        sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, trial=False,
        data_path=data_path, lr_decay_per_round=lr_decay_per_round, sim_gamma=0.1)
