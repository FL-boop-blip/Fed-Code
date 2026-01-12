import os

from utils_general import *
from Data.utils_dataset import FEMNISTObject




data_path = 'Folder/'
def main():
    """
    Run FedAvg on FEMNIST data that was preprocessed via LEAF/femnist/preprocess.sh.
    """
    storage_path = 'LEAF/femnist/data/'
    dataset_prefix = 'femnist'
    n_client = 500
    data_obj = FEMNISTObject(storage_path, dataset_prefix, n_client=n_client)

    model_name = 'femnist'
    suffix = model_name
    com_amount = 1000
    save_period = 50
    weight_decay = 1e-4
    batch_size = 32
    act_prob = 0.02
    lr_decay_per_round = 0.995
    epoch = 5
    learning_rate = 0.1
    print_per = 5

    model_func = lambda: client_model(model_name)
    init_model = model_func()
    if not os.path.exists('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' %(data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' %(data_path, data_obj.name))
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)))

    # init_path = f'Output/{data_obj.name}/{model_name}_init_mdl.pt'
    # if not os.path.exists(init_path):
    #     os.makedirs(f'Output/{data_obj.name}', exist_ok=True)
    #     torch.save(init_model.state_dict(), init_path)
    # else:
    #     init_model.load_state_dict(torch.load(init_path))

    # print('FedAvg on FEMNIST')
    train_FedAvg(
        data_obj=data_obj,
        act_prob=act_prob,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epoch=epoch,
        com_amount=com_amount,
        print_per=print_per,
        weight_decay=weight_decay,
        model_func=model_func,
        init_model=init_model,
        sch_step=1, sch_gamma=1,
        save_period=save_period,
        data_path=data_path,    
        lr_decay_per_round=lr_decay_per_round,
    )


    print('SCAFFOLD')

    epoch = 5

    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)

    n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
    learning_rate = 0.1
    print_per = 5


    train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob,
    learning_rate=learning_rate, batch_size=batch_size, n_minibatch=n_minibatch, 
    com_amount=com_amount, print_per=n_minibatch//2, weight_decay=weight_decay, 
    model_func=model_func, init_model=init_model,
    sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, 
    trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)

    epoch = 5
    alpha_coef = 0.1
    rho = 0.1

    train_FedSpeed(data_obj=data_obj, act_prob=act_prob,
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
    

    epoch = 5
    alpha_coef = 0.1
    rho = 0.1
    learning_rate = 0.1
    print_per = epoch // 2
    
    train_A_FedPD(data_obj=data_obj, act_prob=act_prob,
                                                learning_rate=learning_rate,
                                                batch_size=batch_size, epoch=epoch,
                                                com_amount=com_amount, print_per=print_per,
                                                weight_decay=weight_decay,
                                                model_func=model_func, init_model=init_model,
                                                alpha_coef=alpha_coef, rho=rho,
                                                sch_step=1, sch_gamma=1, save_period=save_period,
                                                suffix=suffix, trial=False,
                                                data_path=data_path,
                                                lr_decay_per_round=lr_decay_per_round)
    
    print('FedDG')
#
    epoch = 5
    alpha_coef = 0.1
    learning_rate = 0.1
    print_per = epoch // 2
    beta = 0.1
    train_FedDG(
        data_obj=data_obj, act_prob=act_prob,
        learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
        com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
        model_func=model_func, init_model=init_model, alpha_coef=alpha_coef, beta=beta,
        sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, trial=False,
        data_path=data_path, lr_decay_per_round=lr_decay_per_round, sim_gamma=0.1)

    


if __name__ == '__main__':
    main()
