from utils_general import *
data_path = 'Folder/'
# Dataset initialization

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

#     - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
#########

########
# For FEMNIST experiments
# Generate the dataset with LEAF/femnist/preprocess.sh and set storage_path
# storage_path = 'LEAF/femnist/data/'
# name = 'femnist_experiment'
# data_obj = FEMNISTObject(storage_path, name, n_client=100)
# model_name = 'femnist'
########


# Generate IID or Dirichlet distribution
# IID
n_client = 500
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='iid', unbalanced_sgm=0)
storage_path = 'LEAF/shakespeare/data/'
dataset_prefix = 'shakepeare_nonIID'
data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix, n_client=n_client)
n_client = data_obj.n_client
###
model_name         = 'shakes_LSTM' # Model type
suffix = model_name
com_amount         = 1000
save_period        = 1000
weight_decay       = 1e-3
batch_size         = 50
act_prob           = 0.02
lr_decay_per_round = 1
epoch              = 5
learning_rate      = 0.8
print_per          = 2

# Model function
model_func = lambda : client_model(model_name)
init_model = model_func()
# Initalise the model for all methods or load it from a saved initial model
if not os.path.exists('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' %(data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' %(data_path, data_obj.name))
            torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name))
        else:
            # Load model
            init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)))
# if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
#     print("New directory!")
#     os.mkdir('Output/%s/' %(data_obj.name))
#     torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
# else:
#     # Load model
#     init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    
    
# Methods    
####
# print('FedDyn')

# alpha_coef = 1e-2
# [fed_mdls_sel_FedFyn, trn_perf_sel_FedFyn, tst_perf_sel_FedFyn,
#  fed_mdls_all_FedFyn, trn_perf_all_FedFyn, tst_perf_all_FedFyn,
#  fed_mdls_cld_FedFyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                      model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
#                                      save_period=save_period, lr_decay_per_round=lr_decay_per_round)

# ###
# print('SCAFFOLD')
# n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / data_obj.n_client
# n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
# n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
# print_per_ = print_per*n_iter_per_epoch

# [fed_mdls_sel_SCAFFOLD, trn_perf_sel_SCAFFOLD, tst_perf_sel_SCAFFOLD,
#  fed_mdls_all_SCAFFOLD, trn_perf_all_SCAFFOLD,
#  tst_perf_all_SCAFFOLD] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
#                                          batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
#                                          print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
#                                          init_model=init_model, save_period=save_period, lr_decay_per_round=lr_decay_per_round)
    
####
# print('FedAvg')

# train_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                      model_func=model_func, init_model=init_model, sch_step=1, sch_gamma=1, save_period=save_period,suffix=suffix,
#                                      lr_decay_per_round=lr_decay_per_round)
        
#### 
# print('FedProx')


# train_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
#                                      epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,alpha_coef=0.1,
#                                      model_func=model_func, init_model=init_model, save_period=save_period,suffix=suffix,sch_gamma=1,sch_step=1,
#                                      lr_decay_per_round=lr_decay_per_round)



# print('SCAFFOLD')

# epoch = 5

# n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
# n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)

# n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
# learning_rate = 0.1
# print_per = 5


# train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob,
# learning_rate=learning_rate, batch_size=batch_size, n_minibatch=n_minibatch, 
# com_amount=com_amount, print_per=n_minibatch//2, weight_decay=weight_decay, 
# model_func=model_func, init_model=init_model,
# sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, 
# trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)

# epoch = 5
# alpha_coef = 0.1
# rho = 0.1

# train_FedSpeed(data_obj=data_obj, act_prob=act_prob,
#                                                     learning_rate=learning_rate,
#                                                     batch_size=batch_size, epoch=epoch,
#                                                     com_amount=com_amount, print_per=print_per,
#                                                     weight_decay=weight_decay,
#                                                     model_func=model_func, init_model=init_model,
#                                                     alpha_coef=alpha_coef, rho=rho,
#                                                     sch_step=1, sch_gamma=1,
#                                                     save_period=save_period,
#                                                     suffix=suffix, trial=False,
#                                                     data_path=data_path,
#                                                     lr_decay_per_round=lr_decay_per_round)


# epoch = 5
# alpha_coef = 0.1
# rho = 0.1
# learning_rate = 0.1
# print_per = epoch // 2

# train_A_FedPD(data_obj=data_obj, act_prob=act_prob,
#                                             learning_rate=learning_rate,
#                                             batch_size=batch_size, epoch=epoch,
#                                             com_amount=com_amount, print_per=print_per,
#                                             weight_decay=weight_decay,
#                                             model_func=model_func, init_model=init_model,
#                                             alpha_coef=alpha_coef, rho=rho,
#                                             sch_step=1, sch_gamma=1, save_period=save_period,
#                                             suffix=suffix, trial=False,
#                                             data_path=data_path,
#                                             lr_decay_per_round=lr_decay_per_round)

# print('FedDG')
# #
# epoch = 5
# alpha_coef = 1e-3
# learning_rate = 0.1
# print_per = epoch // 2
# beta = 0.1
# train_FedDG(
#     data_obj=data_obj, act_prob=act_prob,
#     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
#     com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#     model_func=model_func, init_model=init_model, alpha_coef=alpha_coef, beta=beta,
#     sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, trial=False,
#     data_path=data_path, lr_decay_per_round=lr_decay_per_round, sim_gamma=0.1)

# # Plot results
# plt.figure(figsize=(6, 5))
# plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn[:,1], label='FedDyn')
# plt.ylabel('Test Accuracy', fontsize=16)
# plt.xlabel('Communication Rounds', fontsize=16)
# plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
# plt.grid()
# plt.xlim([0, com_amount+1])
# plt.title(data_obj.name, fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.savefig('Output/%s/plot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# # plt.show() 
