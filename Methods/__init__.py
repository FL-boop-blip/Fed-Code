from .utils_methods_FedAvg import train_FedAvg
from .utils_methods_FedProx import train_FedProx
from .utils_methods_Scaffold import train_SCAFFOLD
from .utils_methods_FedCM import train_FedCM
from .utils_methods_FedCVC import train_FedCVC, train_FedCVC_SGD
from .utils_methods_A_FedPD import train_A_FedPD, train_A_FedPD_SGD
from .utils_methods_FedAan import train_FedAdan
from .utils_methods_FedLESAM import train_FedLESAM_D, train_FedLESAM
from .utils_methods_RFL_DFDC import train_RFL_DFDC
from .utils_methods_FedDC import train_FedDC
from .utils_methods_FedDA2 import train_FedDA2
from .utils_methods_FedDyn import train_FedDyn
from .utils_methods_FedGamma import train_FedGamma
from .utils_methods_FedGKD import train_FedGKD
from .utils_methods_FedGloss import train_FedGloss
from .utils_methods_FedSMOO import train_FedSMOO
from .utils_methods_MoFedSAM import train_MoFedSAM
from .utils_methods_FedVARP import train_FedVARP
from .utils_methods_FedDisco import train_FedDisco
from .utils_methods_FedSpeed import train_FedSpeed


__all__ = [
    'train_FedAvg',
    'train_FedProx',
    'train_SCAFFOLD',
    'train_FedCM',
    'train_FedCVC',
    'train_FedCVC_SGD',
    'train_A_FedPD',
    'train_A_FedPD_SGD',
    'train_FedAdan',
    'train_FedLESAM_D',
    'train_FedLESAM',
    'train_RFL_DFDC',
    'train_FedDC',
    'train_FedDA2',
    'train_FedDyn',
    'train_FedGamma',
    'train_FedGKD',
    'train_FedGloss',
    'train_FedSMOO',
    'train_MoFedSAM',
    'train_FedVARP',
    'train_FedDisco',
    'train_FedSpeed',
]