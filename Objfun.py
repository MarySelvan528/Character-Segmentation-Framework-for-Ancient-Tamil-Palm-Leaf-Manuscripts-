import numpy as np

from Evaluation import evaluation
from Global_Vars import Global_Vars


def objfun(Soln):
    Data = Global_Vars.Data
    Dataset = Global_Vars.Data
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Images, results = UNET_Model(Dataset, './UNET/Dataset 1/', model_name='Unet_4000.h5', sol=sol)
            Eval = evaluation(Images, Target)
            Fitn[i] = 1 / (1 - Eval[4])
        return Fitn
    else:
        sol = Soln
        Images, results = UNET_Model(Dataset, './UNET/Dataset 1/', model_name='Unet_4000.h5', sol=sol)
        Eval = evaluation(Images, Target)
        Fitn = 1 / (1 - Eval[4])
        return Fitn
