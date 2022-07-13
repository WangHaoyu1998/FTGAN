import matplotlib.pyplot as plt
import numpy as np
import torch
from tftb.processing import PseudoWignerVilleDistribution, ShortTimeFourierTransform
from cwru_dataset_load import CWRUDataset
from mfpt_dataset_load import MFPTDataset
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gen_models = ['VanGan', 'DCGAN', 'FTGAN']

cwru = False
mfpt = not cwru


if cwru:
    van_dir = './training process/generation/cwru/' + gen_models[0] + '/'
    dc_dir = './training process/generation/cwru/' + gen_models[1] + '/'
    ft_dir = './training process/generation/cwru/' + gen_models[2] + '/'
    cwru_class_name = ["Normal", "0.007-InnerRace", "0.007-Ball", "0.007-OuterRace6",
                       "0.014-InnerRace", "0.014-Ball", "0.014-OuterRace6",
                       "0.021-InnerRace", "0.021-Ball", "0.021-OuterRace6"]
    c = cwru_class_name[9]
    van_model_dir = van_dir + c + '/model/' + c + '.pkl'
    dc_model_dir = dc_dir + c + '/model/' + c + '.pkl'
    ft_model_dir = ft_dir + c + '/model/' + c + '.pkl'
    model_dir_list = [van_model_dir, dc_model_dir, ft_model_dir]
    rand = torch.randint(64, size=[1])
    real_data = CWRUDataset(rootpath='./Data/CaseWesternReserveUniversity', aim_data=c).__getitem__(rand)
    pwvd = ShortTimeFourierTransform(real_data[0])
    pwvd.run()
    pwvd.plot(show=False, show_tf=True, kind='cmap', scale='log', cmap='viridis')
    plt.savefig('./spectrogram/cwru/' + c + '.svg', format='svg', dpi=1000)
    plt.close()
    for idx, model_dir in enumerate(model_dir_list):
        print(model_dir)
        z = torch.randn(64, 100).to(device)
        generator = torch.load(model_dir, map_location=device)
        gen_data = generator(z)

        pwvd = ShortTimeFourierTransform(gen_data[0].detach().numpy())
        pwvd.run()
        pwvd.plot(show=False, show_tf=True, kind='cmap', scale='log', cmap='viridis')
        plt.savefig('./spectrogram/cwru/' + c + '_' + gen_models[idx] + '.svg', format='svg', dpi=1000)
        plt.close()

elif mfpt:
    van_dir = './training process/generation/mfpt/' + gen_models[0] + '/'
    dc_dir = './training process/generation/mfpt/' + gen_models[1] + '/'
    ft_dir = './training process/generation/mfpt/' + gen_models[2] + '/'
    mfpt_class_name = ['Normal',
                       'InnerRaceFault_vload_1', 'InnerRaceFault_vload_2', 'InnerRaceFault_vload_3',
                       'OuterRaceFault_vload_1', 'OuterRaceFault_vload_2', 'OuterRaceFault_vload_3']
    c = mfpt_class_name[5]
    van_model_dir = van_dir + c + '/model/' + c + '.pkl'
    dc_model_dir = dc_dir + c + '/model/' + c + '.pkl'
    ft_model_dir = ft_dir + c + '/model/' + c + '.pkl'
    model_dir_list = [van_model_dir, dc_model_dir, ft_model_dir]
    rand = torch.randint(64, size=[1])
    real_data = MFPTDataset(rootpath='./Data/MachineryFailurePreventionTechnology', aim_data=c).__getitem__(rand)
    pwvd = ShortTimeFourierTransform(real_data[0])
    pwvd.run()
    pwvd.plot(show=False, show_tf=True, kind='cmap', scale='log', cmap='viridis')
    plt.savefig('./spectrogram/mfpt/' + c + '.svg', format='svg', dpi=1000)
    plt.close()
    for idx, model_dir in enumerate(model_dir_list):
        print(model_dir)
        z = torch.randn(64, 100).to(device)
        generator = torch.load(model_dir, map_location=device)
        gen_data = generator(z)

        pwvd = ShortTimeFourierTransform(gen_data[0].detach().numpy())
        pwvd.run()
        pwvd.plot(show=False, show_tf=True, kind='cmap', scale='log', cmap='viridis')
        plt.savefig('./spectrogram/mfpt/' + c + '_' + gen_models[idx] + '.svg', format='svg', dpi=1000)
        plt.close()



