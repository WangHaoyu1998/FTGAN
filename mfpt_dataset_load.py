import os
import torch
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


faultname = ['Normal',
             'InnerRaceFault_vload_1', 'InnerRaceFault_vload_2', 'InnerRaceFault_vload_3',
             'OuterRaceFault_vload_1', 'OuterRaceFault_vload_2', 'OuterRaceFault_vload_3']
label = [0, 1, 2, 3, 4, 5, 6]
signal_size = 400


class MFPTDataset(Dataset):
    def __init__(self, rootpath, aim_data=None):
        # path = './Data/MachineryFailurePreventionTechnology'
        self.rootpath = rootpath
        data_dict = self.dataset_dic(self.rootpath)
        if aim_data is not None:
            self.dataset = data_dict[aim_data][0]
            self.label = data_dict[aim_data][1]
        else:
            print("\033[4;31mPlease choose fault data type to be generated!\033[0m")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        data = data.reshape(-1, signal_size)
        label = self.label[item]
        return data, label

    def dataset_dic(self, rootpath) -> dict:
        data_dict = {}
        # load data
        for i in tqdm(range(len(faultname))):
            file_path = rootpath + '/' + faultname[i] + '.mat'
            org_fault_data = self.load_data(file_path, label[i])
            fault_data, fault_label = self.split_data(org_fault_data, label[i])
            data_dict.update({faultname[i]: (fault_data, fault_label)})
        return data_dict

    @staticmethod
    def split_data(data_file, label):
        """
        This function is mainly use to split original data based on given signal size.
        :param data_file: 数据列表
        :param label: 数据类别
        :return:
        """
        start = 0
        step = 200
        end = signal_size
        data_container = []
        label_container = []
        while end <= data_file.shape[0]:
            data_container.append(data_file[start:end].tolist())
            label_container.append(label)
            start += step
            end += step
            if len(data_container) >= 700:
                break
        # 缩放数据至[-1， 1]
        data_container = MaxAbsScaler().fit_transform(data_container)
        return data_container, label_container

    @staticmethod
    def load_data(datafile_path, label):
        if label == 0:
            datafile = (loadmat(datafile_path)["bearing"][0][0][1]).flatten()  # Take out the data
        else:
            datafile = (loadmat(datafile_path)["bearing"][0][0][2]).flatten()  # Take out the data
        return datafile


class MFPT_MixedDataset(Dataset):
    def __init__(self, rootpath, model_path, imb_ratio=5, purpose='mixed', train=True, generator='DCGAN'):
        self.rootpath = rootpath  # 真实数据路径
        # \training process\generation\mpft
        self.model_path = model_path + '/' + generator + '/'
        self.imb_ratio = imb_ratio
        self.purpose = purpose

        train_data_dict, test_data_dict = self.data_dict(self.rootpath)

        self.dataset = []
        self.label = []

        if train:
            for key in train_data_dict.keys():
                self.dataset.extend(train_data_dict[key][0])
                self.label.extend(train_data_dict[key][1])
        else:
            for key in test_data_dict.keys():
                self.dataset.extend(test_data_dict[key][0])
                self.label.extend(test_data_dict[key][1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        data = data.reshape(-1, signal_size)
        label = self.label[item]
        return data, label

    def data_dict(self, rootpath):
        if self.purpose == 'mixed':
            train_data_dict = {}
            test_data_dict = {}
            # load data
            for i in tqdm(range(len(faultname))):
                file_path = rootpath + '/' + faultname[i] + '.mat'
                org_fault_data = self.load_data(file_path, label[i])
                fault_data, fault_label = self.split_data(org_fault_data, label[i])
                if faultname[i] == 'Normal':
                    fault_data_train, fault_data_test, fault_label_train, fault_label_test = \
                        train_test_split(fault_data, fault_label, test_size=0.2, shuffle=False)
                    train_data_dict.update({faultname[i]: (fault_data_train, fault_label_train)})
                    test_data_dict.update({faultname[i]: (fault_data_test, fault_label_test)})
                else:
                    fault_data_train, fault_data_test, fault_label_train, fault_label_test = \
                        train_test_split(fault_data, fault_label, test_size=0.2, shuffle=False)
                    test_data_dict.update({faultname[i]: (fault_data_test, fault_label_test)})
                    # 计算是否需要合成样本
                    temp_num = len(train_data_dict[faultname[0]][0]) // self.imb_ratio
                    syn_num = len(train_data_dict[faultname[0]][0]) - temp_num
                    if syn_num != 0:
                        model_dir = self.model_path + faultname[i] + \
                                    '/model/' + faultname[i] + '.pkl'
                        z = torch.randn(syn_num, 100)
                        gen_model = torch.load(model_dir, map_location='cpu')
                        gen_data = gen_model(z).detach().numpy().reshape(syn_num, -1)
                        gen_data_label = np.ones(syn_num, ) * label[i]
                        fault_data_train = np.concatenate((fault_data_train[:temp_num], gen_data))
                        fault_label_train = np.concatenate((fault_label_train[:temp_num], gen_data_label))

                        train_data_dict.update({faultname[i]: (fault_data_train, fault_label_train)})
                    else:
                        train_data_dict.update({faultname[i]: (fault_data_train, fault_label_train)})
            return train_data_dict, test_data_dict

        else:
            # real data load
            real_data_dict = {}
            for i in tqdm(range(len(faultname)), desc='Real data loading'):
                file_path = rootpath + '/' + faultname[i] + '.mat'
                org_fault_data = self.load_data(file_path, label[i])
                fault_data, fault_label = self.split_data(org_fault_data, label[i])
                real_data_dict.update({faultname[i]: (fault_data, fault_label)})

            # synthetic data load
            syn_data_dict = {}
            syn_num = len(real_data_dict[faultname[0]][0])
            for i in tqdm(range(len(faultname)), desc='Synthetic data loading'):
                model_dir = self.model_path + faultname[i] + '/model/' + faultname[i] + '.pkl'
                gen_model = torch.load(model_dir, map_location='cpu')
                z = torch.randn(syn_num, 100)
                gen_data = gen_model(z).detach().numpy().reshape(syn_num, -1)
                gen_label = np.ones(syn_num, ) * label[i]

                syn_data_dict.update({faultname[i]: (gen_data, gen_label)})

            if self.purpose == 'trts':
                return real_data_dict, syn_data_dict

            elif self.purpose == 'tstr':
                return syn_data_dict, real_data_dict

            else:
                print("\033[4;31mPlease Check Your purpose!\033[0m")
                raise NotImplementedError

    @staticmethod
    def split_data(data_file, label):
        """
        This function is mainly use to split original data based on given signal size.
        :param data_file: 数据列表
        :param label: 数据类别
        :return:
        """
        start = 0
        step = 200
        end = signal_size
        data_container = []
        label_container = []
        while end <= data_file.shape[0]:
            data_container.append(data_file[start:end].tolist())
            label_container.append(label)
            start += step
            end += step
            if len(data_container) >= 700:
                break
        # 缩放数据至[-1， 1]
        data_container = MaxAbsScaler().fit_transform(data_container)
        return data_container, label_container

    @staticmethod
    def load_data(datafile_path, label):
        if label == 0:
            datafile = (loadmat(datafile_path)["bearing"][0][0][1]).flatten()  # Take out the data
        else:
            datafile = (loadmat(datafile_path)["bearing"][0][0][2]).flatten()  # Take out the data
        return datafile


if __name__ == '__main__':
    mfpt_mixed_data = MFPT_MixedDataset(rootpath='./Data/MachineryFailurePreventionTechnology',
                                        model_path='./training process/generation/mfpt',
                                        generator='FTGAN', train=True, purpose='trts')

