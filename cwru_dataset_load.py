import os

import torch
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# For 12k Drive End Bearing Fault Data
DataName = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data",
            "48k Drive End Bearing Fault Data", "Normal Baseline Data"]
RpmName = ["1797rpm", "1772rpm", "1750rpm", "1730rpm"]
# Normal data
NormalMatName = ["97.mat", "98.mat", "99.mat", "100.mat"]
NormalName = ["Normal"]
# Fault data
FaultMatName_1797 = ["105.mat", "118.mat", "130.mat",
                     "169.mat", "185.mat", "197.mat",
                     "209.mat", "222.mat", "234.mat"]
FaultName_1797 = ["0.007-InnerRace", "0.007-Ball", "0.007-OuterRace6",
                  "0.014-InnerRace", "0.014-Ball", "0.014-OuterRace6",
                  "0.021-InnerRace", "0.021-Ball", "0.021-OuterRace6"]
# axis
axis = ["_DE_time", "_FE_time", "_BA_time"]
# label
label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
signal_size = 400
# ----------------------------------------------------------------------------------------------------
class CWRUDataset(Dataset):
    def __init__(self, rootpath, aim_data=None):
        """
        This class is used to load real CWRU dataset for GAN.
        :param rootpath: the root path of real CWRU data
        :param aim_data: choose a fault type to be generated
        """
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
        data = data.reshape(1, signal_size)
        label = self.label[item]
        return data, label

    def dataset_dic(self, rootpath) -> dict:
        data_dict = {}
        # load source normal data
        # ./Data/CaseWesternReserveUniversity/Normal Baseline Data/1797rpm/97.mat
        normalfile_path = rootpath + '/' + DataName[3] + '/' + RpmName[0] + '/' + NormalMatName[0]
        src_normal = self.data_load(normalfile_path, NormalMatName[0], axis[0])
        normal_data, normal_label = self.data_split(src_normal, target=0)
        data_dict.update({NormalName[0]: (normal_data, normal_label)})
        # load source fault data
        for i in tqdm(range(len(FaultMatName_1797))):
            faultfile_path = rootpath + '/' + DataName[0] + '/' + RpmName[0] + '/' + FaultMatName_1797[i]
            src_fault = self.data_load(faultfile_path, FaultMatName_1797[i], axis[0])
            fault_data, fault_label = self.data_split(src_fault, label[i])
            data_dict.update({FaultName_1797[i]: (fault_data, fault_label)})
        return data_dict

    @staticmethod
    def data_load(data_path, mat_name, axis_name):
        data_number = mat_name.split(".")  # mat文件序号
        # 指定加载mat文件中的通道
        if eval(data_number[0]) < 100:
            aim_axis = "X0" + data_number[0] + axis_name
        else:
            aim_axis = "X" + data_number[0] + axis_name
        # 选择通道加载指定mat文件
        src_data = loadmat(data_path)[aim_axis].flatten()
        return src_data

    @staticmethod
    def data_split(data_file, target):
        start = 0
        end = signal_size
        step = 200
        data_container = []
        target_container = []
        while end <= data_file.shape[0]:
            data_container.append(data_file[start:end].tolist())
            target_container.append(target)
            start += step
            end += step
            if len(data_container) >= 610:
                break
        # 缩放数据至[-1， 1]
        data_container = MaxAbsScaler().fit_transform(data_container)
        return data_container, target_container


class CWRU_MixedDataset(Dataset):
    """

    """
    def __init__(self, rootpath, model_path, imb_ratio=5, purpose='mixed', train=True, generator='DCGAN'):
        self.rootpath = rootpath  # 真实数据路径
        # \training process\generation\
        self.model_path = model_path + '/' + generator + '/'  # 生成模型存放路径
        self.imb_ratio = imb_ratio  # 数据不平衡比
        self.purpose = purpose  # 是否添加生成数据

        train_data_dict, test_data_dict = self.dataset_dict(self.rootpath)

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

    def dataset_dict(self, rootpath):
        if self.purpose == 'mixed':
            train_data_dict = {}
            test_data_dict = {}
            # load source normal data
            normalfile_path = rootpath + '/' + DataName[3] + \
                              '/' + RpmName[0] + '/' + NormalMatName[0]
            src_normal = self.data_load(normalfile_path, NormalMatName[0], axis[0])
            normal_data, normal_label = self.data_split(src_normal, target=0)
            normal_data_train, normal_data_test, normal_label_train, normal_label_test = \
                train_test_split(normal_data, normal_label, test_size=0.2, shuffle=False)
            train_data_dict.update({NormalName[0]: (normal_data_train, normal_label_train)})
            test_data_dict.update({NormalName[0]: (normal_data_test, normal_label_test)})
            # set fault data
            temp_num = len(train_data_dict[NormalName[0]][0]) // self.imb_ratio
            for i in tqdm(range(len(FaultMatName_1797)), desc='Mixed data loading'):
                faultfile_path = rootpath + '/' + DataName[0] + \
                                 '/' + RpmName[0] + '/' + FaultMatName_1797[i]
                src_fault = self.data_load(faultfile_path, FaultMatName_1797[i], axis[0])
                fault_data, fault_label = self.data_split(src_fault, label[i])
                fault_data_train, fault_data_test, fault_label_train, fault_label_test = \
                    train_test_split(fault_data, fault_label, test_size=0.2, shuffle=False)
                if FaultName_1797[i] not in os.listdir(self.model_path):
                    print("\033[4;31m{} Document Does not Exist!\033[0m".format(FaultName_1797[i]))
                    print("\033[4;31mPlease Check Generation Procedure!\033[0m")
                    break

                syn_num = len(train_data_dict[NormalName[0]][0]) - temp_num
                if syn_num != 0:
                    model_dir = self.model_path + FaultName_1797[i] + \
                                '/model/' + FaultName_1797[i] + '.pkl'
                    gen_model = torch.load(model_dir, map_location='cpu')
                    z = torch.randn(syn_num, 100)
                    gen_data = gen_model(z).detach().numpy().reshape(syn_num, -1)
                    gen_data_label = np.ones(syn_num, ) * label[i]
                    fault_data_train = np.concatenate((fault_data_train[:temp_num], gen_data))
                    fault_label_train = np.concatenate((fault_label_train[:temp_num], gen_data_label))
                    train_data_dict.update({FaultName_1797[i]: (fault_data_train, fault_label_train)})
                else:
                    train_data_dict.update({FaultName_1797[i]: (fault_data_train, fault_label_train)})
                test_data_dict.update({FaultName_1797[i]: (fault_data_test, fault_label_test)})

            return train_data_dict, test_data_dict

        else:
            # real data load
            real_data_dict = {}
            normalfile_path = rootpath + '/' + DataName[3] + \
                              '/' + RpmName[0] + '/' + NormalMatName[0]
            src_normal = self.data_load(normalfile_path, NormalMatName[0], axis[0])
            normal_data, normal_label = self.data_split(src_normal, target=0)
            real_data_dict.update({NormalName[0]: (normal_data, normal_label)})
            for i in tqdm(range(len(FaultMatName_1797)), desc='Real data loading'):
                faultfile_path = rootpath + '/' + DataName[0] + '/' + RpmName[0] + '/' + FaultMatName_1797[i]
                src_fault = self.data_load(faultfile_path, FaultMatName_1797[i], axis[0])
                fault_data, fault_label = self.data_split(src_fault, label[i])
                real_data_dict.update({FaultName_1797[i]: (fault_data, fault_label)})

            # synthetic data load
            syn_data_dict = {}
            syn_num = len(real_data_dict[NormalName[0]][0])
            # normal
            model_dir = self.model_path + NormalName[0] + \
                        '/model/' + NormalName[0] + '.pkl'
            gen_model = torch.load(model_dir, map_location='cpu')
            z = torch.randn(syn_num, 100)
            gen_data = gen_model(z).detach().numpy().reshape(syn_num, -1)
            gen_data_label = np.ones(syn_num, ) * 0
            syn_data_dict.update({NormalName[0]: (gen_data, gen_data_label)})
            for i in tqdm(range(len(FaultMatName_1797)), desc='Synthetic data loading'):
                model_dir = self.model_path + FaultName_1797[i] + \
                            '/model/' + FaultName_1797[i] + '.pkl'
                gen_model = torch.load(model_dir, map_location='cpu')
                z = torch.randn(syn_num, 100)
                gen_data = gen_model(z).detach().numpy().reshape(syn_num, -1)
                gen_data_label = np.ones(syn_num, ) * label[i]
                syn_data_dict.update({FaultName_1797[i]: (gen_data, gen_data_label)})

            if self.purpose == 'trts':
                return real_data_dict, syn_data_dict
            elif self.purpose == 'tstr':
                return syn_data_dict, real_data_dict
            else:
                print("\033[4;31mPlease Check Your purpose!\033[0m")
                raise NotImplementedError

    @staticmethod
    def data_load(data_path, mat_name, axis_name):
        data_number = mat_name.split(".")  # mat文件序号
        # 指定加载mat文件中的通道
        if eval(data_number[0]) < 100:
            aim_axis = "X0" + data_number[0] + axis_name
        else:
            aim_axis = "X" + data_number[0] + axis_name
        # 选择通道加载指定mat文件
        src_data = loadmat(data_path)[aim_axis].flatten()
        return src_data

    @staticmethod
    def data_split(data_file, target):
        start = 0
        end = signal_size
        step = 200
        data_container = []
        target_container = []
        while end <= data_file.shape[0]:
            data_container.append(data_file[start:end].tolist())
            target_container.append(target)
            start += step
            end += step
            if len(data_container) >= 610:
                break
        # 缩放数据至[-1， 1]
        data_container = MaxAbsScaler().fit_transform(data_container)
        return data_container, target_container


if __name__ == '__main__':
    # cwru = CWRUDataset(rootpath='../Data/CaseWesternReserveUniversity',
    #                    aim_data=FaultName_1797[0])
    # # 整理完成后 相对路径形式要改
    cwru_mixed_train = CWRU_MixedDataset(rootpath='./Data/CaseWesternReserveUniversity',
                                         model_path='./training process/generation/cwru',
                                         imb_ratio=1,
                                         purpose='trts', train=True, generator='FTGAN')


