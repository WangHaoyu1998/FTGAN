import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('-data_path', type=str, default='./Data/MachineryFailurePreventionTechnology',
                        help='the directory of the data')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch size of the training process')
    parser.add_argument('-beta1', type=int, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('-beta2', type=int, default=0.999,
                        help='beta2 for adam')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='the initial learning rate of gan')
    parser.add_argument('-epochs', type=int, default=501,
                        help='max number of epoch')
    parser.add_argument('-latent_dim', type=int, default=100,
                        help='the dimension of latent code')
    parser.add_argument('-critics', type=int, default=5,
                        help='max number of critics')

    parser.add_argument('-signal_length', type=int, default=400, help='the length of signal')

    arguments = parser.parse_args()
    return arguments


# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class VanGanCwru(object):
    def __init__(self, args, aim_data=None):
        self.args = args
        self.aim_data = aim_data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup(self):
        # load data
        from mfpt_dataset_load import MFPTDataset
        from mfpt_dataset_load import MFPTDataset
        mfpt_dataset = MFPTDataset(rootpath=self.args.data_path,
                                   aim_data=self.aim_data)
        self.train_loader = DataLoader(mfpt_dataset, batch_size=self.args.batch_size,
                                       shuffle=True)
        # load model and initialization
        import modules as modules
        self.gen = getattr(modules, 'Van_Gen')().to(self.device)
        self.dis = getattr(modules, 'Van_Dis')().to(self.device)
        # self.gen.apply(weights_init)
        # self.dis.apply(weights_init)
        # set optimizer
        self.gen_optim = optim.Adam(self.gen.parameters(),
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2))
        self.dis_optim = optim.Adam(self.dis.parameters(),
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2))
        # set loss function
        self.adv_loss = nn.BCELoss()

    def train(self):
        loss = {'g_loss': [], 'd_loss': []}
        d_loss, g_loss = None, None
        fixed_z = torch.randn(self.args.batch_size,
                              self.args.latent_dim).to(self.device)

        for epoch in range(self.args.epochs):
            for idx, (data, _) in enumerate(self.train_loader):
                data = data.to(torch.float32).to(self.device)
                # ground truth
                valid = torch.Tensor(data.shape[0], 1).fill_(1.).to(self.device)
                fake = torch.Tensor(data.shape[0], 1).fill_(0.).to(self.device)
                # --------------------
                #  Train Discriminator
                # --------------------
                self.dis_optim.zero_grad()
                z = torch.randn(data.shape[0], self.args.latent_dim).to(self.device)
                gen_data = self.gen(z).detach()
                fake_loss = self.adv_loss(self.dis(gen_data), fake)
                real_loss = self.adv_loss(self.dis(data), valid)
                d_loss = (real_loss + fake_loss) / 2.
                d_loss.backward()
                self.dis_optim.step()

                # -----------------
                #  Train Generator
                # -----------------
                for critics in range(1):
                    self.gen_optim.zero_grad()
                    valid = torch.Tensor(data.shape[0], 1).fill_(1.).to(self.device)
                    z = torch.randn(data.shape[0], self.args.latent_dim).to(self.device)
                    gen_data = self.gen(z)
                    g_loss = self.adv_loss(self.dis(gen_data), valid)
                    g_loss.backward()
                    self.gen_optim.step()

                    loss['d_loss'].append(d_loss.item())
                    loss['g_loss'].append(g_loss.item())

                print("[Epoch %d/%d] [Batch %d/%d] [D_real_loss: %f] [D_fake_loss: %f] [D_loss: %f] [G_loss: %f]"
                      % (epoch, self.args.epochs, idx, len(self.train_loader), real_loss.item(),
                         fake_loss.item(), d_loss.item(), g_loss.item()))

            if epoch % 10 == 0:
                with torch.no_grad():
                    from process_saving import ProcessSaving
                    save_dir = './training process/generation/mfpt/VanGAN/' + self.aim_data
                    self.gen.eval()
                    real_sig = next(iter(self.train_loader))[0]
                    gen_sig = self.gen(fixed_z)

                    proc = ProcessSaving(save_dir,
                                         real_data=real_sig, fake_data=gen_sig, epoch=epoch)
                    proc.loss_plot(loss)
                    proc.data_save()
                    proc.dist_plot()
                    proc.wave_plot()
                    proc.model_save(self.gen, self.aim_data)


class DcGanMfpt(object):
    def __init__(self, args, aim_data=None):
        self.args = args
        self.aim_data = aim_data
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self):
        # load data
        from mfpt_dataset_load import MFPTDataset
        mfpt_dataset = MFPTDataset(rootpath=self.args.data_path,
                                   aim_data=self.aim_data)
        self.train_loader = DataLoader(mfpt_dataset, batch_size=self.args.batch_size,
                                       shuffle=True)

        # load model
        import modules as modules
        self.gen = getattr(modules, 'DC_Gen')().to(self.device)
        self.dis = getattr(modules, 'DC_Dis')().to(self.device)
        # weight initialization
        self.gen.apply(weights_init)
        self.dis.apply(weights_init)
        # set optimizer
        self.gen_optim = optim.Adam(self.gen.parameters(),
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2))
        self.dis_optim = optim.Adam(self.dis.parameters(),
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2))
        # set loss function
        self.adv_loss = nn.BCELoss()

    def train(self):
        loss = {'g_loss': [], 'd_loss': []}
        d_loss, g_loss = None, None
        fixed_z = torch.randn(self.args.batch_size,
                              self.args.latent_dim).to(self.device)
        for epoch in range(self.args.epochs):
            for idx, (data, _) in enumerate(self.train_loader):
                data = data.to(torch.float32).to(self.device)
                # ground truth
                valid = torch.Tensor(data.shape[0], 1).fill_(1.).to(self.device)
                fake = torch.Tensor(data.shape[0], 1).fill_(0.).to(self.device)
                # --------------------
                #  Train Discriminator
                # --------------------
                self.dis_optim.zero_grad()
                z = torch.randn(data.shape[0], self.args.latent_dim).to(self.device)
                gen_data = self.gen(z).detach()
                fake_loss = self.adv_loss(self.dis(gen_data), fake)
                real_loss = self.adv_loss(self.dis(data), valid)
                d_loss = (real_loss + fake_loss) / 2.
                d_loss.backward()
                self.dis_optim.step()

                # -----------------
                #  Train Generator
                # -----------------
                for critics in range(self.args.critics):
                    self.gen_optim.zero_grad()
                    valid = torch.Tensor(data.shape[0], 1).fill_(1.).to(self.device)
                    z = torch.randn(data.shape[0], self.args.latent_dim).to(self.device)
                    gen_data = self.gen(z)
                    g_loss = self.adv_loss(self.dis(gen_data), valid)
                    g_loss.backward()
                    self.gen_optim.step()

                    loss['d_loss'].append(d_loss.item())
                    loss['g_loss'].append(g_loss.item())

                print("[Epoch %d/%d] [Batch %d/%d] [D_real_loss: %f] [D_fake_loss: %f] [D_loss: %f] [G_loss: %f]"
                      % (epoch, self.args.epochs, idx, len(self.train_loader), real_loss.item(),
                         fake_loss.item(), d_loss.item(), g_loss.item()))

            if epoch % 10 == 0:
                with torch.no_grad():
                    from process_saving import ProcessSaving
                    save_dir = './training process/generation/mfpt/DCGAN/' + self.aim_data
                    real_sig = next(iter(self.train_loader))[0]
                    gen_sig = self.gen(fixed_z)
                    proc = ProcessSaving(save_dir,
                                         real_data=real_sig, fake_data=gen_sig, epoch=epoch)
                    proc.loss_plot(loss)
                    proc.data_save()
                    proc.dist_plot()
                    proc.wave_plot()
                    proc.model_save(self.gen, self.aim_data)


class FtGanMfpt(object):
    def __init__(self,  args, aim_data=None):
        self.args = args
        self.aim_data = aim_data
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self):
        # load data
        from mfpt_dataset_load import MFPTDataset
        mfpt_dataset = MFPTDataset(rootpath=self.args.data_path,
                                   aim_data=self.aim_data)
        self.train_loader = DataLoader(mfpt_dataset, batch_size=self.args.batch_size,
                                       shuffle=True)
        # load model and initialization
        import modules as modules
        self.gen = getattr(modules, 'FT_Gen')().to(self.device)
        self.dis = getattr(modules, 'FT_Dis')().to(self.device)
        self.dis_r = getattr(modules, 'FT_Dis_r')().to(self.device)
        self.dis_i = getattr(modules, 'FT_Dis_i')().to(self.device)
        self.l_r = getattr(modules, 'layer_real')().to(self.device)
        self.l_i = getattr(modules, 'layer_imag')().to(self.device)
        # parameter initialization
        self.gen.apply(weights_init)
        self.dis.apply(weights_init)
        self.dis_r.apply(weights_init)
        self.dis_i.apply(weights_init)
        self.l_r.apply(weights_init)
        self.l_i.apply(weights_init)
        # set optimizer
        self.gen_optim = optim.Adam(self.gen.parameters(),
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2))
        self.dis_optim = optim.Adam(self.dis.parameters(),
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2))
        self.dis_r_optim = optim.Adam(self.dis_r.parameters(),
                                      lr=self.args.lr,
                                      betas=(self.args.beta1, self.args.beta2))
        self.dis_i_optim = optim.Adam(self.dis_i.parameters(),
                                      lr=self.args.lr,
                                      betas=(self.args.beta1, self.args.beta2))
        self.l_r_optim = optim.Adam(self.l_r.parameters(),
                                    lr=5e-3,
                                    betas=(self.args.beta1, self.args.beta2))
        self.l_i_optim = optim.Adam(self.l_i.parameters(),
                                    lr=5e-3,
                                    betas=(self.args.beta1, self.args.beta2))
        # set loss function
        self.fft_loss = nn.HuberLoss()
        self.adv_loss = nn.BCELoss()

        # ifft
        import numpy as np
        tvals = torch.from_numpy(np.arange(self.args.signal_length).reshape([-1, 1]))
        freqs = torch.from_numpy(np.arange(self.args.signal_length).reshape([1, -1]))
        self.arg_vals = 2 * np.pi * tvals * freqs / self.args.signal_length
        self.arg_vals = self.arg_vals.to(self.device)

    def train(self):
        loss = {'g_loss': [], 'g_real_loss': [], 'g_imag_loss': [],
                'd_loss': [], 'd_real_loss': [], 'd_imag_loss': [],
                'rec_loss': []}
        g_data_loss, d_data_loss, g_real_loss, d_real_loss, g_imag_loss, d_imag_loss = \
            None, None, None, None, None, None
        fixed_z = torch.randn(self.args.batch_size,
                              self.args.latent_dim).to(self.device)
        for epoch in range(self.args.epochs):
            for idx, (data, _) in enumerate(self.train_loader):
                data = data.to(torch.float32).to(self.device)
                # ---------------------
                # Train FFT layer
                # ---------------------
                self.l_r_optim.zero_grad()
                self.l_i_optim.zero_grad()
                x_real = self.l_r(data)
                x_imag = self.l_i(data)
                x_rec = (torch.matmul(x_real, torch.cos(self.arg_vals)) -
                         torch.matmul(x_imag, torch.sin(self.arg_vals))) / self.args.signal_length
                rec_loss = self.fft_loss(data, x_rec)
                loss['rec_loss'].append(rec_loss.item())
                rec_loss.backward()
                self.l_r_optim.step()
                self.l_i_optim.step()
                # ------------------------
                # Train Each Discriminator
                # ------------------------
                valid = torch.Tensor(data.shape[0], 1).fill_(1.).to(self.device)
                fake = torch.Tensor(data.shape[0], 1).fill_(0.).to(self.device)
                z = torch.randn(data.shape[0], self.args.latent_dim).to(self.device)
                x_gen = self.gen(z)
                # phase1
                self.dis_r_optim.zero_grad()
                real_real_loss = self.adv_loss(self.dis_r(self.l_r(data).detach()), valid)
                fake_real_loss = self.adv_loss(self.dis_r(self.l_r(x_gen).detach()), fake)
                d_real_loss = (real_real_loss + fake_real_loss) / 2
                d_real_loss.backward()
                self.dis_r_optim.step()
                # phase2
                self.dis_i_optim.zero_grad()
                real_imag_loss = self.adv_loss(self.dis_i(self.l_i(data).detach()), valid)
                fake_imag_loss = self.adv_loss(self.dis_i(self.l_i(x_gen).detach()), fake)
                d_imag_loss = (real_imag_loss + fake_imag_loss) / 2
                d_imag_loss.backward()
                self.dis_i_optim.step()
                # phase3
                self.dis.zero_grad()
                real_data_loss = self.adv_loss(self.dis(data), valid)
                fake_data_loss = self.adv_loss(self.dis(x_gen.detach()), fake)
                d_data_loss = (real_data_loss + fake_data_loss) / 2
                d_data_loss.backward()
                self.dis_optim.step()
                # ------------------------
                # Train Each Discriminator
                # ------------------------
                for critic in range(self.args.critics):
                    valid = torch.Tensor(data.shape[0], 1).fill_(1.).to(self.device)
                    z = torch.randn(data.shape[0], self.args.latent_dim).to(self.device)
                    x_gen = self.gen(z)
                    self.gen_optim.zero_grad()
                    # 计算Gx_loss
                    # phase 2
                    g_real_loss = self.adv_loss(self.dis_r(self.l_r(x_gen)), valid)
                    g_real_loss.backward(retain_graph=True)
                    # phase 3
                    g_imag_loss = self.adv_loss(self.dis_i(self.l_i(x_gen)), valid)
                    g_imag_loss.backward(retain_graph=True)
                    # phase 1
                    g_data_loss = self.adv_loss(self.dis(x_gen), valid)
                    g_data_loss.backward()
                    # 更新参数
                    self.gen_optim.step()
                    # ----------------end-------------------
                    loss['g_loss'].append(g_data_loss.item())
                    loss['g_real_loss'].append(g_real_loss.item())
                    loss['g_imag_loss'].append(g_imag_loss.item())
                    loss['d_loss'].append(d_data_loss.item())
                    loss['d_real_loss'].append(d_real_loss.item())
                    loss['d_imag_loss'].append(d_imag_loss.item())

                print("[Epoch %d/%d] [Batch %d/%d] [Rec_loss: %f] [Gx_loss: %f] [Dx_loss: %f] "
                      "[G_real_loss: %f] [D_real_loss: %f] [G_imag_loss:%f] [D_imag_loss: %f]"
                      % (epoch, self.args.epochs, idx, len(self.train_loader), rec_loss.item(),
                         g_data_loss.item(), d_data_loss.item(),
                         g_real_loss.item(), d_real_loss.item(),
                         g_imag_loss.item(), d_imag_loss.item()))

            if epoch % 10 == 0:
                from process_saving import ProcessSaving
                save_dir = './training process/generation/mfpt/FTGAN/' + self.aim_data
                self.gen.eval()
                real_sig = next(iter(self.train_loader))[0]
                gen_sig = self.gen(fixed_z)

                proc = ProcessSaving(save_dir,
                                     real_data=real_sig, fake_data=gen_sig, epoch=epoch)
                proc.loss_plot(loss)
                proc.data_save()
                proc.dist_plot()
                proc.wave_plot()
                proc.model_save(self.gen, self.aim_data)


if __name__ == '__main__':

    aim_datas = ['Normal',
                 'InnerRaceFault_vload_1', 'InnerRaceFault_vload_2', 'InnerRaceFault_vload_3',
                 'OuterRaceFault_vload_1', 'OuterRaceFault_vload_2', 'OuterRaceFault_vload_3']

    for aim_data in aim_datas:

        torch.cuda.empty_cache()
        van = VanGanCwru(args=parse_args(), aim_data=aim_data)
        van.setup()
        van.train()

        torch.cuda.empty_cache()
        dc = DcGanMfpt(args=parse_args(), aim_data=aim_data)
        dc.setup()
        dc.train()

        torch.cuda.empty_cache()
        ft = FtGanMfpt(args=parse_args(), aim_data=aim_data)
        ft.setup()
        ft.train()


