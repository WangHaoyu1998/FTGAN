import matplotlib.pyplot as plt
import os
import torch
import numpy as np


class ProcessSaving(object):
    def __init__(self, save_dir, real_data, fake_data, epoch):
        self.save_dir = save_dir

        if torch.cuda.is_available():
            self.real_data = real_data.cpu()
            self.fake_data = fake_data.cpu()
        else:
            self.real_data = real_data
            self.fake_data = fake_data
        self.real_data = self.real_data.detach().numpy()
        self.fake_data = self.fake_data.detach().numpy()
        self.epoch = epoch

    def loss_plot(self, loss: dict):
        """
        This method is used to plot loss of generator and discriminator
        """
        fig = plt.figure()
        axis = fig.subplots()
        for key in loss.keys():
            axis.plot(loss[key], label=key, linewidth=0.5)
            # axis.plot(loss['d_loss'], label='Dis', linewidth=0.5)
        axis.legend()
        axis.set_xlabel('Loss Value')
        axis.set_ylabel('Step')
        axis.set_title('Loss Curve')
        if os.path.exists(self.save_dir + '/loss') is not True:
            os.makedirs(self.save_dir + '/loss')
        plt.savefig(self.save_dir + '/loss/gan_loss.svg',
                    format='svg', dpi=1000)
        plt.savefig(self.save_dir + '/loss/gan_loss.png',
                    format='png', dpi=1000)
        plt.close()

    def wave_plot(self):
        """
        This method is used to plot real and fake signal wave.
        """
        rows = 3
        columns = 3
        fig = plt.figure()
        axis = fig.subplots(rows, columns)
        for r in range(rows):
            for c in range(columns):
                index = torch.randint(low=0, high=self.fake_data.shape[0], size=(1,))
                axis[r, c].plot(self.real_data[index.item()].flatten(),
                                label='real', linewidth=0.25)
                axis[r, c].plot(self.fake_data[index.item()].flatten(),
                                label='fake', linewidth=0.25)
                axis[r, c].set_xticks(np.arange(0, 401, 200))
                axis[r, c].set_xlabel('Sample')
                axis[r, c].set_ylim(-1, 1)
                axis[r, c].set_ylabel('Magnitude')
                axis[r, c].set_title('Signal Wave')
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper right')
        fig.tight_layout()

        # 保存
        if os.path.exists(self.save_dir + '/wave') is not True:
            os.makedirs(self.save_dir + '/wave')
        plt.savefig(self.save_dir + '/wave/epoch_%d.svg' % self.epoch,
                    format='svg', dpi=1000)
        plt.savefig(self.save_dir + '/wave/epoch_%d.png' % self.epoch,
                    format='png', dpi=1000)
        plt.close()

    def dist_plot(self):
        """
        This function is used to plot distribution of real and fake data based on kernel density estimation.
        """
        import seaborn as sns
        # scipy.stats
        fig = plt.figure()
        axis = fig.subplots()
        sns.kdeplot(self.real_data.flatten(), label='real')
        sns.kdeplot(self.fake_data.flatten(), label='fake')
        axis.legend()
        axis.set_title('The PDF of real and fake data')
        # 保存
        if os.path.exists(self.save_dir + '/dist') is not True:
            os.makedirs(self.save_dir + '/dist')
        plt.savefig(self.save_dir + '/dist/epoch_%d.png' % self.epoch,
                    format='png', dpi=1000)
        plt.savefig(self.save_dir + '/dist/epoch_%d.svg' % self.epoch,
                    format='svg', dpi=1000)
        plt.close()

    def data_save(self):
        import numpy as np
        if os.path.exists(self.save_dir + '/data') is not True:
            os.makedirs(self.save_dir + '/data')
        np.savetxt(self.save_dir + '/data/epoch_%d.csv' % self.epoch,
                   self.fake_data.reshape(-1, 400),
                   delimiter=',')

    def model_save(self, generator, name):
        if os.path.exists(self.save_dir + '/model') is not True:
            os.makedirs(self.save_dir + '/model')
        torch.save(generator,
                   self.save_dir + '/model/' + name + '.pkl')


class DiagnosisResults(object):
    def __init__(self, save_dir):
        # training process/diagnosis/cwru
        self.save_dir = save_dir
        self.softmax = torch.nn.Softmax(dim=1)

    def plot_loss_acc(self, cost: dict, acc: dict):
        if os.path.exists(self.save_dir + '/loss') is not True:
            os.makedirs(self.save_dir + '/loss')

        plt.plot(cost['train'], label='train')
        plt.plot(cost['val'], label='test')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig(self.save_dir + '/loss/' + 'loss.png', format='png', dpi=1000)
        plt.savefig(self.save_dir + '/loss/' + 'loss.svg', format='svg', dpi=1000)
        plt.close()

        plt.plot(acc['train'], label='train')
        plt.plot(acc['val'], label='test')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig(self.save_dir + '/loss/' + 'accuracy.png', format='png', dpi=1000)
        plt.savefig(self.save_dir + '/loss/' + 'accuracy.svg', format='svg', dpi=1000)
        plt.close()


    def plot_roc(self, y_true, y_pred, epoch):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        if torch.cuda.is_available():
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()

        y_true = y_true.detach().numpy()
        y_pred = self.softmax(y_pred).detach().numpy()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Binarize the output
        y_true = label_binarize(y_true, classes=[x for x in range(y_pred.shape[1])])
        n_classes = y_true.shape[1]

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        # 保存
        if os.path.exists(self.save_dir + '/roc') is not True:
            os.makedirs(self.save_dir + '/roc')
        plt.savefig(self.save_dir + '/roc/roc_epoch_%d.png' % epoch, format='png', dpi=1000)
        plt.savefig(self.save_dir + '/roc/roc_epoch_%d.svg' % epoch, format='svg', dpi=1000)

        plt.close()

    def plot_cm(self, y_true, y_pred, epoch):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        if torch.cuda.is_available():
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()

        cm = confusion_matrix(y_true.detach().numpy(),
                              self.softmax(y_pred).argmax(dim=1).detach().numpy())
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_disp.plot()
        # 保存
        if os.path.exists(self.save_dir + '/cm') is not True:
            os.makedirs(self.save_dir + '/cm')
        plt.savefig(self.save_dir + '/cm/cm_epoch_%d.png' % epoch, format='png', dpi=1000)
        plt.savefig(self.save_dir + '/cm/cm_epoch_%d.svg' % epoch, format='svg', dpi=1000)
        plt.close()

    def print_report(self, y_true, y_pred):
        from sklearn.metrics import classification_report
        if torch.cuda.is_available():
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()
        # class_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # class_names = ["Normal",
        #              "0.007-InnerRace", "0.007-Ball", "0.007-OuterRace6",
        #              "0.014-InnerRace", "0.014-Ball", "0.014-OuterRace6",
        #              "0.021-InnerRace", "0.021-Ball", "0.021-OuterRace6"]
        class_numbers = [0, 1, 2, 3, 4, 5, 6]
        class_names = ['Normal',
                       'InnerRaceFault_vload_1', 'InnerRaceFault_vload_2', 'InnerRaceFault_vload_3',
                       'OuterRaceFault_vload_1', 'OuterRaceFault_vload_2', 'OuterRaceFault_vload_3']

        print(classification_report(y_true.detach().numpy(),
                                    self.softmax(y_pred).argmax(dim=1).detach().numpy(),
                                    labels=class_numbers, target_names=class_names))

