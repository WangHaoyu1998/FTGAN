import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-device', type=str,
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    parser.add_argument('-data_path', type=str, default='./Data/MachineryFailurePreventionTechnology',
                        help='the directory of the data')
    parser.add_argument('-model_path', type=str, default='./training process/generation/mfpt',
                        help='the directory of generator model')
    parser.add_argument('-batch_size', type=int, default=64,
                        help='batch size of the training process')
    parser.add_argument('-cnn_lr', type=float, default=0.01,
                        help='the initial learning rate of diagnosis model')
    parser.add_argument('-momentum', type=float, default=0.9,
                        help='the momentum for sgd')
    parser.add_argument('-cnn_epochs', type=int, default=101,
                        help='max number of epoch of diagnosis')

    arguments = parser.parse_args()
    return arguments


class MFPT_Diag(object):
    def __init__(self, args, imb_ratio=1, gen_model='DCGAN', cnn_model='lenet', purpose='trts'):
        self.args = args
        self.imb_ratio = imb_ratio
        self.gen_model = gen_model
        self.cnn_model = cnn_model
        self.purpose = purpose

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setup(self):
        from mfpt_dataset_load import MFPT_MixedDataset
        # make dataloader
        self.train_set = MFPT_MixedDataset(rootpath=self.args.data_path, model_path=self.args.model_path,
                                           imb_ratio=self.imb_ratio, train=True,
                                           purpose=self.purpose, generator=self.gen_model)
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.test_set = MFPT_MixedDataset(rootpath=self.args.data_path, model_path=self.args.model_path,
                                          imb_ratio=self.imb_ratio, train=False,
                                          purpose=self.purpose, generator=self.gen_model)
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=len(self.test_set), shuffle=True)
        # diagnosis model
        import modules as modules
        self.diag_model = getattr(modules, self.cnn_model)\
            (in_channel=1, out_channel=7).to(self.device)
        # optimizer
        self.diag_optim = torch.optim.SGD(self.diag_model.parameters(), lr=self.args.cnn_lr,
                                          momentum=self.args.momentum)
        # loss
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        from process_saving import DiagnosisResults
        dr = DiagnosisResults(save_dir='./training process/diagnosis/mfpt/'
                                       + self.gen_model + '/' + self.purpose + '/' + self.cnn_model)
        cost = {'train': [], 'val': []}
        accuracy = {'train': [], 'val': []}

        for epoch in range(self.args.cnn_epochs):
            train_acc, train_loss = 0.0, 0.0
            test_acc, test_loss = 0.0, 0.0
            for idx, (data, labels) in enumerate(self.train_loader):
                data = data.to(torch.float32).to(self.device)
                labels = labels.to(torch.long).to(self.device)

                self.diag_model.train()
                logits = self.diag_model(data)
                batch_loss = self.criterion(logits, labels)
                self.diag_optim.zero_grad()
                batch_loss.backward()
                self.diag_optim.step()

                train_loss += batch_loss.item() * data.size(0)
                train_acc += accuracy_score(labels, logits.argmax(dim=1)) * data.size(0)
                # train_acc += accuracy_score(labels.cpu(), logits.argmax(dim=1).cpu()) * data.size(0)

            train_loss = train_loss / len(self.train_set)
            train_acc = train_acc / len(self.train_set)
            cost['train'].append(train_loss)
            accuracy['train'].append(train_acc)
            print('Train Set: Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, train_loss, train_acc))

            if epoch % 10 == 0:
                self.diag_model.eval()
                for test_data, test_labels in self.test_loader:
                    test_data = test_data.to(torch.float32).to(self.device)
                    test_labels = test_labels.to(torch.long).to(self.device)
                    #
                    test_logits = self.diag_model(test_data)
                    #
                    batch_test_loss = self.criterion(test_logits, test_labels)
                    #
                    test_loss += batch_test_loss.item() * test_data.size(0)
                    test_acc += accuracy_score(test_labels, test_logits.argmax(dim=1)) * test_data.size(0)
                    # test_acc += accuracy_score(test_labels.cpu(), test_logits.argmax(dim=1).cpu()) * test_data.size(0)
                    # dr.print_report(y_true=test_labels, y_pred=test_logits)
                    dr.plot_roc(test_labels, test_logits, epoch)
                    dr.plot_cm(test_labels, test_logits, epoch)

                test_loss = test_loss / len(self.test_set)
                test_acc = test_acc / len(self.test_set)
                cost['val'].append(test_loss)
                accuracy['val'].append(test_acc)
                dr.plot_loss_acc(cost, accuracy)
                print('Test Set: Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, test_loss, test_acc))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parse_args()
    purposes = ['trts', 'tstr']
    gen_models = ['VanGan', 'DCGAN', 'FTGAN']
    cnn_models = ['lenet', 'alexnet', 'cnn']

    for gen_model in gen_models:
        for cnn_model in cnn_models:
            for purpose in purposes:
                torch.cuda.empty_cache()
                diag = MFPT_Diag(args, gen_model=gen_model,
                                 cnn_model=cnn_model, purpose=purpose)
                diag.setup()
                diag.train()

    # purposes = ['mixed']
    # imb_ratios = [5, 10, 20, 50]
    # gen_models = ['VanGAN', 'DCGAN', 'FTGAN']
    # cnn_models = ['lenet', 'alexnet', 'cnn']
    # args = parse_args()
    # torch.cuda.empty_cache()
    # diag = MFPT_Diag(args, gen_model=gen_models[2], imb_ratio=imb_ratios[3],
    #                  cnn_model=cnn_models[1], purpose=purposes[0])
    # diag.setup()
    # diag.train()

