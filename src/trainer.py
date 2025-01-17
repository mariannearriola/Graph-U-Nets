import torch
from tqdm import tqdm
import torch.optim as optim
from utils.dataset import GraphData
import torch.nn as nn
import random
import numpy as np
import time


class Trainer:
    def __init__(self, args, net, G_data):
        self.args = args
        self.net = net
        self.feat_dim = G_data.feat_dim
        self.fold_idx = G_data.fold_idx
        self.init(args, G_data.train_gs, G_data.test_gs)
        if torch.cuda.is_available():
            self.net.cuda()

    def init(self, args, train_gs, test_gs):
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))
        train_data = GraphData(train_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        self.train_d = train_data.loader(self.args.batch, True)
        self.test_d = test_data.loader(self.args.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, amsgrad=True,
            weight_decay=0.0008)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        return gs

    def run_epoch(self, epoch, data, model, optimizer, jaccWeight,jaccOppWeight):
        losses, accs, n_samples = [], [], 0
        grads = []
        grad_bank = {}
        avg_counter = 0
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys = batch
            gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
            loss, acc = model(gs, hs, ys, jaccWeight,jaccOppWeight)
            losses.append(loss*cur_len)
            accs.append(acc*cur_len)
            n_samples += cur_len
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        return avg_loss.item(), avg_acc.item(), losses

    def train(self):
        max_acc = 0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f'
        line_str = '%d:\t%.5f\n'
        bestAcc = 0
        jaccWeight = .5
        jaccOppWeight = .5
        loss = 0
        prevLosses = []
        prevloss = 0
        start_time = time.time()
        for e_id in range(self.args.num_epochs):
            print("for jaccWeight ", jaccWeight, " and ", jaccOppWeight)
            self.net.train()

            # train
            loss, acc, losses = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer,jaccWeight,jaccOppWeight)
            print(train_str % (e_id, loss, acc))

            # test
            with torch.no_grad():
                self.net.eval()
                loss, acc, losses = self.run_epoch(e_id, self.test_d, self.net, None, jaccWeight,jaccOppWeight)
            with open(self.args.acc_file, 'a+') as f:
                outline = "jaccWeight " + str(jaccWeight) + " jaccOppWeight " + str(jaccOppWeight)
                f.write(outline)
                f.write(line_str % (self.fold_idx, acc))
            
            # update jaccWeight
            idx = -1
            if len(losses) != 0 and e_id != 0:
                idx = random.randint(0,len(losses)-1)
                #grad = sig(losses[idx] - prevLosses[idx]).item()
                grad = (losses[idx] - prevLosses[idx]).item()
            else:
                grad = 0
            #grad = loss - prevloss
            print("grad: ", grad)
            #if idx != -1:
            #    print(" losses[idx]: ", losses[idx], " prevLosses[idx]: ", prevLosses[idx])
            mu, sigma = 0, 0.1 # mean and standard deviation
            s = np.random.normal(mu, sigma, 1)
            jaccWeight = jaccWeight - (0.01*grad) + s
            sig = nn.Sigmoid()
            jaccWeight = sig(torch.FloatTensor([jaccWeight])).item()
            jaccOppWeight = 1 - jaccWeight
            #jaccWeight, jaccOppWeight = torch.FloatTensor([jaccWeight]), torch.FloatTensor([jaccOppWeight]) 
            #jaccWeight, jaccOppWeight = sig(jaccWeight), sig(jaccOppWeight)
            #jaccWeight, jaccOppWeight = jaccWeight.tolist()[0], jaccOppWeight.tolist()[0]
            #jaccWeight, jaccOppWeight = jaccWeight.item(), jaccOppWeight.item()
            max_acc = max(max_acc, acc)
            prevloss = loss
            prevLosses = losses

            print(test_str % (e_id, loss, acc, max_acc))
        print("--- %s seconds ---" % (time.time() - start_time))
        with open(self.args.acc_file, 'a+') as f:
            f.write(line_str % (self.fold_idx, max_acc))
