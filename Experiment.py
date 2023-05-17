from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from load_data import Data
from LorentzModel import HyperNet
from optim import RiemannianAdam, RiemannianSGD

class Experiment:
    def __init__(self,
                 data = None,
                 margin = 0.5,
                 noise_reg = 0.15,
                 learning_rate=1e-3,
                 dim=40,
                 nneg=50,
                 npos=10,
                 valid_steps=10,
                 num_epochs=500,
                 batch_size=128,
                 max_norm=0.5,
                 max_grad_norm=1,
                 optimizer='radam',
                 cuda=True,
                 early_stop=10,
                 real_neg=True,
                 device='cuda:0',
                 step_size=30,
                 gamma=0.6,
                 ):
        self.data = data
        self.learning_rate = learning_rate
        self.dim = dim
        self.npos = npos
        self.nneg = nneg
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.valid_steps = valid_steps
        self.cuda = cuda
        self.early_stop = early_stop
        self.real_neg = real_neg
        self.device = device
        self.margin = margin
        self.noise_reg = noise_reg
        self.step_size = step_size
        self.gamma = gamma
        self.entity_idxs = {data.entities[i]: i for i in range(len(data.entities))}
        self.relation_idxs = {data.relations[i]: i for i in range(len(data.relations))}
        self.relation_reverse_idxs = {}
        for kk, vv in self.relation_idxs.items():
            self.relation_reverse_idxs[vv] = kk

    def get_data_idxs(self, data):
        """ Return the training triplets
        """
        data_idxs = [
            (self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
             self.entity_idxs[data[i][2]]) for i in range(len(data))
        ]
        return data_idxs
        #   results = [t for t in data_idxs if t[1] in {3}]
        #   return results

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        """ Return the valid tail entities for (head, relation) pairs
        Can be used to guarantee that negative samples are true negative.
        """
        er_vocab = defaultdict(set)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].add(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, data, batch=100 ):
        d = self.data
        hits = []
        ranks = []
        rank_by_rela = {}
        hit_by_rela = {}
        for i in range(10):
            hits.append([])

        test_data_idxs = np.array(self.get_data_idxs(data))
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        tt = torch.Tensor(np.array(range(len(d.entities)),
                                   dtype=np.int64)).cuda().long().repeat(
            batch, 1) if self.cuda else torch.Tensor(np.array(range(len(d.entities)),
                                                              dtype=np.int64)).long().repeat(batch, 1)

        for i in range(0, len(test_data_idxs), batch):
            data_point = test_data_idxs[i:i + batch]
            e1_idx = torch.tensor(data_point[:, 0])
            r_idx = torch.tensor(data_point[:, 1])
            e2_idx = torch.tensor(data_point[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            #head batch
            predictions_s_h = model.forward( e1_idx, r_idx, tt[:min(batch, len(test_data_idxs) - i)]) #only use forward link
            #tail batch
            predictions_s_t = model.forward(tt[:min(batch, len(test_data_idxs) - i)], torch.where(r_idx%2==0, r_idx+1, r_idx-1), e1_idx ) #only use reverse link
            predictions_s = torch.stack([predictions_s_t, predictions_s_h], dim=-1)
            predictions_s = torch.mean(predictions_s, dim=-1)

            for j in range(min(batch, len(test_data_idxs) - i)):
                filt = list(sr_vocab[(data_point[j][0], data_point[j][1])])
                target_value = predictions_s[j][e2_idx[j]].item()
                predictions_s[j][filt] = -np.Inf
                predictions_s[j][e1_idx[j]] = -np.Inf
                predictions_s[j][e2_idx[j]] = target_value

                rank = (predictions_s[j] >= target_value).sum().item() - 1 # remove = from filter
                ranks.append(rank + 1)
                rank_by_rela.setdefault(data_point[j][1]-1 if data_point[j][1]%2==1 else data_point[j][1],
                                        []).append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                        hit_by_rela.setdefault(
                            data_point[j][1]-1 if data_point[j][1]%2==1 else data_point[j][1],
                            [[] for ii in range(10)])[hits_level].append(1.)
                    else:
                        hits[hits_level].append(0.0)
                        hit_by_rela.setdefault(
                            data_point[j][1]-1 if data_point[j][1]%2==1 else data_point[j][1],
                            [[] for ii in range(10)])[hits_level].append(0.)

        for keys, values in hit_by_rela.items():
            print(self.relation_reverse_idxs[keys], "->", np.mean(values[9]), np.mean(values[2]), np.mean(values[0]),
                  np.mean(1. / (np.array(rank_by_rela[keys]) + 1e-6)))
        return np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(1. / (np.array(ranks) + 1e-6))

    @property
    def train_and_eval(self):
        d = self.data
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        model = HyperNet(d, self.dim, self.max_norm, self.margin, self.nneg, self.npos, self.noise_reg)
        print("Training the %s model..." % model)
        if self.optimizer == 'radam':
            opt = RiemannianAdam(model.parameters(),
                                 lr=self.learning_rate,
                                 stabilize=1)
        elif self.optimizer == 'rsgd':
            opt = RiemannianSGD(model.parameters(),
                                lr=self.learning_rate,
                                stabilize=1)
        elif self.optimizer == 'adam':
            opt = Adam(model.parameters(),
                       lr=self.learning_rate)
        else:
            raise ValueError("Wrong optimizer")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt,
                                                    step_size=self.step_size, gamma=self.gamma,
                                                    verbose=True)

        if self.cuda:
            model.cuda()

        train_data_idxs_np = np.array(train_data_idxs)
        train_data_idxs = torch.tensor(np.array(train_data_idxs)).cuda() if self.cuda else torch.tensor(
            np.array(train_data_idxs))

        train_order = list(range(len(train_data_idxs)))

        targets = np.zeros((self.batch_size, self.nneg * self.npos + self.npos))
        targets[:, 0:self.npos] = 1
        targets = torch.FloatTensor(targets).cuda() if self.cuda else torch.FloatTensor(targets)
        max_mrr = 0.0
        max_it = 0
        mrr = 0
        bad_cnt = 0
        print("Starting training...")
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        bar = tqdm(range(1, self.num_epochs + 1),
                   desc='Best:%.3f@%d,curr:%.3f,loss:%.6f' %
                        (max_mrr, max_it, 0., 0.),
                   ncols=75)
        best_model = None
        for it in bar:
            # neg sampling tail
            model.train()
            losses = []
            np.random.shuffle(train_order)
            for j in range(0, len(train_data_idxs), self.batch_size * self.npos): # each batch has npos postives
                data_batch = train_data_idxs[train_order[j : j+self.batch_size*self.npos]]
                data_batch_np = train_data_idxs_np[train_order[j:j + self.batch_size*self.npos]]
                if j + self.batch_size*self.npos > len(train_data_idxs):
                    continue

                data_batch = data_batch.view(self.batch_size, -1, 3)
                data_batch_np = data_batch_np.reshape(self.batch_size, -1, 3)

                negsamples = np.random.randint(low=0,
                                               high=len(self.entity_idxs),
                                               size=(data_batch.size(0), self.npos, self.nneg),
                                               dtype=np.int32)
                if self.real_neg:
                    # Filter out the false negative samples.
                    candidate = np.random.randint(low=0,
                                                  high=len(self.entity_idxs),
                                                  size=(data_batch.size(0)),
                                                  dtype=np.int32)
                    e1_idx_np = data_batch_np[:, :, 0]
                    r_idx_np = data_batch_np[:, :, 1]
                    for index in range(negsamples.shape[0]):  # batch
                        for index2 in range(negsamples.shape[1]):  # relas
                            filt = sr_vocab[(e1_idx_np[index, index2], r_idx_np[index, index2])]
                            for index_ in range(negsamples.shape[2]):  # nnegs
                                p_candidate = 0
                                while negsamples[index, index2][index_] in filt:
                                    negsamples[index, index2][index_] = candidate[p_candidate]
                                    p_candidate += 1
                                    if p_candidate == len(candidate):
                                        candidate = np.random.randint(low=0, high=len(self.entity_idxs),
                                                                      size=(self.batch_size), )
                                        p_candidate = 0
                # [batch,npos, nneg]
                negsamples = torch.LongTensor(negsamples).cuda() if self.cuda else torch.LongTensor(negsamples)

                opt.zero_grad()
                #head flow
                e1_idx = data_batch[:,:, 0] #batch, # of npos
                r_idx = data_batch[:,:, 1] #batch, # of relas
                e2_idx = torch.cat([data_batch[:,:, 2:3], negsamples], dim=-1)  # [batch,npos, nneg+1]

                intervals = model.forward(e1_idx, r_idx, e2_idx)
                loss = model.loss(intervals,targets)

                e1_idx = data_batch[:,:, 0] #batch, # of npos
                r_idx = data_batch[:,:, 1] #batch, # of relas
                r_idx = torch.where(r_idx%2==0, r_idx+1, r_idx-1)
                e2_idx = torch.cat([data_batch[:,:, 2:3], negsamples], dim=-1)  # [batch,npos, nneg+1]
                intervals = model.forward(e2_idx, r_idx, e1_idx)
                loss += model.loss(intervals, targets)

                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_( model.parameters(),
                                                              max_norm=self.max_grad_norm, error_if_nonfinite=True)
                opt.step()
                losses.append(loss.item())

            bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.6f' %
                                (max_mrr, max_it, mrr, np.mean(losses)))
            scheduler.step()
            model.eval()
            with torch.no_grad():
                if not it % self.valid_steps:
                    hit10, hit3, hit1, mrr = self.evaluate(model, d.valid_data)
                    if mrr > max_mrr:
                        max_mrr = mrr
                        max_it = it
                        bad_cnt = 0
                        best_model = deepcopy(model.state_dict())
                    else:
                        bad_cnt += 1
                        if bad_cnt == self.early_stop:
                            break
                    bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.6f' %
                                        (max_mrr, max_it, mrr, loss.item()))
        with torch.no_grad():
            model.load_state_dict(best_model)
            model.eval()
            hit10, hit3, hit1, mrr = self.evaluate(model, d.test_data,)
        print(
            'Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' %
            (max_it, hit10, hit3, hit1, mrr))

        return mrr, hit1, hit3, hit10

