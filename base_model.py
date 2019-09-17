import torch
import numpy as np
from torch.autograd import Variable
from metrics import mrr_mr_hitk
from utils import batch_by_size
import logging
import os
import time
from torch.optim import Adam, SGD, Adagrad
from models import TransDModule, TransEModule, TransHModule, DistMultModule, ComplExModule, SimplEModule

class BaseModel(object):
    def __init__(self, n_ent, n_rel, args):
        if args.model == 'TransE':
            self.model = TransEModule(n_ent, n_rel, args)
        elif args.model == 'TransD':
            self.model = TransDModule(n_ent, n_rel, args)
        elif args.model == 'TransH':
            self.model = TransHModule(n_ent, n_rel, args)
        elif args.model == 'DistMult':
            self.model = DistMultModule(n_ent, n_rel, args)
        elif args.model == 'ComplEx':
            self.model = ComplExModule(n_ent, n_rel, args)
        elif args.model == 'SimplE':
            self.model = SimplEModule(n_ent, n_rel, args)
        else:
            raise NotImplementedError

        self.model.cuda()

        self.n_ent = n_ent
        self.weight_decay = args.lamb * args.n_batch / args.n_train
        self.time_tot = 0
        self.args = args


    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))


    def remove_positive(self, remove=True):
        ''' this function removes false negative triplets in cache. '''
        length_h = len(self.head_pos)
        length_t = len(self.tail_pos)
        length = length_h + length_t
        self.count_pos = 0

        # use random variable to replace the false negative in cache
        def head_remove(arr):
            idx = arr[0]
            mark = np.isin(self.head_cache[idx], self.head_pos[idx])
            if remove == True:
                rand = np.random.choice(self.n_ent, size=(self.args.N_1,), replace=False)
                self.head_cache[idx][mark] = rand[mark]
            self.count_pos += np.sum(mark)

        def tail_remove(arr):
            idx = arr[0]
            mark = np.isin(self.tail_cache[idx], self.tail_pos[idx])
            if remove == True:
                rand = np.random.choice(self.n_ent, size=(self.args.N_1,), replace=False)
                self.tail_cache[idx][mark] = rand[mark]
            self.count_pos += np.sum(mark)

        head_idx = np.expand_dims(np.array(range(length_h), dtype='int'), 1)
        tail_idx = np.expand_dims(np.array(range(length_t), dtype='int'), 1)

        np.apply_along_axis(head_remove, 1, head_idx)
        np.apply_along_axis(tail_remove, 1, tail_idx)

        print("number of positives:", self.count_pos, self.count_pos/length)
        return self.count_pos / length
    

    def update_cache(self, head, tail, rela, head_idx, tail_idx):
        ''' update the cache with different schemes '''
        head_idx, head_uniq = np.unique(head_idx, return_index=True)
        tail_idx, tail_uniq = np.unique(tail_idx, return_index=True)

        tail_h = tail[head_uniq]
        rela_h = rela[head_uniq]

        rela_t = rela[tail_uniq]
        head_t = head[tail_uniq]

        # get candidate for updating the cache
        h_cache = self.head_cache[head_idx]
        t_cache = self.tail_cache[tail_idx]
        h_cand = np.concatenate([h_cache, np.random.choice(self.n_ent, (len(head_idx), self.args.N_2))], 1)
        t_cand = np.concatenate([t_cache, np.random.choice(self.n_ent, (len(tail_idx), self.args.N_2))], 1)
        h_cand = torch.from_numpy(h_cand).type(torch.LongTensor).cuda()
        t_cand = torch.from_numpy(t_cand).type(torch.LongTensor).cuda()

        # expand for computing scores/probs
        rela_h = rela_h.unsqueeze(1).expand(-1, self.args.N_1 + self.args.N_2)
        tail_h = tail_h.unsqueeze(1).expand(-1, self.args.N_1 + self.args.N_2)
        head_t = head_t.unsqueeze(1).expand(-1, self.args.N_1 + self.args.N_2)
        rela_t = rela_t.unsqueeze(1).expand(-1, self.args.N_1 + self.args.N_2)

        h_probs = self.model.prob(h_cand, tail_h, rela_h)
        t_probs = self.model.prob(head_t, t_cand, rela_t)

        if self.args.update == 'IS':
            h_new = torch.multinomial(h_probs, self.args.N_1, replacement=False)
            t_new = torch.multinomial(t_probs, self.args.N_1, replacement=False)
        elif self.args.update == 'top':
            _, h_new = torch.topk(h_probs,  k=self.args.N_1, dim=-1)
            _, t_new = torch.topk(t_probs,  k=self.args.N_1, dim=-1)

        h_idx = torch.arange(0, len(head_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.args.N_1)
        t_idx = torch.arange(0, len(tail_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.args.N_1)
        h_rep = h_cand[h_idx, h_new]
        t_rep = t_cand[t_idx, t_new]

        self.head_cache[head_idx] = h_rep.cpu().numpy()
        self.tail_cache[tail_idx] = t_rep.cpu().numpy()


    def neg_sample(self, head, tail, rela, head_idx, tail_idx, sample='basic', loss='pair'):
        '''
        negative sampling schems
        '''
        if sample == 'bern':    # Bernoulli
            n = head_idx.shape[0]
            h_idx = np.random.randint(low=0, high=self.n_ent, size=(n, self.args.n_sample))
            t_idx = np.random.randint(low=0, high=self.n_ent, size=(n, self.args.n_sample))

        elif sample == 'unif':     # NSCaching + uniform
            randint = np.random.randint(low=0, high=self.args.N_1, size=(head.shape[0],))
            h_idx = self.head_cache[head_idx, randint]
            t_idx = self.tail_cache[tail_idx, randint]

        elif sample == 'IS':        # NSCaching + IS
            n = head.size(0)
            h_cand = torch.from_numpy(self.head_cache[head_idx]).type(torch.LongTensor).cuda()
            t_cand = torch.from_numpy(self.tail_cache[tail_idx]).type(torch.LongTensor).cuda()

            head = head.unsqueeze(1).expand_as(h_cand)
            tail = tail.unsqueeze(1).expand_as(h_cand)
            rela = rela.unsqueeze(1).expand_as(h_cand)

            h_probs = self.model.prob(h_cand, tail, rela)
            t_probs = self.model.prob(head, t_cand, rela)
            h_new = torch.multinomial(h_probs, 1).squeeze()     # importance sampling
            t_new = torch.multinomial(t_probs, 1).squeeze()
            row_idx = torch.arange(0, n).type(torch.LongTensor)

            h_idx = h_cand[row_idx, h_new].cpu().numpy()
            t_idx = t_cand[row_idx, t_new].cpu().numpy()

        elif sample == 'top':       # NSCaching + top
            n = head.size(0)
            h_cand = torch.from_numpy(self.head_cache[head_idx]).type(torch.LongTensor).cuda()
            t_cand = torch.from_numpy(self.tail_cache[tail_idx]).type(torch.LongTensor).cuda()

            head = head.unsqueeze(1).expand_as(h_cand)
            tail = tail.unsqueeze(1).expand_as(h_cand)
            rela = rela.unsqueeze(1).expand_as(h_cand)

            h_scores = self.model.prob(h_cand, tail, rela)
            t_scores = self.model.prob(head, t_cand, rela)
            h_new = torch.argmax(h_scores,  dim=-1)     # sample the top-1
            t_new = torch.argmax(t_scores,  dim=-1)
            row_idx = torch.arange(0, n).type(torch.LongTensor)

            h_idx = h_cand[row_idx, h_new].cpu().numpy()
            t_idx = t_cand[row_idx, t_new].cpu().numpy()

        h_rand = torch.LongTensor(h_idx).cuda()
        t_rand = torch.LongTensor(t_idx).cuda()
        return h_rand, t_rand


    def train(self, train_data, caches, corrupter, tester_val, tester_tst):
        head, tail, rela = train_data
        # useful information related to cache
        head_idx, tail_idx, self.head_cache, self.tail_cache, self.head_pos, self.tail_pos = caches
        n_train = len(head)

        if self.args.optim=='adam' or self.args.optim=='Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        elif self.args.optim=='adagrad' or self.args.optim=='adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)

        n_epoch = self.args.n_epoch
        n_batch = self.args.n_batch
        best_mrr = 0

        for epoch in range(n_epoch):
            start = time.time()

            self.epoch = epoch
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx].cuda()
            tail = tail[rand_idx].cuda()
            rela = rela[rand_idx].cuda()
            head_idx = head_idx[rand_idx.numpy()]
            tail_idx = tail_idx[rand_idx.numpy()]
            epoch_loss = 0

            if self.args.save and epoch==self.args.s_epoch:
                self.save(os.path.join(self.args.task_dir, self.args.model + '.mdl'))

            for h, t, r, h_idx, t_idx in batch_by_size(n_batch, head, tail, rela, head_idx, tail_idx, n_sample=n_train):
                self.model.zero_grad()

                h_rand, t_rand = self.neg_sample(h, t, r, h_idx, t_idx, self.args.sample, self.args.loss)
              
                # Bernoulli sampling to select (h', r, t) and (h, r, t')
                prob = corrupter.bern_prob[r]
                selection = torch.bernoulli(prob).type(torch.ByteTensor)
                n_h = torch.LongTensor(h.cpu().numpy()).cuda()
                n_t = torch.LongTensor(t.cpu().numpy()).cuda()
                n_r = torch.LongTensor(r.cpu().numpy()).cuda()
                if n_h.size() != h_rand.size():
                    n_h = n_h.unsqueeze(1).expand_as(h_rand)
                    n_t = n_t.unsqueeze(1).expand_as(h_rand)
                    n_r = n_r.unsqueeze(1).expand_as(h_rand)
                    h = h.unsqueeze(1)
                    r = r.unsqueeze(1)
                    t = t.unsqueeze(1)
                    
                n_h[selection] = h_rand[selection]
                n_t[~selection] = t_rand[~selection]
                
                if not (self.args.sample=='bern'):
                    self.update_cache(h, t, r, h_idx, t_idx)

                if self.args.loss == 'point':
                    p_loss = self.model.point_loss(h, t, r, 1)
                    n_loss = self.model.point_loss(n_h, n_t, n_r, -1)
                    loss = p_loss + n_loss
                else:
                    loss = self.model.pair_loss(h, t, r, n_h, n_t)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.cpu().numpy()
            # get the time of each epoch
            self.time_tot += time.time() - start
            print("Epoch: %d/%d, Loss=%.8f, Time=%.4f"%(epoch+1, n_epoch, epoch_loss/n_train, time.time()-start))
           
            if self.args.remove:
                self.remove_positive(self.args.remove)
               
            if (epoch+1) % self.args.epoch_per_test == 0:
                # output performance 
                valid_mrr, valid_mr, valid_1, valid_3, valid_10 = tester_val()
                test_mrr,  test_mr,  test_1,  test_3,  test_10 =  tester_tst()
                out_str = '%d\t%.2f\t%.4f %.1f %.4f %.4f %.4f\t%.4f %.1f %.4f %.4f %.4f\n' % (epoch, self.time_tot, \
                        valid_mrr, valid_mr, valid_1, valid_3, valid_10, \
                        test_mrr, test_mr, test_1, test_3, test_10)
                with open(self.args.perf_file, 'a') as f:
                    f.write(out_str)

                # remove false negative 

                # output the best performance info
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    best_str = out_str
        return best_str

    def test_link(self, test_data, n_ent, heads, tails, filt=True):
        mrr_tot = 0.
        mr_tot = 0
        #hit10_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        for batch_h, batch_t, batch_r in batch_by_size(self.args.test_batch_size, *test_data):
            batch_size = batch_h.size(0)
            head_val = Variable(batch_h.unsqueeze(1).expand(batch_size, n_ent).cuda())
            tail_val = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
            rela_val = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
            all_val = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent).type(torch.LongTensor).cuda())
            batch_head_scores = self.model.score(all_val, tail_val, rela_val).data
            batch_tail_scores = self.model.score(head_val, all_val, rela_val).data
            # for each positive, compute its head scores and tail scores
            for h, t, r, head_score, tail_score in zip(batch_h, batch_t, batch_r, batch_head_scores, batch_tail_scores):
                h_idx = int(h.data.cpu().numpy())
                t_idx = int(t.data.cpu().numpy())
                r_idx = int(r.data.cpu().numpy())
                if filt:            # filtered setting
                    if tails[(h_idx,r_idx)]._nnz() > 1:
                        tmp = tail_score[t_idx].data.cpu().numpy()
                        idx = tails[(h_idx, r_idx)]._indices()
                        tail_score[idx] = 1e20
                        tail_score[t_idx] = torch.from_numpy(tmp).cuda()
                    if heads[(t_idx, r_idx)]._nnz() > 1:
                        tmp = head_score[h_idx].data.cpu().numpy()
                        idx = heads[(t_idx, r_idx)]._indices()
                        head_score[idx] = 1e20
                        head_score[h_idx] = torch.from_numpy(tmp).cuda()
                mrr, mr, hit = mrr_mr_hitk(tail_score, t_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                mrr, mr, hit = mrr_mr_hitk(head_score, h_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H=%f %f %f, Count=%d', float(mrr_tot)/count, float(mr_tot)/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count, count)
        return float(mrr_tot)/count, mr_tot/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count





