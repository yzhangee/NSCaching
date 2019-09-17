import os
import torch
import numpy as np
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir, n_sample):
        self.inPath = task_dir
        self.n_sample = n_sample

        print("The toolkit is importing datasets.\n")
        with open(os.path.join(self.inPath, "relation2id.txt")) as f:
            tmp = f.readline()
            self.n_rel = int(tmp.strip())
            print("The total of relations is {}".format(self.n_rel))

        with open(os.path.join(self.inPath, "entity2id.txt")) as f:
            tmp = f.readline()
            self.n_ent = int(tmp.strip())
            print("The total of entities is {}".format(self.n_ent))

        self.train_head, self.train_tail, self.train_rela = self.read_data("train2id.txt")
        self.valid_head, self.valid_tail, self.valid_rela = self.read_data("valid2id.txt")
        self.test_head,  self.test_tail,  self.test_rela  = self.read_data("test2id.txt")

    def read_data(self, filename):
        allList = []
        head = []
        tail = []
        rela = []
        with open(os.path.join(self.inPath, filename)) as f:
            tmp = f.readline()
            total = int(tmp.strip())
            for i in range(total):
                tmp = f.readline()
                h, t, r = tmp.strip().split()
                h, t, r = int(h), int(t), int(r)
                allList.append((h, t, r))

        allList.sort(key=lambda l:(l[0], l[1], l[2]))

        head.append(allList[0][0])
        tail.append(allList[0][1])
        rela.append(allList[0][2])

        for i in range(1, total):
            if allList[i] != allList[i-1]:
                h, t, r = allList[i]
                head.append(h)
                tail.append(t)
                rela.append(r)
        return head, tail, rela

    def graph_size(self):
        return (self.n_ent, self.n_rel)

    def load_data(self, index):
        if index == 'train':
            return self.train_head, self.train_tail, self.train_rela
        elif index == 'valid':
            return self.valid_head, self.valid_tail, self.valid_rela
        else:
            return self.test_head,  self.test_tail,  self.test_rela

    def heads_tails(self):
        all_heads = self.train_head + self.valid_head + self.test_head
        all_tails = self.train_tail + self.valid_tail + self.test_tail
        all_relas = self.train_rela + self.valid_rela + self.test_rela

        heads = defaultdict(lambda: set())
        tails = defaultdict(lambda: set())
        for h, t, r in zip(all_heads, all_tails, all_relas):
            tails[(h, r)].add(t)
            heads[(t, r)].add(h)


        heads_sp = {}
        tails_sp = {}
        for k in heads.keys():
            heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                                   torch.ones(len(heads[k])), torch.Size([self.n_ent]))

        for k in tails.keys():
            tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                                   torch.ones(len(tails[k])), torch.Size([self.n_ent]))
        print("heads/tails size:", len(tails), len(heads))

        return heads_sp, tails_sp

    def get_cache_list(self):
        head_cache = {}
        tail_cache = {}
        head_pos = []
        tail_pos = []
        head_idx = []
        tail_idx = []
        count_h = 0
        count_t = 0
        for h, t, r in zip(self.train_head, self.train_tail, self.train_rela):
            if not (t,r) in head_cache:
                head_cache[(t,r)] = count_h
                head_pos.append([h])
                count_h += 1
            else:
                head_pos[head_cache[(t,r)]].append(h)

            if not (h,r) in tail_cache:
                tail_cache[(h,r)] = count_t
                tail_pos.append([t])
                count_t += 1
            else:
                tail_pos[tail_cache[(h,r)]].append(t)

            head_idx.append(head_cache[(t,r)])
            tail_idx.append(tail_cache[(h,r)])
        head_idx = np.array(head_idx, dtype=int)
        tail_idx = np.array(tail_idx, dtype=int)
        head_cache = np.random.randint(low=0, high=self.n_ent, size=(count_h, self.n_sample))
        tail_cache = np.random.randint(low=0, high=self.n_ent, size=(count_t, self.n_sample))
        print('head/tail_idx: head/tail_cache', len(head_idx), len(tail_idx), head_cache.shape, tail_cache.shape, len(head_pos), len(tail_pos))
        return head_idx, tail_idx, head_cache, tail_cache, head_pos, tail_pos



