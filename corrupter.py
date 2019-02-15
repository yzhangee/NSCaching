import torch
from collections import defaultdict
import numpy as np


class BernCorrupter:
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = self.get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    def corrupt(self, head, tail, rela):
        prob = self.bern_prob[rela]
        selection = torch.bernoulli(prob).numpy().astype('int64')
        ent_random = np.random.choice(self.n_ent, len(head))
        head_out = (1 - selection) * head.numpy() + selection * ent_random
        tail_out = selection * tail.numpy() + (1 - selection) * ent_random
        return torch.from_numpy(head_out), torch.from_numpy(tail_out)


    def get_bern_prob(self, data, n_ent, n_rel):
        head, tail, rela = data
        edges = defaultdict(lambda: defaultdict(lambda: set()))
        rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
        for h, t, r in zip(head, tail, rela):
            edges[r][h].add(t)
            rev_edges[r][t].add(h)
        bern_prob = torch.zeros(n_rel)
        for k in edges.keys():
            right = sum(len(tails) for tails in edges[k].values()) / len(edges[k])
            left = sum(len(heads) for heads in rev_edges[k].values()) / len(rev_edges[k])
            bern_prob[k] = right / (right + left)
        return bern_prob


