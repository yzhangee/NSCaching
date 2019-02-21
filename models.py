import torch
import torch.nn.functional as F
import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self, n_ent, n_rel, args):
        super(BaseModule, self).__init__()
        self.lamb = args.lamb
        self.p = args.p         # norm
        self.margin = args.margin
        self.temp = args.temp

    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)

    def score(self, head, tail, rela):
        raise NotImplementedError

    def dist(self, head, tail, rela):
        raise NotImplementedError

    def prob_logit(self, head, tail, rela):
        raise NotImplementedError

    def prob(self, head, tail, rela):
        return F.softmax(self.prob_logit(head, tail, rela), dim=-1)

    def pair_loss(self, head, tail, rela, n_head, n_tail):
        d_pos = self.dist(head, tail, rela)
        d_neg = self.dist(n_head, n_tail, rela)
        return torch.sum(F.relu(self.margin + d_pos - d_neg))

    def softmax_loss(self, head, tail, rela):
        rela = rela.unsqueeze(-1).expand_as(head)
        probs = self.prob_logit(head, tail, rela)
        n = probs.size(0)
        truth = torch.zeros(n).type(torch.LongTensor).cuda()
        truth_probs = torch.nn.LogSoftmax(-1)(probs)[torch.arange(0, n).type(torch.LongTensor).cuda(), truth]
        return -truth_probs

    def point_loss(self, head, tail, rela, label):
        softplus = torch.nn.Softplus().cuda()
        score = self.forward(head, tail, rela)
        score = torch.sum(softplus(-1*label*score))
        return score

    def sigmoid_loss(self, head,tail, rela, n_head, n_tail):
        logsigmoid = torch.nn.LogSigmoid().cuda()
        p_score = self.forward(head, tail, rela)
        n_score = self.forward(n_head, n_tail, rela.unsqueeze(1))
        p_score = torch.sum(logsigmoid(p_score))
        n_score = torch.sum(logsigmoid(-n_score))
        return - p_score - n_score

class TransEModule(BaseModule):
    def __init__(self, n_ent, n_rel, args):
        super(TransEModule, self).__init__(n_ent, n_rel, args)
        self.rel_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.ent_embed = nn.Embedding(n_ent, args.hidden_dim)
        self.init_weight()

    def forward(self, head, tail, rela):
        shape = head.size()
        head = head.contiguous().view(-1)
        tail = tail.contiguous().view(-1)
        rela = rela.contiguous().view(-1)
        head_embed = F.normalize(self.ent_embed(head),2,-1)
        tail_embed = F.normalize(self.ent_embed(tail),2,-1)
        rela_embed = self.rel_embed(rela)
        return torch.norm(tail_embed - head_embed - rela_embed, p=self.p, dim=-1).view(shape)

    def dist(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def score(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def prob_logit(self, head, tail, rela):
        return -self.forward(head, tail, rela) / self.temp


class TransDModule(BaseModule):
    def __init__(self, n_ent, n_rel, args):
        super(TransDModule, self).__init__(n_ent, n_rel, args)
        self.rel_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.ent_embed = nn.Embedding(n_ent, args.hidden_dim)
        self.proj_rel_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.proj_ent_embed = nn.Embedding(n_ent, args.hidden_dim)
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)

    def _transfer(self, e, t, r):
        return F.normalize(e + torch.sum(e*t, -1, True)*r, 2, -1)


    def forward(self, head, tail, rela):
        shape = head.size()
        head = head.contiguous().view(-1)
        tail = tail.contiguous().view(-1)
        rela = rela.contiguous().view(-1)
        head_e = F.normalize(self.ent_embed(head), 2, -1)
        tail_e = F.normalize(self.ent_embed(tail), 2, -1)
        rela_e = F.normalize(self.rel_embed(rela), 2, -1)

        head_t = self.proj_ent_embed(head)
        tail_t = self.proj_ent_embed(tail)
        rela_t = self.proj_rel_embed(rela)

        head_proj = self._transfer(head_e, head_t, rela_t)
        tail_proj = self._transfer(tail_e, tail_t, rela_t)
        return torch.norm(tail_proj - head_proj - rela_e, p=self.p, dim=-1).view(shape)

    def dist(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def score(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def prob_logit(self, head, tail, rela):
        return -self.forward(head, tail, rela) / self.temp


class TransHModule(BaseModule):
    def __init__(self, n_ent, n_rel, args):
        super(TransHModule, self).__init__()
        self.rel_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.ent_embed = nn.Embedding(n_ent, args.hidden_dim)
        self.proj_rel_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.init_weight()

    def forward(self, head, tail, rela):
        shape = head.size()
        head = head.contiguous().view(-1)
        tail = tail.contiguous().view(-1)
        rela = rela.contiguous().view(-1)
        head_embed = F.normalize(self.ent_embed(head), 2, -1)
        tail_embed = F.normalize(self.ent_embed(tail), 2, -1)
        rela_embed = F.normalize(self.rel_embed(rela), 2, -1)
        w_embed = F.normalize(self.proj_rel_embed(rela), 2, -1)
        head_proj = head_embed - torch.sum(w_embed * head_embed, dim=-1, keepdim=True) * w_embed
        tail_proj = tail_embed - torch.sum(w_embed * tail_embed, dim=-1, keepdim=True) * w_embed
        return torch.norm(tail_proj - head_proj - rela_embed, p=self.p, dim=-1).view(shape)

    def dist(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def score(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def prob_logit(self, head, tail, rela):
        return -self.forward(head, tail, rela) / self.temp

class DistMultModule(BaseModule):
    def __init__(self, n_ent, n_rel, args):
        super(DistMultModule, self).__init__(n_ent, n_rel, args)
        self.ent_embed = nn.Embedding(n_ent, args.hidden_dim)
        self.rel_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.init_weight()

    def forward(self, head, tail, rela):
        shapes = head.size()
        head = head.contiguous().view(-1)
        tail = tail.contiguous().view(-1)
        rela = rela.contiguous().view(-1)

        head_embed = self.ent_embed(head)
        tail_embed = self.ent_embed(tail)
        rela_embed = self.rel_embed(rela)
        return torch.sum(tail_embed * head_embed * rela_embed, dim=-1).view(shapes)

    def dist(self, head, tail, rela):
        return -self.forward(head, tail, rela)

    def score(self, head, tail, rela):
        return -self.forward(head, tail, rela)

    def prob_logit(self, head, tail, rela):
        return self.forward(head, tail, rela)/self.temp


class ComplExModule(BaseModule):
    def __init__(self, n_ent, n_rel, args):
        super(ComplExModule, self).__init__(n_ent, n_rel, args)

        self.ent_re_embed = nn.Embedding(n_ent, args.hidden_dim)
        self.ent_im_embed = nn.Embedding(n_ent, args.hidden_dim)

        self.rel_re_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.rel_im_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.init_weight()

    def forward(self, head, tail, rela):
        shapes = head.size()
        head = head.contiguous().view(-1)
        tail = tail.contiguous().view(-1)
        rela = rela.contiguous().view(-1)

        head_re_embed = self.ent_re_embed(head)
        tail_re_embed = self.ent_re_embed(tail)
        rela_re_embed = self.rel_re_embed(rela)
        head_im_embed = self.ent_im_embed(head)
        tail_im_embed = self.ent_im_embed(tail)
        rela_im_embed = self.rel_im_embed(rela)

        score = torch.sum(rela_re_embed * head_re_embed * tail_re_embed, dim=-1) \
                + torch.sum(rela_re_embed * head_im_embed * tail_im_embed, dim=-1) \
                + torch.sum(rela_im_embed * head_re_embed * tail_im_embed, dim=-1) \
                - torch.sum(rela_im_embed * head_im_embed * tail_re_embed, dim=-1)
        return score.view(shapes)

    def dist(self, head, tail, rela):
        return -self.forward(head, tail, rela)

    def score(self, head, tail, rela):
        return -self.forward(head, tail, rela)

    def prob_logit(self, head, tail, rela):
        return self.forward(head, tail, rela)/self.temp

class SimplEModule(BaseModule):
    def __init__(self, n_ent, n_rel, args):
        super(SimplEModule, self).__init__(n_ent, n_rel, args)

        self.head_embed = nn.Embedding(n_ent, args.hidden_dim)
        self.tail_embed = nn.Embedding(n_ent, args.hidden_dim)

        self.rela_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.rela_inv_embed = nn.Embedding(n_rel, args.hidden_dim)
        self.init_weight()

    def forward(self, head, tail, rela):
        shapes = head.size()
        head = head.contiguous().view(-1)
        tail = tail.contiguous().view(-1)
        rela = rela.contiguous().view(-1)

        head_embed = self.head_embed(head)
        tail_embed = self.tail_embed(tail)
        rela_embed = self.rela_embed(rela)
        head_inv_embed = self.tail_embed(head)
        tail_inv_embed = self.head_embed(tail)
        rela_inv_embed = self.rela_inv_embed(rela)

        score = torch.sum(head_embed * rela_embed * tail_embed, dim=-1) \
                + torch.sum(head_inv_embed * rela_inv_embed * tail_inv_embed, dim=-1)
        return score.view(shapes)

    def dist(self, head, tail, rela):
        return -self.forward(head, tail, rela)

    def score(self, head, tail, rela):
        return -self.forward(head, tail, rela)

    def prob_logit(self, head, tail, rela):
        return self.forward(head, tail, rela)/self.temp

