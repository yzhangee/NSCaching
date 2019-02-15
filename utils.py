import logging
import os
import datetime
import random

def logger_init(args):
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    if args.log_to_file:
        log_filename = os.path.join(args.log_dir, args.log_prefix+datetime.datetime.now().strftime("%m%d%H%M%S"))
        logging.getLogget().addHandler(logging.FileHandler(log_filename))

def plot_config(args):
    out_str = "\noptim:{} r:{} lamb:{}, d:{}, temp:{}, lr:{}, N_1:{}, N_2:{}\n".format(
            args.optim, args.margin, args.lamb, args.hidden_dim, args.temp, args.lr, args.N_1, args.N_2)
    with open(args.perf_file, 'a') as f:
        f.write(out_str)

def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(random.randint(0, i))
    for ls in lists:
        j = idx[i]
        ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    for i in range(n_batch):
        start = int(n_sample * i / n_batch)
        end = int(n_sample * (i+1) / n_batch)
        ret = [ls[start:end] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]
        

