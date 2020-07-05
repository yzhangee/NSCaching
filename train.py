import os 
import argparse
import torch
import warning
from corrupter import BernCorrupter
from read_data import DataLoader
from utils import logger_init, plot_config
from base_model import BaseModel

# TODO

parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
parser.add_argument('--task_dir', type=str, default='../KG_Data/FB15K', help='the directory to dataset')
parser.add_argument('--model', type=str, default='ComplEx',  help='scoring function, support [TransE, TransD, TransH, DistMult, ComplEx, SimplE]')
parser.add_argument('--sample', type=str, default='unif', help='sampling method from the cache')
parser.add_argument('--update', type=str, default='IS', help='cache update method')
parser.add_argument('--remove', type=bool, default=False, help='whether to remove false negative in cache periodically')
parser.add_argument('--loss', type=str, default='point', help='loss function, pair_loss or  point_loss')
parser.add_argument('--save', type=bool, default=False, help='whether save model')
parser.add_argument('--s_epoch', type=int, default=1000, help='which epoch should be saved, only work when save=True')
parser.add_argument('--load', type=bool, default=False, help='whether load from pretrain model')
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--margin', type=float, default=4.0, help='set margin value for pair loss')
parser.add_argument('--lamb', type=float, default=0.01, help='set weight decay value')
parser.add_argument('--hidden_dim', type=int, default=100, help='set embedding dimension')
parser.add_argument('--temp', type=float, default=2.0, help='set temporature value to avoid device trigger')
parser.add_argument('--gpu', type=str, default='0', help='set gpu #')
parser.add_argument('--p', type=int, default=1, help='set distance norm')
parser.add_argument('--lr', type=float, default=0.0001, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=4096, help='number of batch size')
parser.add_argument('--N_1', type=int, default=30, help='cache_size')
parser.add_argument('--N_2', type=int, default=30, help='random subset size')
parser.add_argument('--n_sample', type=int, default=1, help='number of negative samples')
parser.add_argument('--epoch_per_test', type=int, default=50, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--out_file_info', type=str, default='', help='extra string for the output file name')
parser.add_argument('--log_to_file', type=bool, default=False, help='log to file')
parser.add_argument('--log_dir', type=str, default='./log', help='log save dir')

parser.add_argument('--log_prefix', type=str, default='', help='log prefix')

args = parser.parse_args()

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "5"
    os.environ["MKL_NUM_THREADS"] = "5"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(5)
    warnings.filterwarnings("ignore", category=UserWarning)

    dataset = args.task_dir.split('/')[-1]
    directory = os.path.join('results', args.model)
    if not os.path.exists(directory):
        os.makedirs(directory)
   
    args.out_dir = directory
    args.perf_file = os.path.join(directory, '_'.join([dataset, args.sample, args.update]) + args.out_file_info + '.txt')
    args.stat_file = os.path.join(directory, '_'.join([dataset, args.sample, args.update]) + '.stat')
    print('output file name:', args.perf_file, args.stat_file)

    logger_init(args)

    task_dir = args.task_dir
    loader = DataLoader(task_dir, args.N_1)

    n_ent, n_rel = loader.graph_size()

    train_data = loader.load_data('train')
    valid_data = loader.load_data('valid')
    test_data  = loader.load_data('test')
    args.n_train = len(train_data[0])
    print("Number of train:{}, valid:{}, test:{}.".format(len(train_data[0]), len(valid_data[0]), len(test_data[0])))

    plot_config(args)

    heads, tails = loader.heads_tails()
    head_idx, tail_idx, head_cache, tail_cache, head_pos, tail_pos = loader.get_cache_list()
    caches = [head_idx, tail_idx, head_cache, tail_cache, head_pos, tail_pos]

    train_data = [torch.LongTensor(vec) for vec in train_data]
    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data  = [torch.LongTensor(vec) for vec in test_data]

    tester_val = lambda: model.test_link(valid_data, n_ent, heads, tails, args.filter)
    tester_tst = lambda: model.test_link(test_data, n_ent, heads, tails, args.filter)

    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    model = BaseModel(n_ent, n_rel, args)


    best_str = model.train(train_data, caches, corrupter, tester_val, tester_tst)
    with open(args.perf_file, 'a') as f:
        print('Training finished and best performance:', best_str)
        f.write('best_performance: '+best_str)


