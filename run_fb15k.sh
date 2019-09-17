#!/bin/bash
python train.py --task_dir=../KG_Data/FB15K --sample=bern --model=ComplEx --loss=point --save=True --s_epoch=500 --hidden_dim=100 --lamb=0.01 --lr=0.0003 --n_epoch=3000 --n_batch=4096 --filter=True --epoch_per_test=20 --test_batch_size=60 --optim=adam --out_file=_base;

python train.py --task_dir=../KG_Data/FB15K --sample=unif --model=ComplEx --loss=point --load=False --hidden_dim=100 --lamb=0.01 --lr=0.0003 --n_epoch=1000 --n_batch=4096 --filter=True --epoch_per_test=20 --test_batch_size=20 --optim=adam --out_file=_cache;

python train.py --task_dir=../KG_Data/FB15K --sample=bern --model=DistMult --loss=point --save=True --s_epoch=500 --hidden_dim=100 --lamb=0.01 --lr=0.001 --n_epoch=3000 --n_batch=4096 --filter=True --epoch_per_test=200 --test_batch_size=60 --optim=adam --out_file=_base;

python train.py --task_dir=../KG_Data/FB15K --sample=unif --model=DistMult --loss=point --load=False --hidden_dim=100 --lamb=0.01 --lr=0.001 --n_epoch=1000 --n_batch=4096 --filter=True --epoch_per_test=100 --test_batch_size=60 --optim=adam --out_file=_cache;

