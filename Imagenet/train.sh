# train
torchrun --nproc_per_node=4 train_resnet.py --model E_SResNet_S -b 128 --tb --amp --sync-bn --T 6
torchrun --nproc_per_node=4 train_resnet.py --model E_SResNet_M -b 128 --tb --amp --sync-bn --T 6
