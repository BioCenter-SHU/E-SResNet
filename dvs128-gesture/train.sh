# train
python train_resnet.py --T 16 --tb --amp --weight-decay 0. --model E_SResNet_T --SE 2>&1 | tee logs/lif/E_SResNet_T_T16_wd0.log
python train_resnet.py --T 10 --tb --amp --weight-decay 0. --model E_SResNet_T --SE 2>&1 | tee logs/lif/E_SResNet_T_T10_wd0.log
