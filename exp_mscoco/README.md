# Multi-label Image Classification with MS-COCO


## Default Profile

Here is defalut profile of Variational Information Bottleneck (Hirakawa+, 2024), which is based on the default profile of Twoway Multi-label Loss proposed by Kobayashi, CVPR 2023.

### ResNet-50

```bash
# Twoway Multi-label Loss
python3 main.py --logdir ./runs_resnet50/resnet50_tml --model resnet50 --pretrained --use_nesterov --loss tml --lr 0.01 --b_lr 1.0 --gpu_id 0

# Asymmetric Loss
python3 main.py --logdir ./runs_resnet50/resnet50_asl --model resnet50 --pretrained --use_nesterov --loss asl --lr 2e-4 --b_lr 1.0 --gpu_id 0

# Binary Cross Entropy Loss
python3 main.py --logdir ./runs_resnet50/resnet50_bce --model resnet50 --pretrained --use_nesterov --loss bce --lr 0.1 --b_lr 1.0 --gpu_id 0

# Focal Loss
python3 main.py --logdir ./runs_resnet50/resnet50_fl --model resnet50 --pretrained --use_nesterov --loss fl --lr 0.1 --b_lr 1.0 --gpu_id 0
```

### ResNeXt-50_32x4d

```bash
# Twoway Multi-label Loss
python3 main.py --logdir ./runs_resnext50/resnext50_tml --model resnext50_32x4åd --pretrained --use_nesterov --loss tml --lr 0.01 --b_lr 1.0 --gpu_id 0

# Asymmetric Loss
python3 main.py --logdir ./runs_resnext50/resnext50_asl --model resnext50_32x4d --pretrained --use_nesterov --loss asl --lr 2e-4 --b_lr 1.0 --gpu_id 0

# Binary Cross Entropy Loss
python3 main.py --logdir ./runs_resnext50/resnext50_bce --model resnext50_32x4d --pretrained --use_nesterov --loss bce --lr 0.1 --b_lr 1.0 --gpu_id 0

# Focal Loss
python3 main.py --logdir ./runs_resnext50/resnext50_fl --model resnext50_32x4d --pretrained --use_nesterov --loss fl --lr 0.1 --b_lr 1.0 --gpu_id 0
```

### DenseNet-169

```bash
# Twoway Multi-label Loss
python3 main.py --logdir ./runs_densenet169/densenet169_tml --model densenet169 --pretrained --use_nesterov --loss tml --lr 0.01 --b_lr 1.0 --gpu_id 0

# Asymmetric Loss
python3 main.py --logdir ./runs_densenet169/densenet169_asl --model densenet169 --pretrained --use_nesterov --loss asl --lr 1e-05 --b_lr 1.0 --gpu_id 0

# Binary Cross Entropy Loss
python3 main.py --logdir ./runs_densenet169/densenet169_bce --model densenet169 --pretrained --use_nesterov --loss bce --lr 0.1 --b_lr 1.0 --gpu_id 0

# Focal Loss
python3 main.py --logdir ./runs_densenet169/densenet169_fl --model densenet169 --pretrained --use_nesterov --loss fl --lr 0.1 --b_lr 1.0 --gpu_id 0
```

### regnet_y_32gf

```bash
# Twoway Multi-label Loss
python3 main.py --logdir ./runs_regnet/regnet_tml --model regnet_y_32gf --pretrained --use_nesterov --loss tml --lr 0.01 --b_lr 1.0 --gpu_id 0,1

# Asymmetric Loss
python3 main.py --logdir ./runs_regnet/regnet_asl --model regnet_y_32gf --pretrained --use_nesterov --loss asl --lr 2e-4 --b_lr 1.0 --gpu_id 0,1

# Binary Cross Entropy Loss
python3 main.py --logdir ./runs_regnet/regnet_bce --model regnet_y_32gf --pretrained --use_nesterov --loss bce --lr 0.1 --b_lr 1.0 --gpu_id 0,1

# Focal Loss
python3 main.py --logdir ./runs_regnet/regnet_fl --model regnet_y_32gf --pretrained --use_nesterov --loss fl --lr 0.1 --b_lr 1.0 --gpu_id 0,1
```

## Twoway Multi-label Loss Profile

To examine with the profile of TML (Kobayashi+, CVPR2023), please change the following arguments.

* `b_lr`: `0.1`

## Asymmetric Loss Profile

To examine with the profile of Asymmetric Loss (Ridnic+, ICCV 2021), please change the following arguments.

* `epochs`: `60`
* `scheduler`: `onesycle`
