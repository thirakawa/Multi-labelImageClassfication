# Multi-label Image Classification with MS-COCO


## Default Profile

### ResNet-50

```bash
# Binary Cross Entropy
python3 main.py --logdir ./runs_resnet50/bce --loss bce --gpu_id 0

# Focal Loss
python3 main.py --logdir ./runs_resnet50/fl --loss fl --gpu_id 0

# Asymmetric Loss
python3 main.py --logdir ./runs_resnet50/asl --loss asl --gpu_id 0

# Twoway Multi-label Loss
python3 main.py --logdir ./runs_resnet50/tml --loss tml --gpu_id 0
```
