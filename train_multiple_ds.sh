
dset=dtd
python train.py --dataset-config-file configs/datasets/$dset.yaml

dset=ucf101
python train.py --dataset-config-file configs/datasets/$dset.yaml

dset=sun397
python train.py --dataset-config-file configs/datasets/$dset.yaml

# dset=caltech101
# python train.py --dataset-config-file configs/datasets/$dset.yaml   

# dset=resisc45
# python train.py --dataset-config-file configs/datasets/$dset.yaml

# dset=oxford_flowers
# python train.py --dataset-config-file configs/datasets/$dset.yaml

# dset=oxfordpets
# python train.py --dataset-config-file configs/datasets/$dset.yaml

# dset=sun397
# python train.py --dataset-config-file configs/datasets/$dset.yaml

# dset=cifar10
# python train.py --dataset-config-file configs/datasets/$dset.yaml

# dset=cifar100
# python train.py --dataset-config-file configs/datasets/$dset.yaml
