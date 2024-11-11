rm -rf embeddings/*
for dset in "eurosat" "caltech101" "oxfordpets" "oxford_flowers" "sun397" "cifar100" "resisc45" "ucf101" "dtd" "cifar10";
do
    python train.py --dataset-config-file configs/datasets/$dset.yaml
done
