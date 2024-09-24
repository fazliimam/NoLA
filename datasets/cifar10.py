# import os
# import pickle

# from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from dassl.utils import mkdir_if_missing

# from .oxford_pets import OxfordPets
# from .dtd import DescribableTextures as DTD

# NEW_CNAMES = {
#     "airplane": "airplane",
#     "automobile": "automobile",
#     "bird": "bird",
#     "cat": "cat",
#     "deer": "deer",
#     "dog": "dog",
#     "frog": "frog",
#     "horse": "horse",
#     "ship": "ship",
#     "truck": "truck",
# }

# @DATASET_REGISTRY.register()
# class CIFAR10(DatasetBase):

#     dataset_dir = "cifar10"
    

#     def __init__(self, cfg):
#         root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
#         self.dataset_dir = os.path.join(root, self.dataset_dir)
#         self.image_dir = os.path.join(self.dataset_dir, "images")
#         self.split_path = os.path.join(self.dataset_dir, "split_CIFAR10.json")
        
#         self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
#         mkdir_if_missing(self.split_fewshot_dir)

#         if os.path.exists(self.split_path):
#             train_, val_, test_ = OxfordPets.read_split(self.split_path, self.image_dir)
#         else:
#             # breakpoint()
#             train_, val_, test_ = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
#             OxfordPets.save_split(train_, val_, test_, self.split_path, self.image_dir)

#         num_shots = cfg.DATASET.NUM_SHOTS
#         if num_shots >= 1:
#             seed = cfg.SEED
#             preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
#             if os.path.exists(preprocessed):
#                 print(f"Loading preprocessed few-shot data from {preprocessed}")
#                 with open(preprocessed, "rb") as file:
#                     data = pickle.load(file)
#                     train_, val_ = data["train"], data["val"]
#             else:
#                 train_ = self.generate_fewshot_dataset(train_, num_shots=num_shots)
#                 val_ = self.generate_fewshot_dataset(val_, num_shots=min(num_shots, 4))
#                 data = {"train": train_, "val": val_}
#                 print(f"Saving preprocessed few-shot data to {preprocessed}")
#                 with open(preprocessed, "wb") as file:
#                     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

#         subsample = cfg.DATASET.SUBSAMPLE_CLASSES
#         train, val, test = OxfordPets.subsample_classes(train_, val_, test_, subsample=subsample)
#         train_all, _, _ = OxfordPets.subsample_classes(train_, val_, test_, subsample='all')

#         # super().__init__(train_x=train, val=val, test=test, train_u=train_all)
#         super().__init__(train_x=train, val=val, test=test)

