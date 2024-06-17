import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
"airplane":"airplane",
"airport":"airport",
"bareland":"bareland",
"baseball_diamond":"baseball diamond",
"basketball_court":"basketball court",
"beach":"beach",
"bridge":"bridge",
"chaparral":"chaparral",
"cloud":"cloud",
"commercial_area":"commercial area",
"dense_residential_area":"dense residential area",
"desert":"desert",
"farmland":"farmland",
"forest":"forest",
"freeway":"freeway",
"golf_course":"golf course",
"ground_track_field":"ground track field",
"harbor&port":"harbor and/or port",
"industrial_area":"industrial area",
"intersection":"intersection",
"island":"island",
"lake":"lake",
"meadow":"meadow",
"mobile_home_park":"mobile home park",
"mountain":"mountain",
"overpass":"overpass",
"park":"park",
"parking_lot":"parking lot",
"parkway":"parkway",
"railway":"railway",
"railway_station":"railway station",
"river":"river",
"roundabout":"roundabout",
"shipping_yard":"shipping yard",
"snowberg":"snowberg",
"sparse_residential_area":"sparse residential area",
"stadium":"stadium",
"storage_tank":"storage tank",
"swimmimg_pool":"swimmimg pool",
"tennis_court":"tennis court",
"terrace":"terrace",
"transmission_tower":"transmission tower",
"vegetable_greenhouse":"vegetable greenhouse",
"wetland":"wetland",
"wind_turbine":"wind turbine",
}


@DATASET_REGISTRY.register()
class MLRSNet(DatasetBase):

    dataset_dir = "mlrsnet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Images")
        print("mlrs_split_new.json")
        self.split_path = os.path.join(self.dataset_dir, "mlrsnet_split_new.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train_, val_, test_ = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train_, val_, test_ = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train_, val_, test_, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train_, val_ = data["train"], data["val"]
            else:
                train_ = self.generate_fewshot_dataset(train_, num_shots=num_shots)
                val_ = self.generate_fewshot_dataset(val_, num_shots=min(num_shots, 4))
                data = {"train": train_, "val": val_}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train_, val_, test_, subsample=subsample)
        train_all, _, _ = OxfordPets.subsample_classes(train_, val_, test_, subsample='all')

        # super().__init__(train_x=train, val=val, test=test, train_u=train_all)
        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new