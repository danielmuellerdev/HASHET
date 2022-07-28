import os
from pathlib import Path
import random
import numpy as np
import pickle

from typing import List, Dict

from torch import Tensor

from dataset import DataModule


def get_all_samples(data_module: DataModule) -> List[Dict[str, Tensor]]:
    all_samples = []
    all_samples.extend(data_module.train_dataloader())
    all_samples.extend(data_module.val_dataloader())
    all_samples.extend(data_module.test_dataloader())

    return all_samples

def create_k_fold_cross_validation_splits(
    samples: List[Dict[str, Tensor]], k: int, percent_of_train_set_used_for_val: float = 1 / 8,
    save_dir: Path = 'save_files/'
) -> None:
    random.shuffle(samples)

    partitions = np.array_split(samples, k)

    for split_idx in range(k):
        test_set = partitions[split_idx]
        train_val_set = []
        for i in range(k):
            if i != split_idx:
                train_val_set.extend(partitions[i])
            
        val_set = train_val_set[:int(len(train_val_set) * percent_of_train_set_used_for_val)]
        train_set = train_val_set[int(len(train_val_set) * percent_of_train_set_used_for_val):]

        for split, split_name in [(train_set, 'train'), (val_set, 'val'), (test_set, 'test')]:
            cache = {
                i: {'x': sample['x'].squeeze(), 'y': sample['y'].squeeze()} # [Batchsize=1, 150] -> [150]
                for i, sample in enumerate(split)
            }

            file_path = os.path.join(save_dir, f'{split_name}_dataset_split_{split_idx}.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(cache, file)

        print(
            f'{k}-fold cross-validation split {split_idx}: '
            f'trainset-size: {len(train_set)} valset-size: {len(val_set)} testset-size: {len(test_set)}'
        )

def main():
    data_module = DataModule.restore_from_file()
    samples = get_all_samples(data_module)
    create_k_fold_cross_validation_splits(samples, k=5)

if __name__ == '__main__':
    main()
