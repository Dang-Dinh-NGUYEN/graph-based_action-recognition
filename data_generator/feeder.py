import pickle
import random
import numpy as np
import torch

from data_generator import register_feeder
from data_generator.utils import augmentation

@register_feeder("single_feeder")
class Single_feeder(torch.utils.data.Dataset):
    """Feeder for single inputs"""

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True, **kwargs):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)
    
    def num_classes(self):
        return max(self.label) + 1
    
    def num_features(self):
        return self.data[0].shape[0]
    
    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = augmentation.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = augmentation.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy
    
@register_feeder("dual_feeder")
class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True, **kwargs):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing (augmentations)
        data1 = self._aug(data_numpy)
        data2 = self._aug(data_numpy)

        # Convert to torch.Tensor if still numpy
        if isinstance(data1, np.ndarray):
            data1 = torch.from_numpy(data1).float()
        if isinstance(data2, np.ndarray):
            data2 = torch.from_numpy(data2).float()

        # Convert label to tensor as well
        label = torch.tensor(label, dtype=torch.long)

        return [data1, data2], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = augmentation.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = augmentation.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy
    
@register_feeder("semi_feeder")
class Feeder_semi(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.0, temperal_padding_ratio=0,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = augmentation.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = augmentation.shear(data_numpy, self.shear_amplitude)

        return data_numpy