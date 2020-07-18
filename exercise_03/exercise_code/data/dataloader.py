"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))  # define indices as iterator
        else:
            index_iterator = iter(range(len(self.dataset)))  # define indices as iterator
        
        batch = []
        element_counter = 0
        
        for index in index_iterator:  # iterate over indices using the iterator
            
            batch.append(self.dataset[index])
            element_counter += 1
            
            if self.drop_last or self.is_dataset_size_divisible():
                if len(batch) == self.batch_size:
                    yield batch_to_numpy(combine_batch_dicts(batch))  # use yield keyword to define a iterable generator
                    batch = []
            else:
                if len(batch) == self.batch_size:
                    yield batch_to_numpy(combine_batch_dicts(batch))
                    batch = []
                elif len(batch) < self.batch_size and element_counter == len(self.dataset): 
                    yield batch_to_numpy(combine_batch_dicts(batch))  # also return smaller batch which has smaller size
                    batch = []

    def __len__(self):
        
        if self.drop_last or self.is_dataset_size_divisible():
            return len(self.dataset) // self.batch_size  # do the integer division thing
        else:
            return (len(self.dataset) // self.batch_size) + 1

    
    def is_dataset_size_divisible(self):
        return (len(self.dataset) % self.batch_size) == 0
    
def combine_batch_dicts(batch):
    batch_dict = {}
    for data_dict in batch:
        for key, value in data_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    return batch_dict

def batch_to_numpy(batch):
    numpy_batch = {}
    for key, value in batch.items():
        numpy_batch[key] = np.array(value)
    return numpy_batch