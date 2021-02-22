import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from abc import abstractmethod
import torch
from PIL import ImageEnhance

from collections import defaultdict


"""
Data Manager for meta-training methods.
This would need additional params: [n_way, n_shot, n_query, n_eposide]
"""
class MetaDataLoader:
    
    def __init__(self,
                dataset,
                n_batches,
                batch_size,
                n_way,
                n_shot,
                n_query,
                randomize_query,
                p_dict=None):        
        """object to create the dataloader

        Args:
            dataset (MetaDataset): given a class index, return random support and query examples
            batch_size (int): the number of tasks in a task batch
            n_batches (int): total number of task batches
            n_way (int): number of ways in a task
            n_shot (int): number of support examples per class
            n_query (int): average number of query examples per class
            randomize_query (bool): whether to use exactly the same number of examples per class.
            p_dict (dict): maps a class to its probability of being selected, defaults to None (uniform prob.)
        """        
        # super(MetaDataLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.randomize_query = randomize_query

        print("Size of Support:", self.n_shot)
        print("Size of Query:", self.n_query, "randomize query", self.randomize_query)

        self.p_dict = p_dict
        self.sampler = EpisodicBatchSampler(
                            classes=dataset.classes,
                            n_way=self.n_way, 
                            n_shot=self.n_shot,
                            n_query=self.n_query,
                            random_query=self.randomize_query,
                            n_batches=self.n_batches,
                            n_tasks=self.batch_size,
                            p_dict=p_dict)

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=12,
            pin_memory=True,
            collate_fn=lambda ls: collate_fn(
                                    ls=ls,
                                    has_support=(self.n_shot != 0),
                                    has_query=(self.n_query != 0))
        )

    def __iter__(self):
        return iter(self.data_loader)



"""
Samples n_way random classes for each episode.
EpisodicBatchSampler is hereditary. 
Used by almost all pytorch implementations released after Protonet.
"""

class EpisodicBatchSampler(torch.utils.data.Sampler):

    def __init__(self, classes, n_way, n_shot, n_query, random_query, n_tasks, n_batches, p_dict):
        self.classes = classes
        self.n_classes = len(classes)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.random_query = random_query
        self.n_tasks = n_tasks
        self.n_batches = n_batches

        # construct array of probabilities for sampler
        if p_dict is None:
            self.p = np.ones(self.n_classes) / self.n_classes
        else:
            self.p = []
            for cl in self.classes:
                self.p.append(p_dict[cl])

        print("Setting an episodic sampler over classes")
        for cl, prob in zip(self.classes, self.p):
            print(f'({cl}, {prob})')

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            '''
            for self.n_batches number of times,
            each time return the sampled classes' indices for self.n_tasks
            '''
            yield_result = []
            for task_idx in range(self.n_tasks):
                # first determine how many samples to create for each class
                if self.random_query:
                    # here add np.ones ensure every class has at least one query example
                    counts = np.random.multinomial(
                                n=(self.n_query - 1) * self.n_way,
                                pvals=[1/self.n_way] * self.n_way) + \
                                np.ones(shape=self.n_way, dtype=int)
                else:
                    counts = [self.n_query] * self.n_way

                # choose unique classes (the class composition for this task)
                classes_chosen = np.random.choice(
                                    self.classes,
                                    self.n_way,
                                    replace=False,
                                    p=self.p)

                # label these classes with possible label shuffling.
                cl_labels = np.random.permutation(self.n_way)

                for cl, n_query, cl_label in zip(classes_chosen, counts, cl_labels):
                    yield_result.append({
                        'task_idx': task_idx,
                        'cl': cl,
                        'n_shot': self.n_shot,
                        'n_query': n_query,
                        'cl_label': cl_label,
                    })

            yield yield_result


def collate_fn(ls, has_support, has_query):
    result = defaultdict(lambda: defaultdict(list))

    for task_cl_dict in ls:
        task_idx = task_cl_dict['task_idx']
        if has_support:
            result[task_idx]['support_x'].append(task_cl_dict['support_x_cl'])
            result[task_idx]['support_y'].append(task_cl_dict['support_y_cl'])
        if has_query:
            result[task_idx]['query_x'].append(task_cl_dict['query_x_cl'])
            result[task_idx]['query_y'].append(task_cl_dict['query_y_cl'])

    task_indices = sorted(result.keys())

    if has_support:
        support_x_tb = torch.stack(
            [torch.cat(result[task_idx]['support_x'], dim=0)
                for task_idx in task_indices],
            dim=0)
        support_y_tb = torch.stack(
            [torch.cat(result[task_idx]['support_y'], dim=0)
                for task_idx in task_indices],
            dim=0)
    
    if has_query:
        query_x_tb = torch.stack(
            [torch.cat(result[task_idx]['query_x'], dim=0)
                for task_idx in task_indices],
            dim=0)
        query_y_tb = torch.stack(
            [torch.cat(result[task_idx]['query_y'], dim=0)
                for task_idx in task_indices],
            dim=0)

    if has_support and has_query:
        return (support_x_tb, support_y_tb, query_x_tb, query_y_tb)
    elif has_support and (not has_query):
        return support_x_tb, support_y_tb
    elif (not has_support) and has_query:
        return query_x_tb, query_y_tb
    else:
        assert False, 'no support and no query'
        