# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get data."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.
    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.
    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.separate_eval = separate_eval
        if not separate_eval:
            if len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'All the datasets should have same types')

    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        total_eval_results = dict()
        dataset_idx = -1
        for size, dataset in zip(self.cumulative_sizes, self.datasets):
            start_idx = 0 if dataset_idx == -1 else self.cumulative_sizes[dataset_idx]
            end_idx = self.cumulative_sizes[dataset_idx + 1]

            results_per_dataset = results[start_idx:end_idx]
            eval_results_per_dataset = dataset.evaluate(results_per_dataset, res_folder, metric, **kwargs)
            dataset_idx += 1
            for k, v in eval_results_per_dataset.items():
                if self.separate_eval:
                    total_eval_results.update({f'{dataset_idx}_{k}': v})
                else:
                    total_eval_results[k] = total_eval_results.get(k, 0.0) + v
        if not self.separate_eval:
            for k in total_eval_results:
                total_eval_results[k] /= len(self.datasets)
        return total_eval_results
