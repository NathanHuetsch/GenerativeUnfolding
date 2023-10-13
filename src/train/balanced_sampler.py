import torch

class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, one_hot_categories, target_ratios):
        self.categories = [
            torch.where(one_hot_categories[:,i] != 0)[0]
            for i in range(one_hot_categories.shape[1])
        ]
        counts = [len(cat) for cat in self.categories]
        min_count = min(counts)
        self.counts = [int(ratio * min_count) for ratio in target_ratios]
        self.length = sum(self.counts)

    def __iter__(self):
        indices = [
            cat[torch.randperm(len(cat))[:count]]
            for cat, count in zip(self.categories, self.counts)
        ]
        indices = torch.cat(indices)
        yield from indices[torch.randperm(len(indices))].tolist()

    def __len__(self):
        return self.length

