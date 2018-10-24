from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self,features, targets):
        self.features = features
        self.targets = targets

        self.len_features = len(self.features)

    def __getitem__(self, item):

        features = self.features[item]
        targets = self.targets[item]

        return features,targets

    def __len__(self):
        return self.len_features