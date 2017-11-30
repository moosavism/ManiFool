import torch
import torch.utils.data as data


class CIFAR10TransformedDataset(data.Dataset):
    """
    A class extending the Dataset class to hold transformed CIFAR10 images. Used mainly for fine tuning.
    """
    def __init__(self, path):
        """
        Initialize the object

        :param path: the path of the dataset.
        """
        super(CIFAR10TransformedDataset, self).__init__()

        self.image_list = torch.load(path)

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.image_list)

    def __getitem__(self,index):
        """
        Gets the image with the index from the dataset
        :param index: index of the requested image
        :return: the image and its label
        """
        img, label = self.image_list[index]

        return img, label
