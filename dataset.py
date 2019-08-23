import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.pytorch.functional import img_to_tensor
from loss import class2one_hot, one_hot2dist


class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                if self.problem_type == 'parts':
                    num_classes = 4
                elif self.problem_type == 'instruments':
                     num_classes = 8
                # mask_tensor = torch.tensor(mask, dtype=torch.int64)
                # mask_onehot = class2one_hot(mask_tensor, num_classes)[0]
                # mask_distmap = one_hot2dist(mask_onehot.cpu().numpy())
                # mask_distmap = torch.from_numpy(mask_distmap).float()
                # return img_to_tensor(image), mask_tensor.long(), mask_onehot, mask_distmap
                return img_to_tensor(image), torch.from_numpy(mask).long()
                #return img_to_tensor(image), mask_tensor.long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = prepare_data.binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = prepare_data.parts_factor
    elif problem_type == 'instruments':
        factor = prepare_data.instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)
