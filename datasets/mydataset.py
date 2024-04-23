import torch
from PIL import Image
from torchvision import transforms
from .split_data import read_split_data
from torch.utils.data import Dataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform


class MyDataset(Dataset):
    def __init__(self, image_paths, image_labels, transforms=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transforms = transforms

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert('RGB')
        label = self.image_labels[item]
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # size = int((256 / 224) * args.input_size)
        size = int((1.0 / 0.96) * args.input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(args):
    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(args.data_root)

    train_transform = build_transform(True, args)
    valid_transform = build_transform(False, args)

    train_set = MyDataset(train_image_path, train_image_label, train_transform)
    valid_set = MyDataset(val_image_path, val_image_label, valid_transform)

    return train_set, valid_set

