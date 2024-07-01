import os
from glob import glob
from os.path import join

import MinkowskiEngine as ME
import numpy as np
import torch
import yaml

from utils.voxelizer import Voxelizer


def get_dataset(name):
    if name == "SemanticKITTI":
        return SemanticKITTIRestrictedDataset
    elif name == "SemanticPOSS":
        return SemanticPOSSRestrictedDataset
    elif name == "S3DIS":
        return S3DISRestrictedDataset
    else:
        raise NameError(f'Dataset "{name}" not yet implemented')


class SemanticKITTIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        self.num_files = len(self.filenames)

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return discrete_coords, unique_feats, unique_labels, selected_idx, t


class SemanticKITTIRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            t,
        )


class SemanticKITTIRestrictedDatasetCleanSplit(SemanticKITTIRestrictedDataset):
    def __init__(
        self,
        clean_mask,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
    ):
        super().__init__(
            config_file, split, voxel_size, downsampling, augment, label_mapping
        )
        self.filenames = np.array(self.filenames)[clean_mask]
        self.num_files = len(self.filenames)
        for key in self.files.keys():
            self.files[key] = np.array(self.files[key])[clean_mask]


class SemanticPOSSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/semposs_dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        learning_map = self.config["learning_map"]
        self.learning_map_function = np.vectorize(lambda x: learning_map[x])

        self.num_files = len(self.filenames)

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            labels = self.learning_map_function(labels)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return discrete_coords, unique_feats, unique_labels, selected_idx, t


class SemanticPOSSRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/semposs_dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        learning_map = self.config["learning_map"]
        self.learning_map_function = np.vectorize(lambda x: learning_map[x])

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            labels = self.learning_map_function(labels)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            t,
        )


class SemanticPOSSRestrictedDatasetCleanSplit(SemanticPOSSRestrictedDataset):
    def __init__(
        self,
        clean_mask,
        config_file="config/semposs_dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
    ):
        super().__init__(
            config_file, split, voxel_size, downsampling, augment, label_mapping
        )
        self.filenames = np.array(self.filenames)[clean_mask]
        self.num_files = len(self.filenames)
        for key in self.files.keys():
            self.files[key] = np.array(self.files[key])[clean_mask]

class S3DISRestrictedDataset(torch.utils.data.Dataset):
    def __init__(self, config_file='/home/csaltori/Projects/CVPR23/config/s3dis.yaml',
                 split='train', voxel_size=0.05, downsampling=80000, augment=False,
                 label_mapping=None):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.augment = augment
        self.downsampling = downsampling
        self.ignore_label = 100
        # self.unknown_classes = unknown_classes

        self.pcd_path = []
        self.labels_path = []

        self.splits = {'train': ['Area1', 'Area2', 'Area3', 'Area4', 'Area6'], 'valid': ['Area5']}
        # self.splits = {'train': ['Area1'], 'valid': ['Area5']}
        self.split_name = split
        self.areas = self.splits[split]

        self.voxel_size = voxel_size

        self.clip_bounds = None
        self.scale_augmentation_bound = (0.95, 1.05)
        self.rotation_augmentation_bound = ((-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20))
        self.translation_augmentation_ratio_bound = None

        self.color_map = np.asarray([(0., 0., 0.),
                                     (174., 199., 232.),
                                     (152., 223., 138.),
                                     (31., 119., 180.),
                                     (255., 187., 120.),
                                     (188., 189., 34.),
                                     (140., 86., 75.),
                                     (255., 152., 150.),
                                     (214., 39., 40.),
                                     (197., 176., 213.),
                                     (148., 103., 189.),
                                     (196., 156., 148.),
                                     (23., 190., 207.)])/255.

        self.labels2names = self.config['names']
        # self.labels2unknown = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        if label_mapping is not None:
            self.labels2unknown = np.array(list((dict(sorted(label_mapping.items()))).values()))
        else:
            self.labels2unknown = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        self.voxelizer = Voxelizer(voxel_size=self.voxel_size,
                                    clip_bound=self.clip_bounds,
                                    use_augmentation=self.augment,
                                    scale_augmentation_bound=self.scale_augmentation_bound,
                                    rotation_augmentation_bound=self.rotation_augmentation_bound,
                                    translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                                    ignore_label=self.ignore_label)

        for area in self.areas:
            files = os.listdir(os.path.join(self.config['data_path'], area, 'points'))
            for file in files:
                pcd_path = os.path.join(self.config['data_path'], area, 'points', file)
                labels_path = os.path.join(self.config['data_path'], area, 'labels', file)

                if os.path.exists(pcd_path) and os.path.exists(labels_path):
                    self.pcd_path.append(pcd_path)
                    self.labels_path.append(labels_path)

        print(f'--> Loaded S3DIS with {len(self.pcd_path)} files')

    def __len__(self):
        return len(self.pcd_path)

    def map_to_unknown(self, labels):
        return self.labels2unknown[labels]

    def __getitem__(self, t, downsample = True):
        pcd_path = self.pcd_path[t]
        label_path = self.labels_path[t]
        scan = np.load(pcd_path)
        labels = np.load(label_path)

        coordinates = scan[:, 0:3]    # get xyz
        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment and self.split_name == 'train':
            # DOWNSAMPLING
            if downsample:
                selected_idx = np.random.choice(coordinates.shape[0], self.downsampling, replace=False)
                coordinates = coordinates[selected_idx]
                features = features[selected_idx]
                labels = labels[selected_idx]
            else:
                selected_idx = np.arange(coordinates.shape[0])

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((coordinates, np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        mapped_labels = self.map_to_unknown(labels)

        if downsample:
            discrete_coords, unique_map = ME.utils.sparse_quantize(
                coordinates=coordinates,
                return_index=True,
                quantization_size=self.voxel_size)

            unique_feats = features[unique_map]
            unique_labels = labels[unique_map]
            mapped_labels = mapped_labels[unique_map]
            selected_idx = selected_idx[unique_map]

            return discrete_coords, unique_feats, unique_labels, selected_idx, mapped_labels, t
        else:
            return coordinates, features, labels, selected_idx, mapped_labels, t

class dataset_wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, t):
        if isinstance(self.dataset, S3DISRestrictedDataset):
            coordinates1, features1, labels1, selected_idx1, mapped_labels1, _ = self.dataset.__getitem__(t, downsample=False)        
            coordinates2, features2, labels2, selected_idx2, mapped_labels2, t = self.dataset.__getitem__(t, downsample=False)

            selected_idx = np.random.choice(coordinates1.shape[0], self.dataset.downsampling, replace=False)
            
            coordinates1 = coordinates1[selected_idx]
            features1 = features1[selected_idx]
            labels1 = labels1[selected_idx]
            mapped_labels1 = mapped_labels1[selected_idx]
            selected_idx1 = selected_idx1[selected_idx]

            discrete_coords1, unique_map = ME.utils.sparse_quantize(
                coordinates=coordinates1,
                return_index=True,
                quantization_size=self.dataset.voxel_size)
            unique_feats1 = features1[unique_map]
            unique_labels1 = labels1[unique_map]
            mapped_labels1 = mapped_labels1[unique_map]
            selected_idx1 = selected_idx1[unique_map]

            coordinates2 = coordinates2[selected_idx]
            features2 = features2[selected_idx]
            labels2 = labels2[selected_idx]
            mapped_labels2 = mapped_labels2[selected_idx]
            selected_idx2 = selected_idx2[selected_idx]

            discrete_coords2, unique_map = ME.utils.sparse_quantize(
                coordinates=coordinates2,
                return_index=True,
                quantization_size=self.dataset.voxel_size)
            unique_feats2 = features2[unique_map]
            unique_labels2 = labels2[unique_map]
            mapped_labels2 = mapped_labels2[unique_map]
            selected_idx2 = selected_idx2[unique_map]

            return discrete_coords1, unique_feats1, unique_labels1, selected_idx1, mapped_labels1, discrete_coords2, unique_feats2, unique_labels2, selected_idx2, mapped_labels2, t
        else:
            to_ret = self.dataset.__getitem__(t)[:-1] + self.dataset.__getitem__(t)

        return to_ret
