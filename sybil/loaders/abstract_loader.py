import torch
import os
import sys
import os.path
import warnings
from sybil.datasets.utils import get_scaled_annotation_mask, IMG_PAD_TOKEN
from sybil.augmentations import ComposeAug
import numpy as np
from abc import ABCMeta, abstractmethod
import hashlib


CACHED_FILES_EXT = ".png"
DEFAULT_CACHE_DIR = "default/"

CORUPTED_FILE_ERR = (
    "WARNING! Error processing file from cache - removed file from cache. Error: {}"
)


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()


def split_augmentations_by_cache(augmentations):
    """
    Given a list of augmentations, returns a list of tuples. Each tuple
    contains a caching key of the augmentations up to the spiltting point,
    and a list of augmentations that should be applied afterwards.

    split_augmentations will contain all possible splits by cachable augmentations,
    ordered from latest possible one to the former ones.
    The last tuple will have all augmentations.

    Note - splitting will be done for indexes that all augmentations up to them are
    cachable.
    """
    # list of (cache key, post augmentations)
    split_augmentations = []
    split_augmentations.append((DEFAULT_CACHE_DIR, augmentations))
    all_prev_cachable = True
    key = ""
    for ind, trans in enumerate(augmentations):

        # check trans.cachable() first separately to save run time
        if not all_prev_cachable or not trans.cachable():
            all_prev_cachable = False
        else:
            key += trans.caching_keys()
            post_augmentations = (
                augmentations[ind + 1 :] if ind < len(augmentations) else []
            )
            split_augmentations.append((key, post_augmentations))

    return list(reversed(split_augmentations))


def apply_augmentations_and_cache(
    loaded_input, sample, img_path, augmentations, cache, base_key=""
):
    """
    Loads the loaded input by its absolute path and apply the augmentations one
    by one (similar to what the composed one is doing).  All first cachable
    transformer's output is cached (until reaching a non cachable one).
    """
    all_prev_cachable = True
    key = base_key
    for ind, trans in enumerate(augmentations):
        loaded_input = trans(loaded_input, sample)
        if not all_prev_cachable or not trans.cachable():
            all_prev_cachable = False
        else:
            key += trans.caching_keys()
            cache.add(img_path, key, loaded_input["input"])

    return loaded_input


class cache:
    def __init__(self, path, extension=CACHED_FILES_EXT):
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache_dir = path
        self.files_extension = extension
        if ".npy" != extension:
            self.files_extension += ".npy"

    def _file_dir(self, attr_key, par_dir):
        return os.path.join(self.cache_dir, attr_key, par_dir)

    def _file_path(self, attr_key, par_dir, hashed_key):
        return os.path.join(
            self.cache_dir, attr_key, par_dir, hashed_key + self.files_extension
        )

    def _parent_dir(self, path):
        return os.path.basename(os.path.dirname(path))

    def exists(self, image_path, attr_key):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        return os.path.isfile(self._file_path(attr_key, par_dir, hashed_key))

    def get(self, image_path, attr_key):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        return np.load(self._file_path(attr_key, par_dir, hashed_key))

    def add(self, image_path, attr_key, image):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        file_dir = self._file_dir(attr_key, par_dir)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save(self._file_path(attr_key, par_dir, hashed_key), image)

    def rem(self, image_path, attr_key):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        try:
            os.remove(self._file_path(attr_key, par_dir, hashed_key))
        # Don't raise error if file not exists.
        except OSError:
            pass


class abstract_loader:
    __metaclass__ = ABCMeta

    def __init__(self, cache_path, augmentations, args):
        self.pad_token = IMG_PAD_TOKEN
        self.augmentations = augmentations
        self.args = args
        if cache_path is not None:
            self.use_cache = True
            self.cache = cache(cache_path, self.cached_extension)
            self.split_augmentations = split_augmentations_by_cache(augmentations)
        else:
            self.use_cache = False
            self.composed_all_augmentations = ComposeAug(augmentations)

    @abstractmethod
    def load_input(self, path, sample):
        pass

    @property
    @abstractmethod
    def cached_extension(self):
        pass

    @property
    @abstractmethod
    def apply_augmentations(self):
        return True

    def configure_path(self, path, sample):
        return path 

    def get_image(self, path, sample):
        """
        Returns a transformed image by its absolute path.
        If cache is used - transformed image will be loaded if available,
        and saved to cache if not.
        """
        input_dict = {}
        input_path = self.configure_path(path, sample)

        if input_path == self.pad_token:
            return self.load_input(input_path, sample)

        if not self.use_cache:
            input_dict = self.load_input(input_path, sample)
            # hidden loaders typically do not use augmentation
            if self.apply_augmentations:
                input_dict = self.composed_all_augmentations(input_dict, sample)
            return input_dict

        if self.args.use_annotations:
            input_dict["mask"] = get_scaled_annotation_mask(
                sample["annotations"], self.args
            )

        for key, post_augmentations in self.split_augmentations:
            base_key = (
                DEFAULT_CACHE_DIR + key
                if key != DEFAULT_CACHE_DIR
                else DEFAULT_CACHE_DIR
            )
            if self.cache.exists(input_path, base_key):
                try:
                    input_dict["input"] = self.cache.get(input_path, base_key)
                    if self.apply_augmentations:
                        input_dict = apply_augmentations_and_cache(
                            input_dict,
                            sample,
                            input_path,
                            post_augmentations,
                            self.cache,
                            base_key=base_key,
                        )
                    return input_dict
                except Exception as e:
                    print(e)
                    hashed_key = md5(input_path)
                    par_dir = self.cache._parent_dir(input_path)
                    corrupted_file = self.cache._file_path(key, par_dir, hashed_key)
                    warnings.warn(CORUPTED_FILE_ERR.format(sys.exc_info()[0]))
                    self.cache.rem(input_path, key)
        all_augmentations = self.split_augmentations[-1][1]
        input_dict = self.load_input(input_path, sample)
        if self.apply_augmentations:
            input_dict = apply_augmentations_and_cache(
                input_dict,
                sample,
                input_path,
                all_augmentations,
                self.cache,
                base_key=key,
            )

        return input_dict
