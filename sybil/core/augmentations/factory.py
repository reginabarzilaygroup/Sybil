from sandstone.augmentations.basic import ToTensor
NON_AUG_ERR = "Augmentation {} not in AUGMENTATION_REGISTRY! Available augmentations are {}"

IMAGE_AUGMENTATION_REGISTRY = {}
TENSOR_AUGMENTATION_REGISTRY = {}


def RegisterTensorAugmentation(name):
    """Registers a dataset."""

    def decorator(obj):
        TENSOR_AUGMENTATION_REGISTRY[name] = obj
        obj.name = name
        return obj

    return decorator


def RegisterImageAugmentation(name):
    """Registers a dataset."""

    def decorator(obj):
        IMAGE_AUGMENTATION_REGISTRY[name] = obj
        obj.name = name
        return obj

    return decorator


def get_augmentations(image_augmentations, tensor_augmentations, args):
    augmentations =  []
    augmentations = _add_augmentations(augmentations, image_augmentations,
                                     IMAGE_AUGMENTATION_REGISTRY, args)
    augmentations.append(ToTensor())
    augmentations = _add_augmentations(augmentations, tensor_augmentations,
                                     TENSOR_AUGMENTATION_REGISTRY, args)
    return augmentations

def get_tensor_augmentation(augmentation_name):
    if augmentation_name not in TENSOR_AUGMENTATION_REGISTRY:
        raise Exception(NON_AUG_ERR.format(augmentation_name, TENSOR_AUGMENTATION_REGISTRY.keys()))

    return TENSOR_AUGMENTATION_REGISTRY[augmentation_name]

def _add_augmentations(augmentations, new_augmentations, registry, args):
    for trans in new_augmentations:
        name = trans[0]
        kwargs = trans[1]
        if name not in registry:
            raise Exception(NON_AUG_ERR.format(name, registry.keys()))

        augmentations.append(registry[name](args, kwargs))

    return augmentations

