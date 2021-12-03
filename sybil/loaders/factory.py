from sandstone.utils.generic import log

NO_LOADER_ERR = "Image loader function {} not in INPUT_LOADER_REGISTRY! Available loaders are {}"
INPUT_LOADER_REGISTRY = {}


def RegisterInputLoader(loader_name):
    """Registers an image loader."""

    def decorator(f):
        INPUT_LOADER_REGISTRY[loader_name] = f
        return f
    return decorator

def get_input_loader(cache_path, augmentations, args):
    if args.input_loader_name not in INPUT_LOADER_REGISTRY:
        raise Exception(NO_LOADER_ERR.format(args.input_loader_name, INPUT_LOADER_REGISTRY.keys()))
    return INPUT_LOADER_REGISTRY[args.input_loader_name](cache_path, augmentations, args)