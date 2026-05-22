from .sweeps_dataset import *


def get_dataloaders(**kwargs):
    version = kwargs.pop('version', 'fusion_model')
    if version == 'fusion_model': 
        from .loader_factory.fusion_model_training import get_loaders
        return get_loaders(**kwargs)
    elif version == 'local_encoder': 
        from .loader_factory.local_encoder_pretraining import get_loaders_simple, get_loaders_kw
        return get_loaders_simple(**kwargs)
    elif version == 'local_encoder_v2': 
        from .loader_factory.local_encoder_pretraining_v2 import get_loaders
        return get_loaders(**kwargs)
    elif version == 'local_encoder_fullspec': 
        from .loader_factory.local_encoder_pretraining import get_loaders_simple, get_loaders_kw
        return get_loaders_kw(**kwargs)
    elif version == 'local_encoder_optimized': 
        from .loader_factory.local_encoder_pretraining import get_loaders_optimized
        return get_loaders_optimized(**kwargs)
    elif version == 'global_encoder': 
        from .loader_factory.global_encoder_pretraining import get_loaders_simple
        return get_loaders_simple(**kwargs)
    elif version == 'custom': 
        from .loader_factory.get_loaders import get_loaders
        return get_loaders(**kwargs)
    else: 
        raise NotImplementedError(version)

