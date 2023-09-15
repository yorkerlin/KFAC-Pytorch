from .kfac import KFACOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    else:
        raise NotImplementedError
