# sitecustomize.py  (Python auto-imports this if it's on PYTHONPATH)
import logging


def modify_function(func, pos):

    def patched_func(*args, **kwargs):
        passed_positionally = (len(args) >= max(0, pos))
        if "antialias" not in kwargs and not passed_positionally:
            kwargs["antialias"] = True
            return func(*args, **kwargs)

    return patched_func


def _force_antialias_true():
    import inspect
    from inspect import getmembers, isfunction

    modules = []
    try:
        import torchvision.transforms.functional as F
        modules.append(F)
    except Exception:
        print('F does not exist')
    try:
        from torchvision.transforms.v2 import functional as F2
        modules.append(F2)
    except Exception:
        print('F2 does not exist')

    for module in modules:

        for (_origname, _orig) in getmembers(module, isfunction):

            try:
                sig = inspect.signature(_orig)
            except TypeError:
                continue

            params = list(sig.parameters.keys())
            if "antialias" not in params:
                continue

            _wstr = '{} from {} will be forced to antialias default to True, no matter the version'
            logging.warning(_wstr.format(_origname, module.__name__))
            aa_pos = params.index("antialias")

            setattr(module, _origname, modify_function(_orig, aa_pos))


_force_antialias_true()
