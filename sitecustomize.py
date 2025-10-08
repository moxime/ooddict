# sitecustomize.py  (Python auto-imports this if it's on PYTHONPATH)
print('***\n'*100)


def _force_antialias_true():
    import inspect

    # Patch v1 functional API used by transforms.Resize
    try:
        import torchvision.transforms.functional as F
        _orig = F.resize
        sig = inspect.signature(_orig)
        params = list(sig.parameters.keys())
        has_antialias = "antialias" in params
        aa_pos = params.index("antialias") if has_antialias else None

        def patched_resize(img, size, *args, **kwargs):
            if has_antialias:
                # If antialias not provided as kwarg and not passed positionally, force True
                passed_positionally = (len(args) >= max(0, aa_pos - 2))  # after img,size
                if "antialias" not in kwargs and not passed_positionally:
                    kwargs["antialias"] = True
            # Drop kwargs the original doesn't accept (older torchvision)
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return _orig(img, size, *args, **kwargs)

        F.resize = patched_resize
    except Exception:
        pass

    # Patch v2 functional API if present
    try:
        from torchvision.transforms.v2 import functional as F2
        _orig2 = F2.resize
        sig2 = inspect.signature(_orig2)
        if "antialias" in sig2.parameters:
            def patched_resize2(img, size, *args, **kwargs):
                if "antialias" not in kwargs:
                    kwargs["antialias"] = True
                kwargs = {k: v for k, v in kwargs.items() if k in sig2.parameters}
                return _orig2(img, size, *args, **kwargs)
            F2.resize = patched_resize2
    except Exception:
        pass


_force_antialias_true()
