def get_models(models):
    model_names = []
    temp = [name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])]
    model_names.extend(temp)
    sorted(model_names)
    return  model_names
