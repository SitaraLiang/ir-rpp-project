def get_shape(d):
    shape = []
    while isinstance(d, dict):
        shape.append(len(d))
        d = next(iter(d.values()))
    return tuple(shape)
