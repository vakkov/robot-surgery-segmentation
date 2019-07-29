
def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, string_types)
    if hasattr(input, '__iter__') and not is_str:
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized




def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None):
    """
    Exact euclidean distance transform.
    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element is returned along the first axis of the result.
    Parameters
    ----------
    input : array_like
        Input data to transform. Can be any type but will be converted
        into binary: 1 wherever input equates to True, 0 elsewhere.
    sampling : float or int, or sequence of same, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to return distance matrix. At least one of
        return_distances/return_indices must be True. Default is True.
    return_indices : bool, optional
        Whether to return indices matrix. Default is False.
    distances : ndarray, optional
        Used for output of distance array, must be of type float64.
    indices : ndarray, optional
        Used for output of indices, must be of type int32.
    Returns
    -------
    distance_transform_edt : ndarray or list of ndarrays
        Either distance matrix, index matrix, or a list of the two,
        depending on `return_x` flags and `distance` and `indices`
        input parameters. 
    """ 
          
    if (not return_distances) and (not return_indices):
        msg = 'at least one of distances/indices must be specified'
        raise RuntimeError(msg)

    ft_inplace = isinstance(indices, torch.int32)
    dt_inplace = isinstance(distances, torch.int32)
    # calculate the feature transform
    #input = numpy.atleast_1d(numpy.where(input, 1, 0).astype(numpy.int8))
    input = (torch.where(input, 1, 0).type(torch.int8))
    if sampling is not None:
        sampling = _normalize_sequence(sampling, input.ndim)
        sampling = sampling.type(torch.float64)
        if not sampling.is_contiguous():
            sampling = sampling.clone()

    if ft_inplace:
        ft = indices
        if ft.shape != (input.ndim,) + input.shape:
            raise RuntimeError('indices has wrong shape')
        if ft.dtype.type != torch.int32:
            raise RuntimeError('indices must be of int32 type')
    else:
        ft = torch.zeros((input.ndim,) + input.shape).type(torch.int32)

    _nd_image.euclidean_feature_transform(input, sampling, ft)
    # if requested, calculate the distance transform
    if return_distances:
        dt = ft - numpy.indices(input.shape).type(ft.dtype)
        dt = dt.type(torch.float64)
        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]
        torch.mul(dt, dt, dt)
        if dt_inplace:
            dt = torch.sum(dt, dim=0)
            if distances.shape != dt.shape:
                raise RuntimeError('indices has wrong shape')
            if distances.dtype != torch.float64:
                raise RuntimeError('indices must be of float64 type')
            numpy.sqrt(dt, distances)
        else:
            dt = torch.sum(dt, dim=0)
            dt = torch.sqrt(dt)

    # construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
return None
