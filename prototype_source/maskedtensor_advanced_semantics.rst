Advanced Semantics
==================

The purpose of this tutorial is to help users understand how some of the advanced semantics work
and how they came to be. We'll focus on two particular ones:

1. Differences between MaskedTensor and `NumPy's MaskedArray <https://numpy.org/doc/stable/reference/maskedarray.html>`__  
2. Reduction semantics

MaskedTensor vs NumPy's MaskedArray
-----------------------------------

NumPy's ``MaskedArray`` has a few fundamental semantics differences from MaskedTensor.

1. Their factory function and basic definition inverts the mask (similar to ``torch.nn.MHA``); that is, MaskedTensor
uses ``True`` to denote "specified" and ``False`` to denote "unspecified", or "valid"/"invalid", whereas NumPy does the
opposite. We believe that our mask definition is not only more intuitive, but it also aligns more with the
existing semantics in PyTorch as a whole.
2. Intersection semantics. In NumPy, if one of two elements are masked out, the resulting element will be
masked out as well -- in practice, they
`apply the logical_or operator <https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024>`__.

    >>> data = torch.arange(5.)
    >>> mask = torch.tensor([True, True, False, True, False])
    >>> npm0 = np.ma.masked_array(data.numpy(), (~mask).numpy())
    >>> npm1 = np.ma.masked_array(data.numpy(), (mask).numpy())
    >>> npm0
    masked_array(data=[0.0, 1.0, --, 3.0, --],
                mask=[False, False,  True, False,  True],
          fill_value=1e+20,
                dtype=float32)
    >>> npm1
    masked_array(data=[--, --, 2.0, --, 4.0],
                mask=[ True,  True, False,  True, False],
          fill_value=1e+20,
                dtype=float32)
    >>> npm0 + npm1
    masked_array(data=[--, --, --, --, --],
                mask=[ True,  True,  True,  True,  True],
          fill_value=1e+20,
                dtype=float32)

Meanwhile, MaskedTensor does not support addition or binary operators with masks that don't match -- to understand why,
please find the :ref:`section on reductions <reduction-semantics>`.

    >>> mt0 = masked_tensor(data, mask)
    >>> mt1 = masked_tensor(data, ~mask)
    >>> m0
    MaskedTensor(
      [  0.0000,   1.0000,       --,   3.0000,       --]
    )
    >>> mt0 = masked_tensor(data, mask)
    >>> mt1 = masked_tensor(data, ~mask)
    >>> mt0
    MaskedTensor(
      [  0.0000,   1.0000,       --,   3.0000,       --]
    )
    >>> mt1
    MaskedTensor(
      [      --,       --,   2.0000,       --,   4.0000]
    )
    >>> mt0 + mt1
    ValueError: Input masks must match. If you need support for this, please open an issue on Github.

However, if this behavior is desired, MaskedTensor does support these semantics by giving access to the data and masks
and conveniently converting a MaskedTensor to a Tensor with masked values filled in using :func:`to_tensor`.

    >>> t0 = mt0.to_tensor(0)
    >>> t1 = mt1.to_tensor(0)
    >>> mt2 = masked_tensor(t0 + t1, mt0.get_mask() & mt1.get_mask())
    >>> t0
    tensor([0., 1., 0., 3., 0.])
    >>> t1
    tensor([0., 0., 2., 0., 4.])
    >>> mt2
    MaskedTensor(
      [      --,       --,       --,       --,       --]

Note that the mask is `mt0.get_mask() & mt1.get_mask()` since :class:`MaskedTensor`'s mask is the inverse of NumPy's.

.. _reduction-semantics:

Reduction semantics
-------------------

Recall in `MaskedTensor's Overview tutorial <https://pytorch.org/tutorials/prototype/maskedtensor_overview.html>`__
we discussed "Implementing missing torch.nan* ops". Those are examples of reductions -- operators that remove one
(or more) dimensions from a Tensor and then aggregate the result. In this section, we'll use reduction semantics
to motivate our strict requirements around matching masks from above.

Fundamentally, :class:`MaskedTensor`s perform the same reduction operation while ignoring the masked out
(unspecified) values. By way of example:

    >>> data = torch.arange(12, dtype=torch.float).reshape(3, 4)
    >>> mask = torch.randint(2, (3, 4), dtype=torch.bool)
    >>> mt = masked_tensor(data, mask)
    >>> data
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]])
    >>> mask
    tensor([[False,  True, False, False],
            [False,  True,  True,  True],
            [ True,  True, False,  True]])
    >>> mt
    MaskedTensor(
      [
        [      --,   1.0000,       --,       --],
        [      --,   5.0000,   6.0000,   7.0000],
        [  8.0000,   9.0000,       --,  11.0000]
      ]
    )

    >>> torch.sum(mt, 1)
    MaskedTensor(
      [  1.0000,  18.0000,  28.0000]
    )
    >>> torch.mean(mt, 1)
    MaskedTensor(
      [  1.0000,   6.0000,   9.3333]
    )
    >>> torch.prod(mt, 1)
    MaskedTensor(
      [  1.0000, 210.0000, 792.0000]
    )
    >>> torch.amin(mt, 1)
    MaskedTensor(
      [  1.0000,   5.0000,   8.0000]
    )
    >>> torch.amax(mt, 1)
    MaskedTensor(
      [  1.0000,   7.0000,  11.0000]
    )

Of note, the value under a masked out element is not guaranteed to have any specific value, especially if the
row or column is entirely masked out (the same is true for normalizations).
For more details on masked semantics, you can find this `RFC <https://github.com/pytorch/rfcs/pull/27>`__.

Now we can revisit the question: why do we enforce the invariant that masks must match for binary operators?
In other words, why don't we use the same semantics as ``np.ma.masked_array``? Consider the following example:

    >>> data0 = torch.arange(10.).reshape(2, 5)
    >>> data1 = torch.arange(10.).reshape(2, 5) + 10
    >>> mask0 = torch.tensor([[True, True, False, False, False], [False, False, False, True, True]])
    >>> mask1 = torch.tensor([[False, False, False, True, True], [True, True, False, False, False]])
    >>> npm0 = np.ma.masked_array(data0.numpy(), (mask0).numpy())
    >>> npm1 = np.ma.masked_array(data1.numpy(), (mask1).numpy())
    >>> npm0
    masked_array(
      data=[[--, --, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, --, --]],
      mask=[[ True,  True, False, False, False],
            [False, False, False,  True,  True]],
      fill_value=1e+20,
      dtype=float32)
    >>> npm1
    masked_array(
      data=[[10.0, 11.0, 12.0, --, --],
            [--, --, 17.0, 18.0, 19.0]],
      mask=[[False, False, False,  True,  True],
            [ True,  True, False, False, False]],
      fill_value=1e+20,
      dtype=float32)

Now let's try addition:

    >>> (npm0 + npm1).sum(0)
    masked_array(data=[--, --, 38.0, --, --],
                mask=[ True,  True, False,  True,  True],
          fill_value=1e+20,
                dtype=float32)
    >>> npm0.sum(0) + npm1.sum(0)
    masked_array(data=[15.0, 17.0, 38.0, 21.0, 23.0],
                mask=[False, False, False, False, False],
          fill_value=1e+20,
                dtype=float32)

Sum and addition should clearly be associative, but with NumPy's semantics, they are allowed to not be,
which can certainly be confusing for the user.

:class:`MaskedTensor`, on the other hand, will simply not allow this operation since `mask0 != mask1`.
That being said, if the user wishes, there are ways around this
(e.g. filling in the MaskedTensor's undefined elements with 0 values using :func:`to_tensor` like shown below),
but the user must now be more explicit with their intentions.

    >>> mt0 = masked_tensor(data0, ~mask0)
    >>> mt1 = masked_tensor(data1, ~mask1)
    >>> (mt0.to_tensor(0) + mt1.to_tensor(0)).sum(0)
    tensor([15., 17., 38., 21., 23.])
