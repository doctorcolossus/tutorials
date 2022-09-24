MaskedTensor Overview
=====================

This tutorial is designed to serve as a starting point for using MaskedTensors
and discuss its masking semantics.

Using MaskedTensor
++++++++++++++++++

Construction
------------

There are a few different ways to construct a MaskedTensor:

* The first way is to directly invoke the MaskedTensor class
* The second (and our recommended way) is to use :func:`masked.masked_tensor` and :func:`masked.as_masked_tensor` factory functions,
  which are analogous to :func:`torch.tensor` and :func:`torch.as_tensor`

  .. autosummary::
    :toctree: generated
    :nosignatures:

    masked.masked_tensor
    masked.as_masked_tensor

Throughout this tutorial, we will be assuming the import line: `from torch.masked import masked_tensor`.

Accessing the data and mask
---------------------------

The underlying fields in a MaskedTensor can be accessed through:

* the :meth:`MaskedTensor.get_data` function
* the :meth:`MaskedTensor.get_mask` function. Recall that ``True`` indicates "specified" or "valid"
  while ``False`` indicates "unspecified" or "invalid".

In general, the underlying data that is returned may not be valid in the unspecified entries, so we recommend that
when users require a Tensor without any masked entries, that they use :meth:`MaskedTensor.to_tensor` (as shown above) to
return a Tensor with filled values.

Indexing and slicing
--------------------

:class:`MaskedTensor` is a Tensor subclass, which means that it inherits the same semantics for indexing and slicing
as :class:`torch.Tensor`. Below are some examples of common indexing and slicing patterns:

    >>> data = torch.arange(24).reshape(2, 3, 4)
    >>> mask = data % 2 == 0
    >>> data
    tensor([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

            [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> mask
    tensor([[[ True, False,  True, False],
            [ True, False,  True, False],
            [ True, False,  True, False]],

            [[ True, False,  True, False],
            [ True, False,  True, False],
            [ True, False,  True, False]]])
    >>> mt = masked_tensor(data.float(), mask)  # .float() used for visualization
    >>> mt
    MaskedTensor(
    [
        [
            [0,       --, 2,       --],
            [4,       --, 6,       --],
            [8,       --, 10,       --]
        ],
        [
            [12,       --, 14,       --],
            [16,       --, 18,       --],
            [20,       --, 22,       --]
        ]
    ])
    >>> mt[0]
    MaskedTensor(
    [
        [0,       --, 2,       --],
        [4,       --, 6,       --],
        [8,       --, 10,       --]
    ])
    >>> mt[:, :, 2:4]
    MaskedTensor(
    [
        [
            [2,       --],
            [6,       --],
            [10,       --]
        ],
        [
            [14,       --],
            [18,       --],
            [22,       --]
        ]
    ])

Why is MaskedTensor useful?
+++++++++++++++++++++++++++

Because of :class:`MaskedTensor`'s treatment of specified and unspecified values as a first-class citizen
instead of an afterthought (with filled values, nans, etc.), it is able to solve for several of the shortcomings
that regular Tensors are unable to; indeed, :class:`MaskedTensor` was born in a large part due to these recurring issues.

Below, we will discuss some of the most common issues that are still unresolved in PyTorch today
and illustrate how :class:`MaskedTensor` can solve these problems.

Distinguishing between 0 and NaN gradient
-----------------------------------------

One issue that :class:`torch.Tensor` runs into is the inability to distinguish between gradients that are
undefined (NaN) vs. gradients that are actually 0. Because PyTorch does not have a way of marking a value
as specified/valid vs. unspecified/invalid, it is forced to rely on NaN or 0 (depending on the use case), leading
to unreliable semantics since many operations aren't meant to handle NaN values properly. What's even more confusing
is that sometimes depending on the order of operations, the gradient could vary (e.g. depending on how early
in the chain of operations a NaN value manifests).

:class:`MaskedTensor` is the perfect solution for this!

`Issue 10729 <https://github.com/pytorch/pytorch/issues/10729>`__ -- :func:`torch.where`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current result:

    >>> x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True, dtype=torch.float)
    >>> y = torch.where(x < 0, torch.exp(x), torch.ones_like(x))
    >>> y.sum().backward()
    >>> x.grad
    tensor([4.5400e-05, 6.7379e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00,        nan,        nan])

:class:`MaskedTensor` result:

    >>> x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100])
    >>> mask = x < 0
    >>> mx = masked_tensor(x, mask, requires_grad=True)
    >>> my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
    >>> y = torch.where(mask, torch.exp(mx), my)
    >>> y.sum().backward()
    >>> mx.grad
    MaskedTensor(
      [  0.0000,   0.0067,       --,       --,       --,       --,       --,       --,       --,       --,       --]
    )

The gradient here is only provided to the selected subset. Effectively, this changes the gradient of `where`
to mask out elements instead of setting them to zero.

`Issue 52248 <https://github.com/pytorch/pytorch/issues/52248>`__ -- another :func:`torch.where`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current result:

    >>> a = torch.randn((), requires_grad=True)
    >>> b = torch.tensor(False)
    >>> c = torch.ones(())
    >>> torch.where(b, a/0, c)
    tensor(1., grad_fn=<WhereBackward0>)
    >>> torch.autograd.grad(torch.where(b, a/0, c), a)
    (tensor(nan),)

:class:`MaskedTensor` result:

    >>> a = masked_tensor(torch.randn(()), torch.tensor(True), requires_grad=True)
    >>> b = torch.tensor(False)
    >>> c = torch.ones(())
    >>> torch.where(b, a/0, c)
    MaskedTensor(  1.0000, True)
    >>> torch.autograd.grad(torch.where(b, a/0, c), a)
    (MaskedTensor(--, False),)

This issue is similar (and even links to the next issue below) in that it expresses frustration with unexpected behavior
because of the inability to differentiate "no gradient" vs "zero gradient", which in turn makes
working with other ops difficult and unreliable.

`Issue 4132 <https://github.com/pytorch/pytorch/issues/4132>`__ -- when using mask, x/0 yields NaN grad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current result:

    >>> x = torch.tensor([1., 1.], requires_grad=True)
    >>> div = torch.tensor([0., 1.])
    >>> y = x/div # => y is [inf, 1]
    >>> mask = (div != 0)  # => mask is [0, 1]
    >>> y[mask].backward()
    >>> x.grad
    tensor([nan, 1.])

:class:`MaskedTensor` result:

    >>> x = torch.tensor([1., 1.], requires_grad=True)
    >>> div = torch.tensor([0., 1.])
    >>> y = x/div # => y is [inf, 1]
    >>>
    >>> mask = (div != 0) # => mask is [0, 1]
    >>> loss = as_masked_tensor(y, mask)
    >>> loss.sum().backward()
    >>> x.grad
    MaskedTensor(
      [      --,   1.0000]
    )

Linked in the issue above, this issue proposes that `x.grad` should be `[0, 1]` instead of the `[nan, 1]`,
whereas :class:`MaskedTensor` makes this very clear by masking out the gradient altogether.

`Issue 67180 <https://github.com/pytorch/pytorch/issues/67180>`__ -- :func:`torch.nansum` and :func:`torch.nanmean`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current result:

    >>> a = torch.tensor([1., 2., float('nan')])
    >>> b = torch.tensor(1.0, requires_grad=True)
    >>> c = a * b
    >>> c1 = torch.nansum(c)
    >>> bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
    >>> bgrad1
    tensor(nan)

:class:`MaskedTensor` result:

    >>> a = torch.tensor([1., 2., float('nan')])
    >>> b = torch.tensor(1.0, requires_grad=True)
    >>> mt = masked_tensor(a, ~torch.isnan(a))
    >>> c = mt * b
    >>> c1 = torch.sum(c)
    >>> bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
    >>> bgrad1
    MaskedTensor(  3.0000, True)

Here, the gradient doesn't even calculate properly (a longstanding issue), whereas :class:`MaskedTensor` handles
it correctly.

Safe Softmax
------------

Safe softmax is another great example of `an issue <https://github.com/pytorch/pytorch/issues/55056>`__
that arises frequently. In a nutshell, if there is an entire batch that is "masked out"
or consists entirely of padding (which, in the softmax case, translates to being set `-inf`),
then this will result in NaNs, which can leading to training divergence.

Luckily, :class:`MaskedTensor` has solved this issue. Consider this setup:

    >>> data = torch.randn(3, 3)
    >>> mask = torch.tensor([[True, False, False], [True, False, True], [False, False, False]])
    >>> x = data.masked_fill(~mask, float('-inf'))
    >>> mt = masked_tensor(data, mask)
    >>> x
    tensor([[ 0.4216,    -inf,    -inf],
            [-0.7676,    -inf,  1.4216],
            [   -inf,    -inf,    -inf]])
    >>> mt
    MaskedTensor(
    [
        [  0.4216,       --,       --],
        [ -0.7676,       --,   1.4216],
        [      --,       --,       --]
    ]
    )

Suppose we want to calculate the softmax along `dim=0`. Note that the second column is "unsafe" (i.e. entirely
masked out), so when the softmax is calculated, the result will yield `0/0 = nan` since `exp(-inf) = 0`.
However, what we would really like is for the gradients to be masked out since they are unspecified and would be
invalid for training.

PyTorch result:

    >>> x.softmax(0)
    tensor([[0.3548,    nan, 0.0000],
            [0.6452,    nan, 1.0000],
            [0.0000,    nan, 0.0000]])

:class:`MaskedTensor` result:

    >>> mt.softmax(0)
    MaskedTensor(
      [
        [  0.3548,       --,       --],
        [  0.6452,       --,   1.0000],
        [      --,       --,       --]
      ]
    )

`Issue 61474 -- Implementing missing torch.nan* operators <https://github.com/pytorch/pytorch/issues/61474>`__
--------------------------------------------------------------------------------------------------------------

In the above issue, there is a request to add additional operators to cover the various `torch.nan*` applications,
such as ``torch.nanmax``, ``torch.nanmin``, etc.

In general, these problems lend themselves more naturally to masked semantics, so instead of introducing additional
operators, we propose using :class:`MaskedTensor`s instead. Since
`nanmean has already landed <https://github.com/pytorch/pytorch/issues/21987>`__, we can use it as a comparison point:

    >>> x = torch.arange(16).float()
    >>> y = x * x.fmod(4)
    >>> z = y.masked_fill(y == 0, float('nan'))  # we want to get the mean of y when ignoring the zeros
    >>> y
    tensor([ 0.,  1.,  4.,  9.,  0.,  5., 12., 21.,  0.,  9., 20., 33.,  0., 13.,
            28., 45.])
    >>> z   # z is just y with the zeros replaced with nan's
    tensor([nan,  1.,  4.,  9., nan,  5., 12., 21., nan,  9., 20., 33., nan, 13.,
            28., 45.])
    >>> y.mean()
    tensor(12.5000)
    >>> z.nanmean()
    tensor(16.6667)
    >>> torch.mean(masked_tensor(y, y != 0))  # MaskedTensor successfully ignores the 0's
    MaskedTensor( 16.6667, True)

In the above example, we've constructed a `y` and would like to calculate the mean of the series while ignoring
the zeros. `torch.nanmean` can be used to do this, but we don't have implementations for the rest of the
`torch.nan*` operations. :class:`MaskedTensor` solves this issue by being able to use the base operation,
and we already have support for the other operations listed in the issue. For example:

    >>> torch.argmin(masked_tensor(y, y != 0))
    MaskedTensor(  1.0000, True)

Indeed, the index of the minimum argument when ignoring the 0's is the 1 in index 1.

:class:`MaskedTensor` can also support reductions when the data is fully masked out, which is equivalent
to the case above when the data Tensor is completely ``nan``. ``nanmean`` would return ``nan``
(an ambiguous return value), while MaskedTensor would more accurately indicate a masked out result.

    >>> x = torch.empty(16).fill_(float('nan'))
    >>> x
    tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    >>> torch.nanmean(x)
    tensor(nan)
    >>> torch.mean(masked_tensor(x, ~torch.isnan(x)))
    MaskedTensor(--, False)

This is a similar problem to safe softmax where `0/0 = nan` when what we really want is an undefined value.
