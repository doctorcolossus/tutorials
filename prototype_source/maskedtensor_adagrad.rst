Efficiency of writing "sparse" semantics for Adagrad
====================================================

`Issue 1369 <https://github.com/pytorch/pytorch/issues/1369>`__ discussed the additional lines of code
that were introduce while writing "sparse" semantics for Adagrad.
But really the code doesn't use sparsity as a compression and optimization technique,
it wants to use masked semantics. We worked around this by introducing one-off semantics and operators
that encode this behavior while forcing users to be aware of storage details such as indices and values.

In particular we'll point out when sparsity is used as a semantic extension, i.e. unspecified values are not zero
and when it is just used to compress zeros.
We'll also compare and contrast this with equivalent code written using MaskedTensor.
In the end the code snippets are repeat without additional comments to show the difference in brevity.

Set up
------

.. code-block:: python

    import torch
    from torch.masked.maskedtensor import masked_tensor

    # Some hyperparameters
    eps = 1e-10
    clr = 0.1

    i = torch.tensor([[0, 1, 1], [2, 0, 2]])
    v = torch.tensor([3, 4, 5], dtype=torch.float32)
    grad = torch.sparse_coo_tensor(i, v, [2, 4])

Original sparse implementation
------------------------------

First, let's break down the current implementation of 
`Adagrad (functional) <https://github.com/pytorch/pytorch/blob/6c2f235d368b697072699e5ca9485fd97d0b9bcc/torch/optim/_functional.py#L16-L51>`__
in PyTorch currently:

.. code-block:: python
    :linenos:

    def _make_sparse(grad, grad_indices, values):
        size = grad.size()
        if grad_indices.numel() == 0 or values.numel() == 0:
            return torch.empty_like(grad)
        return torch.sparse_coo_tensor(grad_indices, values, size)

    # We don't support sparse gradients
    param = torch.arange(8).reshape(2, 4).float()
    state_sum = torch.full_like(param, 0.5)  # initial value for state sum

    grad = grad.coalesce()  # the update is non-linear so indices must be unique
    grad_indices = grad._indices()
    grad_values = grad._values()
    # pow(2) has the same semantics for both sparse and dense memory layouts since 0^2 is zero
    state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))

    # We take care to make std sparse, even though state_sum clearly is not.
    # This means that we're only applying the gradient to parts of the state_sum
    # for which it is specified. This even drives the point home a lot more that
    # the passed gradient is not sparse, but masked.
    std = state_sum.sparse_mask(grad)

    # We currently dodge all these concerns using the private method values.
    std_values = std._values().sqrt_().add_(eps)

    # We currently don't support div for sparse Tensors because zero / zero is
    # not well defined. For a MaskedTensor undefined / undefined is undefined.
    param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)

Line 24 is where we have a very important divergence.

The addition of eps should technically be applied to all values but instead is only applied to specified values.
Here we're using sparsity as a semantic extension and to enforce a certain pattern of defined and undefined values.
If parts of the values of the gradient are zero, they are still included if materialized even though they
could be compressed by other sparse storage layouts.
This is technically quite brittle even though someone could argue that eps is always very small.

Moreover, an implementation `add_` for sparsity as a storage layout and compression scheme should cause densification,
but we force it not to for performance.
For this one-off case it is fine.. until we want to introduce new compression schemes
such as CSR, BSR, or 2:4 block sparsity. We'll then need to introduce separate Tensor types for each
and write variations for gradients compressed using different storage formats, which is inconvenient.

MaskedTensor sparse implementation
----------------------------------

We've been conflating sparsity as an optimization with sparsity as a semantic extension to PyTorch.
MaskedTensor proposes to call the semantic extension through sparsity masked. 
Currently we can't have dense semantics with sparse storage or masked semantics with dense storage, while 
MaskedTensor fixes that because it separates the storage from the semantics.

Consider the above example using a masked gradient:

.. code-block:: python

    # Create an entirely new set of parameters to avoid errors
    param2 = torch.arange(8).reshape(2, 4).float()
    state_sum2 = torch.full_like(param, 0.5)  # initial value for state sum

    mask = (grad.to_dense() != 0).to_sparse()
    masked_grad = masked_tensor(grad, mask)

    state_sum2 = state_sum2 + masked_grad.pow(2).get_data()
    std2 = masked_tensor(state_sum2.to_sparse(), mask)

    # We can add support for in-place operations later. Notice how this doesn't
    # need to access any storage internals and is in general a lot shorter
    std2 = std2.sqrt().add(eps)

    param2 = param2.add((masked_grad / std2).data(), alpha=-clr)

Note that the implementations look quite similar but just shorter. Much of the boilerplate code
around `_make_sparse` (and needing to have a separate implementation per layout) is handled
for you with :class:`MaskedTensor`.

At this point, let's print both this version and original version for easier comparison:

        >>> state_sum
        >>> state_sum
        tensor([[ 0.5000,  0.5000,  9.5000,  0.5000],
                [16.5000,  0.5000, 25.5000,  0.5000]])
        >>> state_sum2
        tensor([[ 0.5000,  0.5000,  9.5000,  0.5000],
                [16.5000,  0.5000, 25.5000,  0.5000]])
        >>> std
        tensor(indices=tensor([[0, 1, 1],
                            [2, 0, 2]]),
            values=tensor([3.0822, 4.0620, 5.0498]),
            size=(2, 4), nnz=3, layout=torch.sparse_coo)
        >>> std2
        MaskedTensor(
        [
            [      --,       --,   3.0822,       --],
            [  4.0620,       --,   5.0498,       --]
        ]
        )
        >>> param
        tensor([[0.0000, 1.0000, 1.9027, 3.0000],
                [3.9015, 5.0000, 5.9010, 7.0000]])
        >>> param2
        tensor([[0.0000, 1.0000, 1.9027, 3.0000],
                [3.9015, 5.0000, 5.9010, 7.0000]])

which proves that the two implementations are indeed the same.

Conclusion: Difference in Code
------------------------------

For reference, this is the regular, dense code path without masked gradients or sparsity:

.. code-block:: python

    state_sum.addcmul_(grad, grad, value=1)
    std = state_sum.sqrt().add_(eps)
    param.addcdiv_(grad, std, value=-clr)
 
The vanilla tensor implementation for sparse is:

.. code-block:: python

    grad = grad.coalesce()  # the update is non-linear so indices must be unique
    grad_indices = grad._indices()
    grad_values = grad._values()

    state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))   # a different _make_sparse per layout
    std = state_sum.sparse_mask(grad)
    std_values = std._values().sqrt_().add_(eps)
    param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)

while :class:`MaskedTensor` minimizes the code to the snippet:

.. code-block:: python

    state_sum2 = state_sum2 + masked_grad.pow(2).data()
    std2 = masked_tensor(state_sum2.to_sparse(), mask)
    std2 = std2.sqrt().add(eps)
    param2 = param2.add((masked_grad / std2).data(), alpha=-clr)

One major goal of :class:`MaskedTensor` is to enable sparsity semantics and applications, such as this one.
To learn more about using sparsity, you can find
[this MaskedTensor sparsity tutorial](https://pytorch.org/tutorials/prototype/maskedtensor_sparsity.html).
Currently, COO and CSR sparse layouts are supported, though there are immediate plans to add more.
