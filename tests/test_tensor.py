from typing import Callable, Iterable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data, lists, permutations

from minitorch import MathTestVariable, Tensor, grad_check, tensor

from .strategies import assert_close, small_floats
from .tensor_strategies import shaped_tensors, tensors

import numpy as np
from minitorch.tensor_data import Shape, Storage, Strides


from minitorch.tensor_ops import tensor_map, tensor_zip, tensor_reduce

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()


@given(lists(small_floats, min_size=1))
def test_create(t1: List[float]) -> None:
    """Test the ability to create an index a 1D Tensor"""
    t2 = tensor(t1)
    for i in range(len(t1)):
        assert t1[i] == t2[i]


@given(tensors())
@pytest.mark.task2_3
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]], t1: Tensor
) -> None:
    """Test one-arg functions compared to floats"""
    name, base_fn, tensor_fn = fn
    t2 = tensor_fn(t1)
    for ind in t2._tensor.indices():
        assert_close(t2[ind], base_fn(t1[ind]))


@given(shaped_tensors(2))
@pytest.mark.task2_3
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: Tuple[Tensor, Tensor],
) -> None:
    name, base_fn, tensor_fn = fn
    t1, t2 = ts
    t3 = tensor_fn(t1, t2)
    for ind in t3._tensor.indices():
        assert_close(t3[ind], base_fn(t1[ind], t2[ind]))


@given(tensors())
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]], t1: Tensor
) -> None:
    """Test the gradient of a one-arg tensor function"""
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data(), tensors())
@pytest.mark.task2_4
def test_permute(data: DataObject, t1: Tensor) -> None:
    """Test the permute function"""
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a: Tensor) -> Tensor:
        return a.permute(*permutation)

    grad_check(permute, t1)


def test_grad_size() -> None:
    """Test the size of the gradient (from @WannaFy)"""
    a = tensor([1], requires_grad=True)
    b = tensor([[1, 1]], requires_grad=True)

    c = (a * b).sum()

    c.backward()
    assert c.shape == (1,)
    assert a.grad is not None
    assert b.grad is not None
    assert a.shape == a.grad.shape
    assert b.shape == b.grad.shape


@given(tensors())
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", red_arg)
def test_grad_reduce(
    fn: Tuple[str, Callable[[Iterable[float]], float], Callable[[Tensor], Tensor]],
    t1: Tensor,
) -> None:
    """Test the grad of a tensor reduce"""
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(shaped_tensors(2))
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: Tuple[Tensor, Tensor],
) -> None:
    name, _, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)


@given(shaped_tensors(2))
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad_broadcast(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: Tuple[Tensor, Tensor],
) -> None:
    """Test the grad of a two argument function"""
    name, base_fn, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)

    # broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


def test_fromlist() -> None:
    """Test longer from list conversion"""
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t = tensor([[[2, 3, 4], [4, 5, 7]]])
    assert t.shape == (1, 2, 3)


def test_view() -> None:
    """Test view"""
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t2 = t.view(6)
    assert t2.shape == (6,)
    t2 = t2.view(1, 6)
    assert t2.shape == (1, 6)
    t2 = t2.view(6, 1)
    assert t2.shape == (6, 1)
    t2 = t2.view(2, 3)
    assert t.is_close(t2).all().item() == 1.0


@given(tensors())
def test_back_view(t1: Tensor) -> None:
    """Test the graident of view"""

    def view(a: Tensor) -> Tensor:
        a = a.contiguous()
        return a.view(a.size)

    grad_check(view, t1)


@pytest.mark.xfail
def test_permute_view() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t2 = t.permute(1, 0)
    t2.view(6)


@pytest.mark.xfail
def test_index() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t[50, 2]


def test_fromnumpy() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    print(t)
    assert t.shape == (2, 3)
    n = t.to_numpy()
    t2 = tensor(n.tolist())
    for ind in t._tensor.indices():
        assert t[ind] == t2[ind]


# my tests for map
def double_fn(x: float) -> float:
    return x * 2


@pytest.mark.task2_3_1
# 函数定义：将每个元素乘以 2
def test_tensor_map_broadcast() -> None:
    # 初始化存储
    in_storage: Storage = np.array([1.0, 2.0, 3.0])
    out_storage: Storage = np.zeros(6)  # (2, 3) 的输出张量

    # 广播形状
    in_shape: Shape = np.array([1, 3])
    out_shape: Shape = np.array([2, 3])

    # 对应的步幅
    in_strides: Strides = np.array([0, 1])  # 第一维度的步幅为 0，表示广播
    out_strides: Strides = np.array([3, 1])  # 正常的步幅

    # 创建映射函数
    map_fn = tensor_map(double_fn)

    # 应用映射
    map_fn(out_storage, out_shape, out_strides, in_storage, in_shape, in_strides)

    # 期望的输出值
    expected_out_storage = np.array([2.0, 4.0, 6.0, 2.0, 4.0, 6.0], dtype=np.float32)

    # 验证结果是否符合期望
    assert np.array_equal(
        out_storage, expected_out_storage
    ), f"Expected {expected_out_storage}, but got {out_storage}"


# my tests for zip
# 函数定义：将两个元素相加
def add_fn(x: float, y: float) -> float:
    return x + y


@pytest.mark.task2_3_2
def test_tensor_zip_broadcast() -> None:
    # 初始化存储
    a_storage: Storage = np.array(
        [1.0, 2.0, 3.0], dtype=np.float64
    )  # (1, 3) 的输入张量
    b_storage: Storage = np.array(
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64
    )  # (2, 3) 的输入张量
    out_storage: Storage = np.zeros(6)  # (2, 3) 的输出张量

    # 广播形状
    a_shape: Shape = np.array([1, 3])
    b_shape: Shape = np.array([2, 3])
    out_shape: Shape = np.array([2, 3])

    # 对应的步幅
    a_strides: Strides = np.array([0, 1])  # 第一维度的步幅为 0，表示广播
    b_strides: Strides = np.array([3, 1])  # 正常的步幅
    out_strides: Strides = np.array([3, 1])  # 输出张量的步幅

    # 创建 tensor_zip 映射函数
    zip_fn = tensor_zip(add_fn)

    # 应用映射
    zip_fn(
        out_storage,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    )

    # 期望的输出值
    expected_out_storage = np.array([5.0, 7.0, 9.0, 8.0, 10.0, 12.0], dtype=np.float64)

    # 验证结果是否符合期望
    assert np.array_equal(
        out_storage, expected_out_storage
    ), f"Expected {expected_out_storage}, but got {out_storage}"


# my tests for reduce
# 函数定义：将两个元素相加
def sum_fn(x: float, y: float) -> float:
    return x + y


@pytest.mark.task2_3_3
def test_tensor_reduce() -> None:
    # 初始化存储
    a_storage: Storage = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64
    )  # (2, 3) 的输入张量
    out_storage: Storage = np.zeros(2, dtype=np.float64)  # (2, 1) 的输出张量

    # 原始和目标形状
    a_shape: Shape = np.array([2, 3])
    out_shape: Shape = np.array([2, 1])  # reduce 后的形状

    # 对应的步幅
    a_strides: Strides = np.array([3, 1])  # 正常的步幅
    out_strides: Strides = np.array([1, 1])  # 输出张量的步幅

    # 创建 tensor_reduce 函数
    reduce_fn = tensor_reduce(sum_fn)

    # 应用 reduce 函数
    reduce_dim = 1
    reduce_fn(
        out_storage, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim
    )

    # 期望的输出值
    expected_out_storage = np.array([6.0, 15.0], dtype=np.float64)

    # 验证结果是否符合期望
    assert np.array_equal(
        out_storage, expected_out_storage
    ), f"Expected {expected_out_storage}, but got {out_storage}"


@pytest.mark.task2_3_3
def test_tensor_reduce_1() -> None:
    # 初始化存储
    a_storage: Storage = np.array([11.0, 16.0], dtype=np.float64)
    out_storage: Storage = np.zeros(1, dtype=np.float64)

    # 原始和目标形状
    a_shape: Shape = np.array([1, 2])
    out_shape: Shape = np.array([1, 1])  # reduce 后的形状

    # 对应的步幅
    a_strides: Strides = np.array([2, 1])  # 正常的步幅
    out_strides: Strides = np.array([1, 1])  # 输出张量的步幅

    # 创建 tensor_reduce 函数
    reduce_fn = tensor_reduce(sum_fn)

    # 应用 reduce 函数
    reduce_dim = 1
    reduce_fn(
        out_storage, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim
    )

    # 期望的输出值
    expected_out_storage = np.array([27.0], dtype=np.float64)

    # 验证结果是否符合期望
    assert np.array_equal(
        out_storage, expected_out_storage
    ), f"Expected {expected_out_storage}, but got {out_storage}"


# Student Submitted Tests


@pytest.mark.task2_3
def test_reduce_forward_one_dim() -> None:
    # shape (3, 2)
    t = tensor([[2, 3], [4, 6], [5, 7]])

    # here 0 means to reduce the 0th dim, 3 -> nothing
    t_summed = t.sum(0)

    # shape (2)
    t_sum_expected = tensor([[11, 16]])
    assert t_summed.is_close(t_sum_expected).all().item()


@pytest.mark.task2_3
def test_reduce_forward_one_dim_2() -> None:
    # shape (3, 2)
    t = tensor([[2, 3], [4, 6], [5, 7]])

    # here 1 means reduce the 1st dim, 2 -> nothing
    t_summed_2 = t.sum(1)

    # shape (3)
    t_sum_2_expected = tensor([[5], [10], [12]])
    assert t_summed_2.is_close(t_sum_2_expected).all().item()


@pytest.mark.task2_3
def test_reduce_forward_all_dims() -> None:
    # shape (3, 2)
    t = tensor([[2, 3], [4, 6], [5, 7]])

    # reduce all dims, (3 -> 1, 2 -> 1)
    t_summed_all = t.sum()
    print("---------------------- t_summed_all----------------------", t_summed_all)

    # shape (1, 1)
    t_summed_all_expected = tensor([27])

    assert_close(t_summed_all[0], t_summed_all_expected[0])
