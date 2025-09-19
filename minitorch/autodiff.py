from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass

def ensure_tensor(g, like_tensor):
    from .tensor import Tensor  # adjust import to your layout

    if isinstance(g, (int, float)):
        if g == 0:
            return like_tensor.zeros_like()
        # scalar non-zero: create a scalar tensor and broadcast later as needed
        return Tensor([g], backend=like_tensor.backend)

    return g
def topological_sort(variable):
    seen = set()
    ordered = []

    def recursive_helper(node, is_root):
        if id(node) in seen:
            return
        seen.add(id(node))

        has_history = getattr(node, "history", None) is not None
        if not has_history:
            # append every leaf, root or not
            ordered.append(node)
            return

        for parent in node.parents:
            recursive_helper(parent, False)
        ordered.append(node)

    recursive_helper(variable, True)
    return ordered

def backpropagate(variable: Variable, deriv: Any) -> None:
    topo = list(topological_sort(variable))

    grads: dict[int, Any] = {}
    seed = ensure_tensor(deriv, variable)
    grads[variable.unique_id] = seed

    for node in topo:
        g = grads.get(node.unique_id, 0)
        g = ensure_tensor(g, node)

        # Treat nodes with no history as leaves/consts: don't call chain_rule.
        has_history = getattr(node, "history", None) is not None
        if not has_history:
            # If it's a true leaf (user tensor with requires_grad), accumulate.
            if node.is_leaf():
                node.accumulate_derivative(g)
            # Either way, do NOT recurse.
            continue

        # Regular non-leaf case:
        for parent, contrib in node.chain_rule(g):
            if parent.is_constant():
                continue
            contrib = ensure_tensor(contrib, parent)
            prev = grads.get(parent.unique_id, 0)
            prev = ensure_tensor(prev, parent)
            # (optional) if you have reductions, do: contrib = sum_to_shape(contrib, parent.shape)
            grads[parent.unique_id] = prev + contrib




@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
