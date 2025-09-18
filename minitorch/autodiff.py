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

def topological_sort(variable: Variable) -> Iterable[Variable]:
    
    ordered = []
    seen = set()
    
    def recursive_helper(node: Variable, is_root: bool = False):

        # track the already seen nodes and skip them 
        if node.unique_id in seen:
            return
        seen.add(node.unique_id)
        
        # visit parents even if this node is "constant"
        for parent in node.parents:
            recursive_helper(parent, False)
        
        if is_root or not node.is_constant():
            ordered.append(node)  
    
    recursive_helper(variable, True)
    
    # Reverse the list to get correct order for backpropagation
    ordered.reverse()
    
    return ordered


def backpropagate(variable: Variable, deriv: Any) -> None:

    topo = list(topological_sort(variable))
    # Initialize all nodes to 0
    grads: dict[int, Any] = {node.unique_id: 0 for node in topo}
    # Set the root gradient
    grads[variable.unique_id] = deriv

    for node in topo:
        g = grads.get(node.unique_id)
        if g is not None:
            if node.is_leaf():
                node.accumulate_derivative(g)
            else:
                for parent, contrib in node.chain_rule(g):
                    # Only update gradients for non-constant nodes
                    if not parent.is_constant():
                        id = parent.unique_id
                        grads[id] = grads[id] + contrib


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
