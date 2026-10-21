import ast

from turboscan.ast_transform.mutation_detector import MutationDetector


def detect_mutation(code):
    tree = ast.parse(code)
    func_def = tree.body[0]
    detector = MutationDetector(func_def)
    detector.visit(func_def)
    return detector.is_pure


def test_pure_function() -> None:
    code = """
def safe_func(self, x):
    return x + 1
    """
    assert detect_mutation(code) is True


def test_mutation_via_assign() -> None:
    code = """
def unsafe_func(self, x):
    self.val = x  # Mutation!
    """
    assert detect_mutation(code) is False


def test_mutation_via_aug_assign() -> None:
    code = """
def unsafe_func(self, x):
    self.count += 1  # Mutation!
    """
    assert detect_mutation(code) is False


def test_local_assign_is_safe() -> None:
    code = """
def safe_func(self, x):
    temp = self.val + x  # Local variable, safe
    return temp
    """
    assert detect_mutation(code) is True


import pytest


@pytest.mark.parametrize(
    "body",
    [
        "self.a, self.b = x, x",
        "self.a, *self.b = x",
        "self.d[x] = 1",
        "self.d[x] += 1",
        "self.v: int = x",
        "del self.v",
        "self.a.b = x",
        "self.items.append(x)",
        "for self.i in range(x): pass",
    ],
)
def test_mutation_forms_are_detected(body) -> None:
    code = f"def m(self, x):\n    {body}\n"
    assert detect_mutation(code) is False


def test_annotation_without_value_is_safe() -> None:
    assert detect_mutation("def m(self, x):\n    self.v: int\n") is True


def test_rebinding_self_and_locals_is_safe() -> None:
    code = "def m(self, x):\n    y = [x]\n    y[0] = 1\n    y.append(2)\n    return y\n"
    assert detect_mutation(code) is True
