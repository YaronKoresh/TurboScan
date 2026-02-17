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
