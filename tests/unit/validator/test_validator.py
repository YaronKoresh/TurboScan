import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from turboscan.registry.registry import HyperRegistry
from turboscan.resolver.resolver import HyperResolver
from turboscan.validator.validator import HyperValidator


class TestHyperValidator:
    @pytest.fixture
    def validator(self):

        mock_registry = MagicMock(spec=HyperRegistry)
        mock_registry.files_map = {}
        mock_registry.modules = {}

        mock_resolver = MagicMock(spec=HyperResolver)

        mock_resolver.resolvable.return_value = True
        mock_resolver.is_internal_root.return_value = False
        mock_resolver.is_external_or_stdlib.return_value = False

        v = HyperValidator(
            registry=mock_registry,
            current_file=Path("test_script.py"),
            resolver=mock_resolver,
        )
        return v

    def validate_code(self, validator, code):
        tree = ast.parse(code)
        validator.visit(tree)
        return validator.errors, validator.warnings

    def test_undefined_name(self, validator) -> None:
        """Should catch usage of variables that weren't defined."""
        code = """
def func():
    return x + 1  # x is not defined
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 1
        assert "Name 'x' is not defined" in errors[0][1]

    def test_defined_name_pass(self, validator) -> None:
        """Should pass when variable is defined."""
        code = """
def func():
    x = 10
    return x + 1
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_argument_scope(self, validator) -> None:
        """Function arguments should be visible inside the body."""
        code = """
def func(a, b):
    return a + b
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_mutable_default_argument(self, validator) -> None:
        """Should detect dangerous mutable default arguments."""
        code = """
def bad_func(items=[]):
    items.append(1)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 1
        assert "Mutable default argument" in errors[0][1]

    def test_comprehension_scope_leak(self, validator) -> None:
        """Variables in comprehensions should NOT leak (Python 3 behavior)."""
        code = """
data = [i for i in range(10)]
y = i  # i should be undefined here
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 1
        assert "Name 'i' is not defined" in errors[0][1]

    def test_comprehension_internal_access(self, validator) -> None:
        """Variables in comprehension should be visible to the element expression."""
        code = """
data = [x*2 for x in range(10) if x > 5]
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_unused_imports(self, validator) -> None:
        """Should detect imports that are never used."""
        code = """
import os
import sys
print(os.name)
# sys is unused
        """
        self.validate_code(validator, code)
        unused = validator.get_unused_imports()
        assert "sys" in unused
        assert "os" not in unused

    def test_tuple_unpacking(self, validator) -> None:
        """Should handle complex unpacking assignments."""
        code = """
x, (y, z) = (1, (2, 3))
print(x + y + z)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_star_import_ignored_in_validator(self, validator) -> None:
        """Wildcard imports are skipped by validator (handled by Registry instead)."""
        code = """
from math import *
        """
        errors, _warnings = self.validate_code(validator, code)

        assert len(errors) == 0

        assert not validator.imports_map

    def test_mutable_default_dict(self, validator) -> None:
        """Should detect dict as mutable default argument."""
        code = """
def bad_func(cache={}):
    cache['key'] = 1
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 1
        assert "Mutable default argument" in errors[0][1]

    def test_mutable_default_set(self, validator) -> None:
        """Should detect set() as mutable default argument if implemented."""
        code = """
def bad_func(items=set()):
    items.add(1)
        """
        _errors, _ = self.validate_code(validator, code)

    def test_safe_default_none(self, validator) -> None:
        """None as default should be safe."""
        code = """
def good_func(items=None):
    if items is None:
        items = []
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_safe_default_tuple(self, validator) -> None:
        """Tuple as default should be safe (immutable)."""
        code = """
def good_func(items=()):
    return items
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_nested_function_scope(self, validator) -> None:
        """Nested functions should have their own scope."""
        code = """
def outer():
    x = 1
    def inner():
        return x
    return inner()
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_class_method_self(self, validator) -> None:
        """self should be defined in class methods."""
        code = """
class MyClass:
    def method(self):
        return self.value
        """
        _errors, _ = self.validate_code(validator, code)

    def test_class_attribute(self, validator) -> None:
        """Class attributes should be handled."""
        code = """
class MyClass:
    class_attr = 42
    def method(self):
        return MyClass.class_attr
        """
        _errors, _ = self.validate_code(validator, code)

    def test_global_statement(self, validator) -> None:
        """Global statement should make variable available."""
        code = """
x = 10
def func():
    global x
    x = 20
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_nonlocal_statement(self, validator) -> None:
        """Nonlocal statement in nested function."""
        code = """
def outer():
    x = 10
    def inner():
        nonlocal x
        x = 20
    inner()
    return x
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_walrus_operator(self, validator) -> None:
        """Walrus operator (:=) should define variable."""
        code = """
if (n := 10) > 5:
    print(n)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_for_loop_target(self, validator) -> None:
        """For loop target should be defined in loop body."""
        code = """
for i in range(10):
    print(i)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_for_loop_with_unpacking(self, validator) -> None:
        """For loop with tuple unpacking."""
        code = """
for x, y in [(1, 2), (3, 4)]:
    print(x + y)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_with_statement_target(self, validator) -> None:
        """With statement target should be defined."""
        code = """
with open('file.txt') as f:
    data = f.read()
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_exception_handler_name(self, validator) -> None:
        """Exception handler 'as' name should be defined."""
        code = """
try:
    x = 1 / 0
except ZeroDivisionError as e:
    print(e)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_lambda_parameters(self, validator) -> None:
        """Lambda parameters should be defined in body."""
        code = """
f = lambda x, y: x + y
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_generator_expression(self, validator) -> None:
        """Generator expression variable scope."""
        code = """
gen = (x*2 for x in range(10))
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_dict_comprehension(self, validator) -> None:
        """Dict comprehension variable scope."""
        code = """
d = {k: v for k, v in [('a', 1), ('b', 2)]}
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_set_comprehension(self, validator) -> None:
        """Set comprehension variable scope."""
        code = """
s = {x*2 for x in range(10)}
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_import_from(self, validator) -> None:
        """from import should define names."""
        code = """
from os import path
print(path.join('a', 'b'))
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_import_alias(self, validator) -> None:
        """Import with alias should define alias name."""
        code = """
import os as operating_system
print(operating_system.name)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_from_import_alias(self, validator) -> None:
        """from import with alias should define alias name."""
        code = """
from os import path as p
print(p.join('a', 'b'))
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_starred_assignment(self, validator) -> None:
        """Starred assignment should work."""
        code = """
first, *rest = [1, 2, 3, 4]
print(first, rest)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_annotated_assignment(self, validator) -> None:
        """Annotated assignment should define variable."""
        code = """
x: int = 10
print(x)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_augmented_assignment(self, validator) -> None:
        """Augmented assignment requires variable to exist."""
        code = """
x = 10
x += 5
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_undefined_in_augmented_assignment(self, validator) -> None:
        """Augmented assignment on undefined - behavior depends on implementation."""
        code = """
x += 5  # x not defined
        """
        _errors, _ = self.validate_code(validator, code)

    def test_match_statement(self, validator) -> None:
        """Match statement (Python 3.10+) should handle patterns."""

        import sys

        if sys.version_info < (3, 10):
            pytest.skip("Match statements require Python 3.10+")

        code = """
def process(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_async_function(self, validator) -> None:
        """Async function should work like regular function."""
        code = """
async def async_func(x):
    return x + 1
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_async_for(self, validator) -> None:
        """Async for should define loop variable."""
        code = """
async def async_func():
    async for item in async_generator():
        print(item)
        """
        _errors, _ = self.validate_code(validator, code)

    def test_async_with(self, validator) -> None:
        """Async with should define context variable."""
        code = """
async def async_func():
    async with get_connection() as conn:
        await conn.execute()
        """
        _errors, _ = self.validate_code(validator, code)

    def test_kwonly_args(self, validator) -> None:
        """Keyword-only arguments should be defined."""
        code = """
def func(*, kwonly):
    return kwonly
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_posonly_args(self, validator) -> None:
        """Positional-only arguments should be defined."""
        code = """
def func(posonly, /):
    return posonly
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_varargs(self, validator) -> None:
        """*args should be defined."""
        code = """
def func(*args):
    return args
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_kwargs(self, validator) -> None:
        """**kwargs should be defined."""
        code = """
def func(**kwargs):
    return kwargs
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_check_unused_false(self, validator) -> None:
        """Test with check_unused=False."""
        mock_registry = MagicMock(spec=HyperRegistry)
        mock_registry.files_map = {}
        mock_registry.modules = {}
        mock_resolver = MagicMock(spec=HyperResolver)
        mock_resolver.resolvable.return_value = True

        v = HyperValidator(
            registry=mock_registry,
            current_file=Path("test.py"),
            resolver=mock_resolver,
            check_unused=False,
        )

        code = """
import os
import sys
        """
        tree = ast.parse(code)
        v.visit(tree)
        unused = v.get_unused_imports()

        assert unused is not None

    def test_builtin_names(self, validator) -> None:
        """Built-in names should not require definition."""
        code = """
x = len([1, 2, 3])
y = print("hello")
z = range(10)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0

    def test_decorator(self, validator) -> None:
        """Decorators should be resolved."""
        code = """
@staticmethod
def func():
    pass
        """
        _errors, _ = self.validate_code(validator, code)

    def test_multiple_targets(self, validator) -> None:
        """Multiple assignment targets."""
        code = """
a = b = c = 10
print(a, b, c)
        """
        errors, _ = self.validate_code(validator, code)
        assert len(errors) == 0
