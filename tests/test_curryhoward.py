"""Tests for the Curry-Howard Logic Parser."""
from aletheia.data.curryhoward import (
    map_type_to_proposition,
    extract_python_functions,
    build_trg,
    trg_to_json,
)


def test_map_basic_types():
    assert map_type_to_proposition("int") == "ℤ"
    assert map_type_to_proposition("bool") == "Bool"
    assert map_type_to_proposition("None") == "⊤"


def test_map_function_type():
    result = map_type_to_proposition("int -> bool")
    assert "⊃" in result


def test_map_product_type():
    result = map_type_to_proposition("(int, str)")
    assert "∧" in result


def test_map_sum_type():
    result = map_type_to_proposition("int | str")
    assert "∨" in result


def test_map_generic_type():
    result = map_type_to_proposition("List[int]")
    assert "ℤ" in result


def test_extract_python_functions():
    code = '''
def add(x: int, y: int) -> int:
    return x + y

def greet(name: str) -> None:
    print(name)
'''
    funcs = extract_python_functions(code)
    assert len(funcs) == 2
    assert funcs[0]["name"] == "add"
    assert funcs[0]["return_type"] == "int"
    assert len(funcs[0]["args"]) == 2


def test_build_trg_valid():
    code = '''
def transform(x: int) -> str:
    return str(x)
'''
    trg = build_trg(code, language="python", source_file="test.py")
    assert trg.valid
    assert len(trg.term_nodes) == 1
    assert len(trg.type_nodes) == 2  # int input + str output
    assert len(trg.edges) == 2       # input edge + output edge


def test_build_trg_no_functions():
    code = "x = 42\ny = x + 1"
    trg = build_trg(code, language="python")
    assert trg.valid  # empty but valid
    assert len(trg.term_nodes) == 0


def test_trg_serialization():
    code = "def f(x: int) -> bool:\n    return x > 0"
    trg = build_trg(code, language="python")
    json_str = trg_to_json(trg)
    assert '"valid": true' in json_str
    assert "term_f" in json_str


def test_reject_invalid_syntax():
    code = "def broken("  # syntax error
    funcs = extract_python_functions(code)
    assert len(funcs) == 0


def test_unsupported_language():
    """Unsupported language returns empty TRG (with warning)."""
    trg = build_trg("fn main() {}", language="rust")
    assert trg.valid
    assert len(trg.term_nodes) == 0
