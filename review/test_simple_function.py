import pytest
import simple_function as sf

'''
Your test files must be named with "test" in the beginning

To run in terminal:

pytest
pytest -vv
'''

def test_add():
    result = sf.add(1, 2)
    assert result == 3

def test_divide():
    result = sf.divide(9, 3)
    assert result == 3

def test_subtract():
    result = sf.subtract(9, 3)
    assert result == 6

def test_multiply():
    result = sf.multiply(3, 3)
    assert result == 9
