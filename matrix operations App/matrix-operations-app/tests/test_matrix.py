import numpy as np

# Test Addition
def test_addition():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = np.add(A, B)
    expected = np.array([[6, 8], [10, 12]])
    assert np.array_equal(result, expected)

# Test Subtraction
def test_subtraction():
    A = np.array([[5, 6], [7, 8]])
    B = np.array([[1, 2], [3, 4]])
    result = np.subtract(A, B)
    expected = np.array([[4, 4], [4, 4]])
    assert np.array_equal(result, expected)

# Test Multiplication
def test_multiplication():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = np.dot(A, B)
    expected = np.array([[19, 22], [43, 50]])
    assert np.array_equal(result, expected)

# Test Inversion
def test_inversion():
    A = np.array([[1, 2], [3, 4]])
    result = np.linalg.inv(A)
    expected = np.array([[-2. , 1. ], [1.5, -0.5]])
    assert np.allclose(result, expected)  # Use allclose for floats
