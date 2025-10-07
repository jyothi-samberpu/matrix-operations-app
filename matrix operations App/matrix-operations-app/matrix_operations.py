"""
Matrix Operations App (CLI)
- Addition, subtraction, multiplication, inversion using NumPy
- Multiple input modes: manual row-by-row, paste, random, file
- Pretty printing and optional save/visualization
"""

import numpy as np

#  visualization
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def pretty_print(mat: np.ndarray, precision: int = 3):
    """Print a 2D numpy array in aligned columns."""
    mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    rows, cols = mat.shape
    fmt = f"{{:>{8}.{precision}f}}"
    for r in range(rows):
        print(" ".join(fmt.format(float(mat[r, c])) for c in range(cols)))
    print()  # blank line


def parse_numbers_from_line(line: str):
    """Parse a line of numbers separated by spaces or commas."""
    # replace commas with spaces, split on whitespace
    parts = line.replace(",", " ").split()
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            # ignore non-numeric tokens
            pass
    return nums


def input_matrix(name="A"):
    """Interactive input of a matrix from the user."""
    while True:
        try:
            rows = int(input(f"Number of rows for matrix {name}: ").strip())
            cols = int(input(f"Number of columns for matrix {name}: ").strip())
            if rows <= 0 or cols <= 0:
                print("Rows and columns must be positive integers. Try again.")
                continue
            break
        except ValueError:
            print("Invalid integer. Try again.")

    print("Choose entry mode:")
    print("  1) Enter row-by-row")
    print("  2) Paste all numbers (space/comma separated)")
    print("  3) Fill with random numbers")
    mode = input("Mode (1/2/3) [1]: ").strip() or "1"

    if mode == "3":
        mat = np.random.rand(rows, cols)
        return mat.astype(float)

    if mode == "2":
        print(f"Paste {rows*cols} numbers (space or comma separated), then press Enter:")
        s = input().strip()
        nums = parse_numbers_from_line(s)
        if len(nums) != rows * cols:
            print(f"Expected {rows*cols} numbers but got {len(nums)}. Falling back to row-by-row input.")
        else:
            return np.array(nums, dtype=float).reshape(rows, cols)

    # row-by-row input (fallback or selected)
    mat = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        while True:
            line = input(f"Row {r+1} (enter {cols} numbers separated by spaces or commas): ").strip()
            nums = parse_numbers_from_line(line)
            if len(nums) != cols:
                print(f"Expected {cols} numbers but got {len(nums)}. Try again.")
                continue
            mat[r, :] = nums
            break
    return mat


def load_matrix_from_file():
    """Load matrix from a whitespace/comma separated file using numpy.loadtxt."""
    path = input("Enter file path (CSV or whitespace-separated): ").strip()
    try:
        mat = np.loadtxt(path, delimiter=",")
        # If file had a single row it might be 1D; force 2D
        mat = np.atleast_2d(mat).astype(float)
        print(f"Loaded matrix from {path}: shape {mat.shape}")
        return mat
    except Exception as e:
        print("Failed to load file:", e)
        return None


def choose_matrix(name="A"):
    """Let the user choose how to supply a matrix."""
    print(f"Options to provide matrix {name}:")
    print("  1) Type/paste interactively")
    print("  2) Load from file")
    print("  3) Random matrix")
    choice = input("Choose (1/2/3) [1]: ").strip() or "1"
    if choice == "2":
        m = load_matrix_from_file()
        if m is None:
            print("Falling back to interactive input.")
            return input_matrix(name)
        return m
    elif choice == "3":
        rows = int(input("rows: "))
        cols = int(input("cols: "))
        scale = float(input("scale (max value) [1.0]: ") or 1.0)
        return np.random.rand(rows, cols) * scale
    else:
        return input_matrix(name)


def add_matrices(A, B):
    if A.shape != B.shape:
        raise ValueError("Addition requires matrices of the same shape.")
    return A + B


def sub_matrices(A, B):
    if A.shape != B.shape:
        raise ValueError("Subtraction requires matrices of the same shape.")
    return A - B


def mul_matrices(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix multiplication requires inner dimensions to match (A.cols == B.rows).")
    return A.dot(B)


def invert_matrix(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Only square matrices can be inverted.")
    cond = np.linalg.cond(A)
    if cond > 1e12:
        print(f"Warning: condition number is very large ({cond:.3e}) — inverse may be inaccurate or matrix nearly singular.")
    try:
        inv = np.linalg.inv(A)
        return inv
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted.")


def save_matrix(mat):
    path = input("Enter filename to save (CSV) or leave blank to skip: ").strip()
    if not path:
        return
    try:
        np.savetxt(path, mat, delimiter=",", fmt="%.8g")
        print(f"Saved to {path}")
    except Exception as e:
        print("Failed to save:", e)


def visualize_matrix(mat, title="Matrix"):
    if not HAS_MPL:
        print("matplotlib not available. Install it with 'pip install matplotlib' to visualize.")
        return
    plt.figure(figsize=(4, 4))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


def main():
    print("Matrix Operations App — NumPy\n")
    while True:
        print("Select operation:")
        print("  1) Add (A + B)")
        print("  2) Subtract (A - B)")
        print("  3) Multiply (A * B)")
        print("  4) Inverse (A^-1)")
        print("  5) Demo (sample matrices)")
        print("  q) Quit")
        choice = input("Choice: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break

        if choice == "5":
            print("\nDemo with A=[[1,2],[3,4]] and B=[[5,6],[7,8]]")
            A = np.array([[1, 2], [3, 4]], dtype=float)
            B = np.array([[5, 6], [7, 8]], dtype=float)
            print("Matrix A:")
            pretty_print(A)
            print("Matrix B:")
            pretty_print(B)
            print("A + B:")
            pretty_print(add_matrices(A, B))
            print("A - B:")
            pretty_print(sub_matrices(A, B))
            print("A * B:")
            pretty_print(mul_matrices(A, B))
            print("Inverse of A:")
            try:
                pretty_print(invert_matrix(A))
            except ValueError as e:
                print("Inverse error:", e)
            continue

        try:
            if choice in ("1", "2", "3"):
                A = choose_matrix("A")
                print("\nMatrix A:")
                pretty_print(A)
                B = choose_matrix("B")
                print("\nMatrix B:")
                pretty_print(B)

                if choice == "1":
                    res = add_matrices(A, B)
                    print("Result (A + B):")
                    pretty_print(res)
                elif choice == "2":
                    res = sub_matrices(A, B)
                    print("Result (A - B):")
                    pretty_print(res)
                else:
                    res = mul_matrices(A, B)
                    print("Result (A * B):")
                    pretty_print(res)

                save_matrix(res)
                if input("Visualize result? (y/N): ").strip().lower() == "y":
                    visualize_matrix(res, title="Result")

            elif choice == "4":
                A = choose_matrix("A")
                print("\nMatrix A:")
                pretty_print(A)
                try:
                    inv = invert_matrix(A)
                    print("Inverse (A^-1):")
                    pretty_print(inv)
                    save_matrix(inv)
                    if input("Visualize inverse? (y/N): ").strip().lower() == "y":
                        visualize_matrix(inv, title="Inverse")
                except ValueError as e:
                    print("Error:", e)

            else:
                print("Unknown choice. Try again.")
        except Exception as e:
            print("Operation failed:", e)

if __name__ == "__main__":
    main()
