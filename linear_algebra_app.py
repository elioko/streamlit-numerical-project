import streamlit as st
import numpy as np

# ----------------- توابع -----------------

def create_empty_matrix(rows, cols):
    """ایجاد ماتریس خالی با ابعاد مشخص."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def parse_matrix_from_table(table):
    """تبدیل داده‌های جدول واردشده توسط کاربر به numpy array."""
    return np.array([[float(cell) for cell in row] for row in table])

def jacobi_method_fixed_iterations(A, b, x0, num_iterations):
    n = len(A)
    x = x0.copy()
    for _ in range(num_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        x = x_new
    return x

def gauss_seidel_method_fixed_iterations(A, b, x0, num_iterations):
    n = len(A)
    x = x0.copy()
    for _ in range(num_iterations):
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
    return x

def sor_method_fixed_iterations(A, b, x0, omega, num_iterations):
    n = len(A)
    x = x0.copy()
    for _ in range(num_iterations):
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (1 - omega) * x[i] + omega * (b[i] - s1 - s2) / A[i][i]
    return x

def gram_schmidt(vectors):
    orthogonal = []
    for v in vectors:
        for u in orthogonal:
            v -= np.dot(v, u) / np.dot(u, u) * u
        if np.linalg.norm(v) > 1e-10:
            orthogonal.append(v)
    return len(orthogonal) == len(vectors)

def qr_decomposition(A):
    Q, R = np.linalg.qr(A)
    return Q, R

def power_method(A, x0, num_iterations):
    x = x0
    for _ in range(num_iterations):
        x = np.dot(A, x)
        x /= np.linalg.norm(x)
    eigenvalue = np.dot(x, np.dot(A, x)) / np.dot(x, x)
    return eigenvalue, x

def householder_reduction(A):
    n = A.shape[0]
    eigenvalues = []
    for _ in range(n - 1):
        x = A[:, 0]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = (x - e) / np.linalg.norm(x - e)
        H = np.eye(n) - 2 * np.outer(u, u)
        A = H @ A @ H.T
        eigenvalues.append(A[0, 0])
        A = A[1:, 1:]
    eigenvalues.append(A[0, 0])
    return eigenvalues

def largest_eigenvalue(A):
    return np.linalg.eigvals(A).max().real

st.title("Numerical Linear Algebra App")
tabs = st.tabs(["Iterative Methods", "Gram-Schmidt", "QR Decomposition", "Power Method"])

with tabs[0]:
    st.subheader("Iterative Methods")
    rows = st.number_input("Number of rows/columns (n):", min_value=1, value=3, step=1)
    st.write("### Enter Matrix (A):")
    matrix_table = st.data_editor(create_empty_matrix(rows, rows), key="iterative_matrix")
    st.write("### Enter Vector (b):")
    b_vector = st.data_editor([[0.0] for _ in range(rows)], key="iterative_b")
    st.write("### Enter Initial Guess (x0):")
    x0_vector = st.data_editor([[0.0] for _ in range(rows)], key="iterative_x0")
    num_iterations = st.number_input("Number of iterations:", min_value=1, value=10, step=1)
    omega = st.number_input("SOR Relaxation Parameter (omega):", min_value=0.1, value=1.25, step=0.05)

    if st.button("Run Iterative Methods"):
        A = parse_matrix_from_table(matrix_table)
        b = np.array([row[0] for row in b_vector])
        x0 = np.array([row[0] for row in x0_vector])
        jacobi_result = jacobi_method_fixed_iterations(A, b, x0, num_iterations)
        gs_result = gauss_seidel_method_fixed_iterations(A, b, x0, num_iterations)
        sor_result = sor_method_fixed_iterations(A, b, x0, omega, num_iterations)
        st.write("### Jacobi Method", jacobi_result)
        st.write("### Gauss-Seidel Method", gs_result)
        st.write("### SOR Method", sor_result)

with tabs[1]:
    st.subheader("Gram-Schmidt Process")
    num_vectors = st.number_input("Number of vectors:", min_value=1, value=2, step=1)
    num_rows = st.number_input("Number of rows in each vector:", min_value=1, value=3, step=1, key="gram_schmidt_rows")

    vectors = []
    for i in range(num_vectors):
        st.write(f"### Enter Vector {i + 1}:")
        vector = st.data_editor([[0.0] for _ in range(num_rows)], key=f"gram_schmidt_vector_{i}")
        vectors.append([row[0] for row in vector])

    if st.button("Run Gram-Schmidt"):
        vectors_np = np.array(vectors)
        is_independent = gram_schmidt(vectors_np)
        if is_independent:
            st.success("The vectors are linearly independent.")
        else:
            st.error("The vectors are not linearly independent.")

with tabs[2]:
    st.subheader("QR Decomposition")
    rows = st.number_input("Number of rows:", min_value=1, value=3, step=1, key="qr_rows")
    cols = st.number_input("Number of columns:", min_value=1, value=3, step=1, key="qr_cols")
    st.write("### Enter Matrix (A):")
    matrix_table = st.data_editor(create_empty_matrix(rows, cols), key="qr_matrix")

    if st.button("Run QR Decomposition"):
        A = parse_matrix_from_table(matrix_table)
        Q, R = qr_decomposition(A)
        st.write("### Q Matrix", Q)
        st.write("### R Matrix", R)

with tabs[3]:
    st.subheader("Power Method")
    rows = st.number_input("Matrix size (n):", min_value=1, value=3, step=1, key="power_rows")
    st.write("### Enter Matrix (A):")
    matrix_table = st.data_editor(create_empty_matrix(rows, rows), key="power_matrix")
    st.write("### Enter Initial Guess (x0):")
    x0_vector = st.data_editor([[0.0] for _ in range(rows)], key="power_x0")
    num_iterations = st.number_input("Number of iterations:", min_value=1, value=10, step=1, key="power_iterations")

    if st.button("Run Power Method"):
        A = parse_matrix_from_table(matrix_table)
        x0 = np.array([row[0] for row in x0_vector])
        largest_eigen = largest_eigenvalue(A)
        eigenvalue, eigenvector = power_method(A, x0, num_iterations)
        st.write("### Largest Real Eigenvalue (Exact)", largest_eigen)
        st.write("### Largest Eigenvalue (Power Method)", eigenvalue)
        st.write("### Corresponding Eigenvector", eigenvector)
