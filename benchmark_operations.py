
import time
import numpy as np
from scipy.sparse.linalg import spsolve

def run(M_dense, M_sparse, n_nodes, n_repeats_fast=10, n_repeats_slow=1):
    """Benchmark various matrix operations"""
    results = {}
    v = np.random.rand(n_nodes)
    
    # Matrix-vector multiplication - repeat for better timing
    dense_times = []
    for _ in range(n_repeats_fast):
        start_time = time.perf_counter()
        result_dense = M_dense @ v
        dense_times.append(time.perf_counter() - start_time)
    dense_matvec_time = np.mean(dense_times)
    
    sparse_times = []
    for _ in range(n_repeats_fast):
        start_time = time.perf_counter()
        result_sparse = M_sparse @ v
        sparse_times.append(time.perf_counter() - start_time)
    sparse_matvec_time = np.mean(sparse_times)

    print(f"  MatVec - Dense: {dense_matvec_time*1000:.4f}ms, Sparse: {sparse_matvec_time*1000:.4f}ms")
    print(f"  MatVec speedup: {dense_matvec_time/sparse_matvec_time:.1f}x")
    if np.allclose(result_dense, result_sparse):
        print("  MatVec check: dense and sparse matrix give the same result")
    else:
        raise RuntimeError("MatVec check: dense and sparse matrix give different results")    
    
    # Linear system solving - usually slow enough that we don't need many repeats
    dense_solve_times = []
    for _ in range(n_repeats_slow):
        start_time = time.perf_counter()
        solution_dense = np.linalg.solve(M_dense, v)
        dense_solve_times.append(time.perf_counter() - start_time)
    dense_solve_time = np.mean(dense_solve_times)
    
    sparse_solve_times = []
    for _ in range(n_repeats_slow):
        start_time = time.perf_counter()
        solution_sparse = spsolve(M_sparse, v)
        sparse_solve_times.append(time.perf_counter() - start_time)
    sparse_solve_time = np.mean(sparse_solve_times)
    
    print(f"  Solve - Dense: {dense_solve_time:.4f}s, Sparse: {sparse_solve_time:.4f}s")
    print(f"  Solve speedup: {dense_solve_time/sparse_solve_time:.1f}x")
    if np.allclose(solution_dense, solution_sparse):
        print("  Solve check: dense and sparse matrix give the same result")
    else:
        raise RuntimeError("Solve check: dense and sparse matrix give different results")    

    assert np.allclose(solution_dense, solution_sparse), "Solve: matrices give different solutions"
    print("  Solve check: dense and sparse matrix give the same solution")
    
    print(f"  Matrix size: {n_nodes} × {n_nodes}")
    print(f"  Dense matrix memory: {M_dense.nbytes / 1024**2:.1f} MB")
    print(f"  Sparse matrix memory: {(M_sparse.data.nbytes + M_sparse.indices.nbytes + M_sparse.indptr.nbytes) / 1024**2:.1f} MB")
    print(f"  Sparsity: {100 * (1 - M_sparse.nnz / (n_nodes**2)):.1f}%")
    
    results['n_nodes'] = n_nodes
    results['dense_memory'] = M_dense.nbytes / 1024**2
    results['sparse_memory'] = (M_sparse.data.nbytes + M_sparse.indices.nbytes + M_sparse.indptr.nbytes) / 1024**2
    results['sparsity'] = 100 * (1 - M_sparse.nnz / (n_nodes**2))
    results['dense_matvec_time'] = dense_matvec_time
    results['sparse_matvec_time'] = sparse_matvec_time
    results['dense_solve_time'] = dense_solve_time
    results['sparse_solve_time'] = sparse_solve_time
    
    return results

