from testbook import testbook

def test_values():
    with testbook('PA_2_2_sparse_v_dense.ipynb', execute=True) as tb:

        results = tb.value("results")
        
        assert len(results) > 2
        assert 'sparse_solve_time' in results[2]
        assert 'dense_solve_time' in results[2]
        speedup = results[2]['dense_solve_time'] / results[2]['sparse_solve_time']
        assert speedup > 10


