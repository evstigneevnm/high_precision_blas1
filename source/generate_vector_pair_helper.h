#ifndef __GENERATE_VECTOR_PAIR_HELPER_H__
#define __GENERATE_VECTOR_PAIR_HELPER_H__

#include <cstddef>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>


template<class T, class T_vec, int BLOCK_SIZE = 1024>
class generate_vector_pair_helper
{
public:
    generate_vector_pair_helper(size_t sz_):
    sz(sz_)
    {
        calculate_cuda_grid();
    }
    ~generate_vector_pair_helper()
    {

    }
    void generate_C_estimated_vector(const T max_power, const T_vec mantisa_v, const T_vec exponent_v, T_vec return_v, size_t N_fixed_value = 0);

    

private:
    size_t sz;
    dim3 dimBlock;
    dim3 dimGrid;
    void calculate_cuda_grid();


};

    

#endif