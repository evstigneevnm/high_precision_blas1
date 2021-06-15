#ifndef __GENERATE_VECTOR_PAIR_HELPER_H__
#define __GENERATE_VECTOR_PAIR_HELPER_H__

#include <cstddef>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <utils/cuda_support.h>

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

    void return_abs_vec(const T_vec x_in, T_vec x_out);
    void return_abs_vec_inplace(T_vec x_);
    void return_abs_double_vec_inplace(double* x_);
    void convert_vector_T_to_double(T_vec x_T_, double* x_D_);

    void init_double_vectors(double*& x1, double*& x2)
    {
        x1 = device_allocate<double>(sz);
        x2 = device_allocate<double>(sz);
    }
    void delete_double_vectors(double*& x1, double*& x2)
    {
        if(x1 != nullptr)
            device_deallocate<double>(x1);
        if(x2 != nullptr)
            device_deallocate<double>(x2);
    }

private:
    size_t sz;
    dim3 dimBlock;
    dim3 dimGrid;
    void calculate_cuda_grid();

};

    

#endif