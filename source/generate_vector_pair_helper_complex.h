#ifndef __GENERATE_VECTOR_PAIR_HELPER_COMPLEX_H__
#define __GENERATE_VECTOR_PAIR_HELPER_COMPLEX_H__

#include <cstddef>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <common/complex_real_type_cast.hpp>

template<class T, class T_vec, int BLOCK_SIZE = 1024>
class generate_vector_pair_helper_complex
{
private:
    using TR = typename deduce_real_type_from_complex::recast_type<T>::real;
    using TR_vec = TR*;
     
public:
    generate_vector_pair_helper_complex(size_t sz_):
    sz(sz_)
    {
        calculate_cuda_grid();
    }
    ~generate_vector_pair_helper_complex()
    {   

    }
    void generate_C_estimated_vector(const TR max_power, const T_vec mantisa_v, const TR_vec exponent_v, T_vec return_v, size_t N_fixed_value = 0);

    void split_complex_vector_to_reals(const T_vec x_in, TR_vec xR_out, TR_vec xI_out);


    void return_abs_vec(const TR_vec x_in, TR_vec x_out);
    void return_abs_vec_inplace(TR_vec x_);
    void return_abs_4double_vec_inplace(double* x1_, double* x2_, double* x3_, double* x4_);
    void return_abs_double_vec_inplace(double* x_);
    void convert_vector_T_to_double(TR_vec x_T_, double* x_D_);

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