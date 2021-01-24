#ifndef __GENERATE_VECTOR_PAIR_HELPER_IMPL_CUH__
#define __GENERATE_VECTOR_PAIR_HELPER_IMPL_CUH__

#include <generate_vector_pair_helper.h>



template<class T, class T_vec, int BLOCK_SIZE>
void generate_vector_pair_helper<T, T_vec, BLOCK_SIZE>::calculate_cuda_grid()
{
    dim3 dimBlock_s(BLOCK_SIZE);
    unsigned int blocks_x=floor(sz/( BLOCK_SIZE ))+1;
    dim3 dimGrid_s(blocks_x);
    dimBlock=dimBlock_s;
    dimGrid=dimGrid_s;
}


template<class T, class T_vec>
__global__ void generate_C_estimated_vector_kernel(size_t N, const T max_power, const T_vec mantisa_v, const T_vec exponent_v, T_vec return_v, size_t N_fixed_value)
{

    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N)
    {
        return;
    }
    
    exponent_v[0] = T(0.0);
    exponent_v[N_fixed_value] = max_power;
    T exponent = T( round( exponent_v[j] ) );

    return_v[j] = mantisa_v[j]*pow( T(2.0), exponent );

}


template<class T, class T_vec, int BLOCK_SIZE>
void generate_vector_pair_helper<T, T_vec, BLOCK_SIZE>::generate_C_estimated_vector(const T max_power, const T_vec mantisa_v, const T_vec exponent_v, T_vec return_v, size_t N_fixed_value)
{

    size_t N_fixed_value_l = N_fixed_value>0?N_fixed_value:std::round(sz/5);
    generate_C_estimated_vector_kernel<T, T_vec><<<dimGrid, dimBlock>>>(sz, max_power, mantisa_v, exponent_v, return_v, N_fixed_value_l);

}




#endif