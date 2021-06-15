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


template<class T>
__device__ T abs_cuda(T x);

template<>
__device__ float abs_cuda(float x)
{
    return(fabsf(x));
}
template<>
__device__ double abs_cuda(double x)
{
    return(fabs(x));
}


template<class T, class T_vec>
__global__ void return_abs_vec_kernel(size_t N, const T_vec x_in, T_vec x_out)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N)
    {
        return;
    }
    x_out[j] = abs_cuda<T>(x_in[j]);

}

template<class T, class T_vec>
__global__ void return_abs_vec_inplace_kernel(size_t N, T_vec x_)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N)
    {
        return;
    }
    x_[j] = abs_cuda<T>(x_[j]);

}

template<class T, class T_vec>
__global__ void convert_vector_T_to_double_kernel(size_t N, T_vec x_T_, double* x_D_)
{
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j>=N)
    {
        return;
    }
    x_D_[j] = double(x_T_[j]);

}


template<class T, class T_vec, int BLOCK_SIZE>
void generate_vector_pair_helper<T, T_vec, BLOCK_SIZE>::return_abs_vec(const T_vec x_in, T_vec x_out)
{
    return_abs_vec_kernel<T, T_vec><<<dimGrid, dimBlock>>>(sz, x_in, x_out);
}
template<class T, class T_vec, int BLOCK_SIZE>
void generate_vector_pair_helper<T, T_vec, BLOCK_SIZE>::return_abs_vec_inplace(T_vec x_)
{
    return_abs_vec_inplace_kernel<T, T_vec><<<dimGrid, dimBlock>>>(sz, x_);
}

template<class T, class T_vec, int BLOCK_SIZE>
void generate_vector_pair_helper<T, T_vec, BLOCK_SIZE>::return_abs_double_vec_inplace(double* x_)
{
    return_abs_vec_inplace_kernel<double, double*><<<dimGrid, dimBlock>>>(sz, x_);
}

template<class T, class T_vec, int BLOCK_SIZE>
void generate_vector_pair_helper<T, T_vec, BLOCK_SIZE>::convert_vector_T_to_double(T_vec x_T_, double* x_D_)
{
    convert_vector_T_to_double_kernel<T, T_vec><<<dimGrid, dimBlock>>>(sz, x_T_, x_D_);
}


#endif