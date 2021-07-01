#ifndef __CSR__GPU_VECTORS_ORDINAL_H__
#define __CSR__GPU_VECTORS_ORDINAL_H__

#include<cstddef>
#include<cstdlib>
#include<utils/cuda_support.h>

namespace csr
{
template<class I = int>
class gpu_vectors_ordinal
{
public:
    using scalar_type = I;
    using vector_type = I*;
    
    gpu_vectors_ordinal(size_t sz_):
    sz(sz_)
    {
    }
    
    ~gpu_vectors_ordinal()
    {
    }

    void init_vector(vector_type& x)const 
    {
        x = nullptr;
    }
    void free_vector(vector_type& x)const 
    {
        if (x != nullptr) 
        {
            cudaFree(x);
        }
    }
    void start_use_vector(vector_type& x)const
    {
        if (x == nullptr) 
        {
           x = device_allocate<scalar_type>(sz);
        }
    }
    void stop_use_vector(vector_type& x)const
    {
    }
    size_t get_vector_size()
    {
        return sz;
    }
    //sets a vector from a host vector. 
    void set(const vector_type& x_host_, vector_type& x_) const
    {
        if(x_!=nullptr)
        {
            host_2_device_cpy<scalar_type>(x_, x_host_, sz);
        }
    }    
private:
    size_t sz;
};
}
#endif