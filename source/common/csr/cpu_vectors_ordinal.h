#ifndef __CSR__CPU_VECTORS_ORDINAL_H__
#define __CSR__CPU_VECTORS_ORDINAL_H__

#include<cstddef>
#include<cstdlib>

namespace csr
{

template<class I = int> //assumed to be an ordinal type
class cpu_vectors_ordinal
{
public:
    using scalar_type = I;
    using vector_type = I*;

    cpu_vectors_ordinal(size_t sz_):
    sz(sz_)
    { }
    ~cpu_vectors_ordinal()
    { }
    
    void init_vector(vector_type& x)const 
    {
        x = nullptr;
    }
    void free_vector(vector_type& x)const 
    {
        if (x != nullptr)
        { 
            std::free(x);
        }
    }
    void start_use_vector(vector_type& x)const
    {
        if (x == nullptr) 
        {
           x = reinterpret_cast<vector_type>( std::malloc(sz*sizeof(I) ) );
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
            #pragma omp parallel for
            for(int j = 0;j<sz;j++)
            {
                x_[j] = x_host_[j];
            }
        }
    }   

private:
    size_t sz;

};
}
#endif