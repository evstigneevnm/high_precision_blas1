#ifndef __GPU_REDUCTION_OGITA_H__
#define __GPU_REDUCTION_OGITA_H__

#include <utility>
#include <cstddef>

#include <utils/cuda_support.h>

template<class T, class T_vec, int BLOCK_SIZE = 1024, int threads_r = 64>
class gpu_reduction_ogita
{
public:
    using min_max_t = std::pair<T,T>;

    gpu_reduction_ogita(size_t vec_size_):
    vec_size(vec_size_)
    {
        vec_helper = device_allocate_host<T>(vec_size);
        vec_helper_d = device_allocate<T>(vec_size);

    }
    ~gpu_reduction_ogita()
    {
        if(vec_helper != nullptr)
        {
            device_deallocate_host(vec_helper);
        }
        if(vec_helper_d != nullptr)
        {
            device_deallocate(vec_helper_d);
        }
    }


    T sum(const T_vec d_in)
    {
        T res = reduction_sum(vec_size, d_in, vec_helper_d, vec_helper);
        return res;
    }

    // T dot(const T_vec d1_in, const T_vec d2_in)
    // {
    //     T res = reduction_dot(vec_size, d1_in, d2_in, vec_helper_d, vec_helper);
    //     return res;
    // }

private:

    size_t vec_size;
    T_vec vec_helper_d = nullptr;
    T_vec vec_helper = nullptr;


    T reduction_sum(int num_el, const T_vec InputV, T_vec OutputV, T_vec Output);
    
    // T reduction_dot(int N, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, T_vec Output);

    void findBlockSize(int* whichSize, int num_el);
    // for any integer returns the closest larger power_of_two neighbour.
    unsigned int nextPow2(unsigned int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }
    bool isPow2(unsigned int x)
    {
        return ( (x&(x-1))==0 );
    }
    void get_blocks_threads_shmem(int n, int maxBlocks, int& blocks, int& threads, int& smemSize);
    
    void wrapper_reduce_sum(int blocks, int threads, int smemSize, const T_vec InputV, T_vec OutputV, int N);
    // void wrapper_reduce_dot(int blocks, int threads, int smemSize, const T_vec InputV1, const T_vec InputV2, T_vec OutputV, int N);

};


#endif

    