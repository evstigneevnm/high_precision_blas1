#ifndef __CUMP_BLAS_KERNELS_H__
#define __CUMP_BLAS_KERNELS_H__


#include <cstdint>
#include <cump.h>

template<int BLOCK_SIZE = 1024, int threads_r = 64>
class cump_blas_kernels
{
public:
    cump_blas_kernels(size_t sz_, unsigned int prec_):
    sz(sz_),
    prec(prec_)
    {

    }
    ~cump_blas_kernels()
    {

    }

    void set_prec(unsigned int prec_)
    {
        prec = prec_;
    }

    void dot(const cumpf_array_t& x_device, const cumpf_array_t& y_device, cumpf_array_t& res_device);
    void sum(const cumpf_array_t& x_device, cumpf_array_t& res_device);


    float get_execution_time_milliseconds()
    {
        return(milliseconds);
    }

private:
    size_t sz;
    unsigned int prec;
    float milliseconds = 0;
    void get_blocks_threads_shmem(int n, int maxBlocks, int &blocks, int &threads, int &sdataSize);
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

};



#endif