#ifndef __GPU_VECTOR_OPERATIONS_CHILD_H__
#define __GPU_VECTOR_OPERATIONS_CHILD_H__

#include <common/gpu_vector_operations.h>

template <typename T, int BLOCK_SIZE = 1024>
class gpu_vector_operations_child: public gpu_vector_operations<T, BLOCK_SIZE>
{
public:
    gpu_vector_operations_child(size_t sz_, cublas_wrap *cuBLAS_):
    gpu_vector_operations<T, BLOCK_SIZE>::gpu_vector_operations(sz_, *cuBLAS_)
    {

    }
    ~gpu_vector_operations_child()
    {

    }
    
};




#endif