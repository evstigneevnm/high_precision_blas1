/*
 * MIT License
 *
 * Copyright (c) 2020 Evstigneev Nikolay Mikhaylovitch <evstigneevnm@ya.ru>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __GPU_FILE_OPERATIONS_H__
#define __GPU_FILE_OPERATIONS_H__

#include <common/file_operations.h>


template<class VectorOperations>
class gpu_file_operations
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    gpu_file_operations(VectorOperations* vec_op_):
    vec_op(vec_op_)
    {
        sz = vec_op->get_vector_size();
    }

    ~gpu_file_operations()
    {

    }

    void write_vector(const std::string &f_name, const T_vec& vec_gpu, unsigned int prec=16) const
    {
        file_operations::write_vector<T>(f_name, sz, vec_op->view(vec_gpu), prec);
    }

    void read_vector(const std::string &f_name, T_vec vec_gpu) const
    {
        
        file_operations::read_vector<T>(f_name, sz, vec_op->view(vec_gpu));
        vec_op->set(vec_gpu);
    }


private:
    VectorOperations* vec_op;
    size_t sz;

};




#endif // __GPU_FILE_OPERATIONS_H__