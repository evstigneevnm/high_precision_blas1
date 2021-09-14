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