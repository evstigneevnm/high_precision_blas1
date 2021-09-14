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

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_VECTORS_ARR_WRAP_STATIC_H__
#define __SCFD_VECTORS_ARR_WRAP_STATIC_H__

#include "vector_wrap.h"

//ISSUE is it better to call free() and stop_use_all() in case of exception in 
//init() and stop_use_all() or in constructors as i planned initically
//Is it wrong that stop_use_all() called in start_use_all() and not in start_use(i)?

namespace numerical_algos
{
namespace detail
{

template<class VectorOperations, int Sz>
struct vectors_arr_wrap_static
{
    typedef VectorOperations                        vector_operations_type;
    typedef typename VectorOperations::vector_type  vector_type;
    typedef vector_wrap<VectorOperations>           vector_wrap_type;

    struct vectors_arr_use_wrap_type
    {
        vectors_arr_wrap_static     &vec_wrap_;

        vectors_arr_use_wrap_type(vectors_arr_wrap_static  &vec_wrap, 
                                  bool call_start_use_all = false) : vec_wrap_(vec_wrap)
        {
            if (call_start_use_all) start_use_all();
        }

        void start_use(int i)
        {
            vec_wrap_.start_use(i);
        }
        void stop_use(int i)
        {
            vec_wrap_.stop_use(i);
        }
        void start_use_all()
        {
            try 
            {
                vec_wrap_.start_use_all();
            }
            catch (...)
            {
                stop_use_all();
                throw;
            }
        }
        void start_use_range(size_t Sz_new, int from = 0)
        {
            try 
            {
                vec_wrap_.start_use_range(Sz_new, from);
            }
            catch (...)
            {
                stop_use_all();
                throw;
            }
        }        
        void stop_use_all()
        {
            vec_wrap_.stop_use_all();
        }


        ~vectors_arr_use_wrap_type()
        {
            stop_use_all();
        }
    };

    const vector_operations_type   *vec_ops_;
    vector_wrap_type                vecs[Sz];

    vectors_arr_wrap_static(const vector_operations_type   *vec_ops, 
                            bool call_init = false) : vec_ops_(vec_ops)
    {
        if (call_init) init();
    }

    const vector_type &operator[](int i)const { return vecs[i].vector(); }
    vector_type &operator[](int i) { return vecs[i].vector(); }

    void init(size_t Sz_new = Sz)
    {
        if(Sz_new > Sz)
            throw std::logic_error("numerical_algos::detail::vector_wrap_static: provided buffer size is greater than the static size.");

        try 
        {
            for (int i = 0;i < Sz_new;++i)
            {
                vecs[i].init(*vec_ops_);
            }
            //free if extra data was taken?
            for (int i = Sz_new;i < Sz;++i)
            {
                vecs[i].stop_use(*vec_ops_);
                vecs[i].free(*vec_ops_);
            }            
        }
        catch (...) 
        {
            free();
            throw;
        }
    }
    void  free()
    {
        for (int i = 0;i < Sz;++i) 
            vecs[i].free(*vec_ops_);
    }

    void start_use(int i)
    {
        vecs[i].start_use(*vec_ops_);
    }
    void stop_use(int i)
    {
        vecs[i].stop_use(*vec_ops_);
    }
    void start_use_all()
    {
        for (int i = 0;i < Sz;++i) 
            vecs[i].start_use(*vec_ops_);
    }
    void stop_use_all()
    {
        for (int i = 0;i < Sz;++i) 
            vecs[i].stop_use(*vec_ops_);
    }
    
    void start_use_range(size_t Sz_new, int from = 0)
    {
        if(Sz_new > Sz)
            throw std::logic_error("numerical_algos::detail::vector_wrap_static: provided buffer size is greater than the static size.");

        for (int i = from;i < Sz_new;++i) 
            vecs[i].start_use(*vec_ops_); 

    }

    ~vectors_arr_wrap_static()
    {
        free();
    }    
};

}
}

#endif
