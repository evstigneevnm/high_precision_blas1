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
#ifndef __CPU_VECTOR_OPERATIONS_CHILD_H__
#define __CPU_VECTOR_OPERATIONS_CHILD_H__

#include <common/cpu_vector_operations.h>

template <typename T>
class cpu_vector_operations_child: public cpu_vector_operations<T>
{
public:    
    using parent_t = cpu_vector_operations<T>;
    using scalar_type = typename parent_t::scalar_type;
    using vector_type = typename parent_t::vector_type;
public:
    cpu_vector_operations_child(size_t sz_, int use_high_precision_dot_product_ = 0, int use_threaded_dot_ = 0):
    cpu_vector_operations<T>::cpu_vector_operations(sz_, use_high_precision_dot_product_, use_threaded_dot_)
    {
        parent_t::init_vector(x1);
        parent_t::start_use_vector(x1);
        parent_t::init_vector(x2);
        parent_t::start_use_vector(x2);

    }
    ~cpu_vector_operations_child()
    {
        parent_t::stop_use_vector(x1);
        parent_t::stop_use_vector(x2);
        parent_t::free_vector(x1);
        parent_t::free_vector(x2);

    }

    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        scalar_type res = parent_t::scalar_prod(x, y);
        

        set_abs_values_x1_x2(x, y);

        parent_t::use_high_precision();
        scalar_type res1 = parent_t::scalar_prod(x, y);
        scalar_type res_abs = parent_t::scalar_prod(x1, x2);
        parent_t::use_default_precision();
        condition_numbers.push_back( res_abs/std::abs(res1) );
        return(res);
    }


    
    std::vector<double> get_condition_numbers() const
    {
        return(condition_numbers);
    }
    mutable std::vector<double> condition_numbers;

private:

    mutable vector_type x1 = nullptr;
    mutable vector_type x2 = nullptr;
    

    void set_abs_values_x1_x2(const vector_type &x, const vector_type &y) const
    {
        for(int j = 0; j<parent_t::sz_;j++)
        {
            x1[j] = std::abs(x[j]);
            x2[j] = std::abs(y[j]);
        }
    }

};




#endif