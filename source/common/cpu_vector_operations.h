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
#ifndef __cpu_vector_operations_H__
#define __cpu_vector_operations_H__

#include <cmath>
#include <common/dot_product.h>
#include <common/threaded_reduction.h>


template <typename T>
class cpu_vector_operations
{
private:
    int use_high_precision_dot_product;
public:
    // typedef T  scalar_type;
    // typedef T* vector_type;
    using scalar_type = T;
    using vector_type = T*;
    bool location;
    size_t sz_;    
    dot_product<T>* dot = nullptr;
    threaded_reduction<scalar_type, vector_type>* threaded_dot = nullptr;
    int use_threaded_dot = 0;

    cpu_vector_operations(size_t sz, int use_high_precision_dot_product_ = 0, int use_threaded_dot_ = 0):
    sz_(sz),
    use_threaded_dot(use_threaded_dot_),
    use_high_precision_dot_product(use_high_precision_dot_product_)
    {
        location=false;
        dot = new dot_product<T>(sz, use_high_precision_dot_product);
        threaded_dot = new threaded_reduction<scalar_type, vector_type>(sz_, use_threaded_dot_, use_high_precision_dot_product);

    }
    ~cpu_vector_operations()
    {
        if(dot!=nullptr)
        {
            delete dot;
        }

        if(threaded_dot!=nullptr)
        {
            delete threaded_dot;
        }
    }

    size_t get_vector_size()
    {
        return sz_;
    }
    bool device_location()
    {
        return location;
    }



    void init_vector(vector_type& x)const 
    {
        x = NULL;
    }
    void free_vector(vector_type& x)const 
    {
        if (x != NULL) free(x);
    }
    void start_use_vector(vector_type& x)const
    {
        if (x == NULL) x = (T*)malloc(sz_*sizeof(T));
    }
    void stop_use_vector(vector_type& x)const
    {
    }

    bool check_is_valid_number(const vector_type &x)const
    {

        for (int i = 0;i < sz_;++i)
        {
            if (std::isinf(x[i]))
            {
                return false;
            }
        }
        
        return true;
    }
    
    void use_high_precision() const
    {
        dot->use_high_prec();
        threaded_dot->use_high_precision();
    }
    void use_standard_precision() const
    {
        dot->use_normal_prec();
        threaded_dot->use_standard_precision();
    }    
    void use_default_precision() const
    {
        if(use_high_precision_dot_product == 0)
        {
            dot->use_normal_prec();
            threaded_dot->use_standard_precision();
        }
        if(use_high_precision_dot_product == 1)
        {
            dot->use_high_prec();
            threaded_dot->use_high_precision();
        }
    }
    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        // T res(0.f);
        // for (int i = 0;i < sz_;++i)
        // {
        //     res += x[i]*y[i];
        // }        
        // return res;
        scalar_type dot_res = T(0.0);

        if (use_threaded_dot <= 1)
        {
            dot_res = dot->dot(x, y);
        }
        else
        {
            dot_res = threaded_dot->dot(x, y);            
        }
        return dot_res;

    }
    scalar_type norm(const vector_type &x)const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    scalar_type norm_sq(const vector_type &x)const
    {
        return scalar_prod(x, x);
    }    

    //calc: x := <vector_type with all elements equal to given scalar value> 
    void assign_scalar(const scalar_type scalar, vector_type& x)const
    {
        for (int i = 0;i < sz_;++i) 
            x[i] = scalar;
    }
    //calc: x := mul_x*x + <vector_type of all scalar value> 
    void add_mul_scalar(scalar_type scalar, scalar_type mul_x, vector_type& x)const
    {
        for (int i = 0;i < sz_;++i) 
            x[i] = mul_x*x[i] + scalar;
    }
    void scale(scalar_type scale, vector_type &x)const
    {
           add_mul_scalar(scalar_type(0),scale, x);
    }
    //copy: y := x
    void assign(const vector_type& x, vector_type& y)const
    {
        for (int i = 0;i < sz_;++i) 
            y[i] = x[i];
    }
    //calc: y := mul_x*x
    void assign_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        for (int i = 0;i < sz_;++i) 
            y[i] = mul_x*x[i];
    }
    
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                               vector_type& z)const
    {
        for (int i = 0;i < sz_;++i) 
            z[i] = mul_x*x[i] + mul_y*y[i];
    }
    //calc: y := mul_x*x + y
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        for (int i = 0;i < sz_;++i) 
            y[i] += mul_x*x[i];
    }
    //calc: y := mul_x*x + mul_y*y
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
    {
        for (int i = 0;i < sz_;++i) 
            y[i] = mul_x*x[i] + mul_y*y[i];
    }
    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, 
                            scalar_type mul_z, vector_type& z)const
    {
        for (int i = 0;i < sz_;++i) 
            z[i] = mul_x*x[i] + mul_y*y[i] + mul_z*z[i];
    }
};




#endif