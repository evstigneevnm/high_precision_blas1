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
#ifndef __VECTOR_WRAP_H__
#define __VECTOR_WRAP_H__


template <class VecOps>
class vector_wrap
{
public: 
    typedef VecOps vector_operations;
    typedef typename VecOps::vector_type  vector_type;
    typedef typename VecOps::scalar_type  scalar_type;

private:
    typedef scalar_type T;
    typedef vector_type T_vec;


    VecOps* vec_ops;
    bool allocated = false;
    void set_op(VecOps* vec_ops_){ vec_ops = vec_ops_; }

public:
    vector_wrap()
    {
    }
    ~vector_wrap()
    {
        free();
    }

    void alloc(VecOps* vec_ops_)
    {
        set_op(vec_ops_);

        if(!allocated)
        {
            vec_ops->init_vector(x); vec_ops->start_use_vector(x); 
            allocated = true;
        }
    }
    void free()
    {
        
        if(allocated)
        {
            vec_ops->stop_use_vector(x); vec_ops->free_vector(x);
            allocated = false;
        }
    }

    T_vec& get_ref()
    {
        return(x);
    }

    T_vec x = nullptr;

};


#endif // __VECTOR_WRAP_H__