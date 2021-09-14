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
#ifndef __LINEAR_OPERATOR_HPP__
#define __LINEAR_OPERATOR_HPP__

//TODO now works only for a > 0
template <class VectorOperations>
struct SystemOperator
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    int sz_;
    T    a_, re_, h_;
    const VectorOperations* vec_ops_;

    SystemOperator(const VectorOperations* vec_ops, T a, T re):
        vec_ops_(vec_ops), a_(a), re_(re)
    {
        sz_ = vec_ops_->sz_;
        h_ = (T(1)/T(sz_));
    }

    void apply(const T_vec& x, T_vec& f)const
    {
        //TODO
        for (int i = 0;i < sz_;++i) {
            T xm = (i > 0  ? x[i-1] : T(0.f)),
              xp = (i+1 < sz_ ? x[i+1] : T(0.f));
            
            f[i] = a_*(x[i]-xm)/h_ - (T(1.f)/re_)*(xp - T(2.f)*x[i] + xm)/(h_*h_);
            //f[i] = (T(1.f)/re_)*(x[i+1] - T(2.f)*x[i] + x[i-1])/(h_*h_);
        }
    }
};

//TODO now works only for a > 0
template<class VectorOperations, class SystemOperator>
struct prec_operator
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    const SystemOperator *op_;

    prec_operator() 
    {}

    void set_operator(const SystemOperator *op)
    {
        op_ = op;
    }

    void apply(T_vec& x)const
    {
        T a = op_->a_;
        T h = op_->h_;
        T re = op_->re_;
        for (int i = 0;i < op_->sz_;++i) 
        {
            // T xm = (i > 0  ? x[i-1] : T(0.f)),
              // xp = (i+1 < sz_ ? x[i+1] : T(0.f));

            x[i] /= (a/h - (T(1.0)/re)*(-T(2.0))/(h*h));

            // T num = 
            //x[i] /= ((T(1.f)/re_)*(-T(2.f))/(h_*h_));
        }
    }
};


#endif