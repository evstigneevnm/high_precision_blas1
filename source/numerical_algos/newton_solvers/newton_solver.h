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
#ifndef __newton_solver_H__
#define __newton_solver_H__
/*
    Newton solver in general

*/

#include <string>
#include <stdexcept>

namespace numerical_algos
{
namespace newton_method
{

template<class vector_operations, class nonlinear_operator, class system_operator, class convergence_strategy>
class newton_solver
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    newton_solver(vector_operations*& vec_ops_, system_operator*& system_op_, convergence_strategy*& conv_strat_):
    vec_ops(vec_ops_),
    system_op(system_op_),
    conv_strat(conv_strat_)
    {
        vec_ops->init_vector(delta_x); vec_ops->start_use_vector(delta_x);
    }
    
    ~newton_solver()
    {
        vec_ops->stop_use_vector(delta_x); vec_ops->free_vector(delta_x); 

    }

    //solve inplace
    bool solve(nonlinear_operator*& nonlin_op, T_vec& x, const T& lambda)
    {
        
        int result_status = 1;
        vec_ops->assign_scalar(T(0.0), delta_x);
        bool converged = false;
        bool finished = false;
        bool linsolver_converged;
        conv_strat->reset_iterations();
        while(!finished)
        {
            vec_ops->assign_scalar(T(0.0), delta_x);
            linsolver_converged = system_op->solve(nonlin_op, x, lambda, delta_x);
            if(linsolver_converged)
            {
                finished = conv_strat->check_convergence(nonlin_op, x, lambda, delta_x, result_status);
            }
            else
            {
                finished = true;
                result_status = 100;
            }

        }
        if(result_status==0)
            converged = true;

        if((result_status==2)||(result_status==3))
        {
            throw std::runtime_error(std::string("newton_method" __FILE__ " " __STR(__LINE__) "invalid number.") );            
        }

        return converged;

    }

    bool solve(nonlinear_operator*& nonlin_op, const T_vec& x0, const T& lambda0, T_vec& x)
    {
        vec_ops->assign(x0, x);
        bool converged = false;
        converged = solve(nonlin_op, x, lambda0);
        if(!converged)
        {
            vec_ops->assign(x0, x);
        }
        return converged;

    }




    convergence_strategy* get_convergence_strategy_handle()
    {
        return conv_strat;
    }

private:
    vector_operations* vec_ops;
    system_operator* system_op;
    convergence_strategy* conv_strat;
    T_vec delta_x;


};



}
}

#endif