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
#ifndef __CSR__MATRIX_H__
#define __CSR__MATRIX_H__

#include <type_traits>
#include <stdexcept>

namespace csr
{

template<class VectorOperations, class VectorOperationsOrdinal, class Add4VectorOpperations = int>
class matrix
{
private:
    Add4VectorOpperations* addition_gpu_param;
    struct matrix_dimentions
    {
        int columns = 0;
        int rows = 0;
    };    
public:
    using dimensions_t = matrix_dimentions;
    using T =  typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using I = typename VectorOperationsOrdinal::scalar_type;
    using I_vec = typename VectorOperationsOrdinal::vector_type;
    using vec_ops_t = VectorOperations;
    using vec_ops_ord_t = VectorOperationsOrdinal;

    matrix()
    {}
    matrix(Add4VectorOpperations* addition_gpu_param_= 0):
    addition_gpu_param(addition_gpu_param_)
    {}    
    matrix(int nnz_, int size_col_, int size_row_, int matrix_columns_, int matrix_rows_, Add4VectorOpperations* addition_gpu_param_= 0):
    addition_gpu_param(addition_gpu_param_)
    {
        init(nnz_, size_col_, size_row_);
    }
    //copy constructor
    matrix(const matrix& m_)
    {
        addition_gpu_param = m_.addition_gpu_param;
        dim.columns = m_.dim.columns;
        dim.rows = m_.dim.rows;
        nnz = m_.nnz;
        size_col = m_.size_col;
        size_row = m_.size_row;
        init(nnz, size_col, size_row);
        set(m_.data, m_.col_ind, m_.row_ptr);

    }
    ~matrix()
    {
     
        vec_ops_d->stop_use_vector(data); vec_ops_d->free_vector(data);
        vec_ops_c->stop_use_vector(col_ind); vec_ops_c->free_vector(col_ind);
        vec_ops_r->stop_use_vector(row_ptr); vec_ops_r->free_vector(row_ptr);

        delete vec_ops_r;
        delete vec_ops_c; 
        delete vec_ops_d;
    }
    void set_dim(int matrix_columns_, int matrix_rows_)
    {
        dim.columns = matrix_columns_;
        dim.rows = matrix_rows_;  
    }
    void init(int nnz_, int size_col_, int size_row_)
    {
        if(!initialized)
        {

            nnz = nnz_;
            size_col = size_col_;
            size_row = size_row_;
            if (std::is_same<Add4VectorOpperations, int>::value)
            {
                vec_ops_d = new vec_ops_t(nnz);
            }
            else
            {
                vec_ops_d = new vec_ops_t(nnz, addition_gpu_param);    
            }
            vec_ops_c = new vec_ops_ord_t(size_col);
            vec_ops_r = new vec_ops_ord_t(size_row);

            vec_ops_d->init_vector(data); vec_ops_d->start_use_vector(data);
            vec_ops_c->init_vector(col_ind); vec_ops_c->start_use_vector(col_ind);
            vec_ops_r->init_vector(row_ptr); vec_ops_r->start_use_vector(row_ptr);
            initialized = true;
        }
    }

    dimensions_t get_dim()const
    {
        return(dim);
    }
    void set(const T_vec data_, const I_vec col_ind_, const I_vec row_ptr_)
    {
        if(initialized)
        {
            vec_ops_d->set(data_, data);
            vec_ops_c->set(col_ind_, col_ind);
            vec_ops_r->set(row_ptr_, row_ptr);
        }
        else
        {
            throw(std::runtime_error("csr::matrix: trying to set an uninitialized matrix.") );
        }

    }

    void init_set(int nnz_, int size_col_, int size_row_, const T_vec data_, const I_vec col_ind_, const I_vec row_ptr_, int matrix_columns_, int matrix_rows_)
    {
        init(nnz_, size_col_, size_row_);
        set(data_, col_ind_, row_ptr_);
    }

protected:
    T_vec data = nullptr;
    I_vec col_ind = nullptr;
    I_vec row_ptr = nullptr;
    int size_row, size_col, nnz;
    bool initialized = false;
    matrix_dimentions dim;
private:

    vec_ops_t* vec_ops_d = nullptr;
    vec_ops_ord_t* vec_ops_c = nullptr;
    vec_ops_ord_t* vec_ops_r = nullptr;




};

}

#endif