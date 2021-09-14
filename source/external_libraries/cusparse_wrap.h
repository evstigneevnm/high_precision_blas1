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
// 
#ifndef __CUSPARSE_WRAP_H__
#define __CUSPARSE_WRAP_H__


#include <cusparse.h>
#include <thrust/complex.h>
#include <utils/cusparse_safe_call.h>

class cusparse_wrap
{
public:
    cusparseHandle_t handle = 0;
    
    cusparse_wrap()
    {
        if(handle == 0)
        {
            cusparseCreate(&handle);
        }
    }
    ~cusparse_wrap()
    {
        if(handle != 0)
        {
            cusparseDestroy(handle);
        }
    }
    
    template<class T>
    void cusparse_csrsv2_bufferSize
    (
        cusparseOperation_t transA,
        int m,
        int nnz,
        const cusparseMatDescr_t descrA,
        T* csrValA,
        const int* csrRowPtrA,
        const int* csrColIndA,
        csrsv2Info_t info,
        int* pBufferSizeInBytes
    ) const;

    template<class T>
    void cusparse_csrsv2_analysis
    (
        cusparseOperation_t      transA,
        int                      m,
        int                      nnz,
        const cusparseMatDescr_t descrA,
        T*                       csrValA,
        const int*               csrRowPtrA,
        const int*               csrColIndA,
        csrsv2Info_t             info,
        cusparseSolvePolicy_t    policy,
        void*                    pBuffer
    ) const;
    
    cusparseStatus_t cusparse_csrsv2_zeroPivot
    (
        csrsv2Info_t     info,
        int*             position
    ) const
    {
        return cusparseXcsrsv2_zeroPivot(handle, info, position);
    }

    template<class T>
    void cusparse_csrsv2_solve
    (
        cusparseOperation_t      transA,
        int                      m,
        int                      nnz,
        T*                       alpha,
        const cusparseMatDescr_t descra,
        T*                       csrValA,
        const int*               csrRowPtrA,
        const int*               csrColIndA,
        csrsv2Info_t             info,
        T*                       x,
        T*                       y,
        cusparseSolvePolicy_t    policy,
        void*                    pBuffer
    ) const;

    template<class T>
    void cusparse_csrilu02_bufferSize
    (
        int                      m,
        int                      nnz,
        const cusparseMatDescr_t descrA,
        T*                       csrValA,
        const int*               csrRowPtrA,
        const int*               csrColIndA,
        csrilu02Info_t           info,
        int*                     pBufferSizeInBytes
    );

    template<class T>
    void cusparse_csrilu02_analysis
    (
        int                      m,
        int                      nnz,
        const cusparseMatDescr_t descrA,
        T*                       csrValA,
        const int*               csrRowPtrA,
        const int*               csrColIndA,
        csrilu02Info_t           info,
        cusparseSolvePolicy_t    policy,
        void*                    pBuffer
    );
    cusparseStatus_t cusparse_csrilu02_zeroPivot
    (
        csrilu02Info_t   info,
        int* position
    )
    {
        return cusparseXcsrilu02_zeroPivot
        (
            handle,
            info,
            position
        );
    }
    template<class T>
    void cusparse_csrilu02
    (
        int                      m,
        int                      nnz,
        const cusparseMatDescr_t descrA,
        T*                       csrValA_valM,
        const int*               csrRowPtrA,
        const int*               csrColIndA,
        csrilu02Info_t           info,
        cusparseSolvePolicy_t    policy,
        void*                    pBuffer
    );


};

//    === specializations ===

template<> inline
void cusparse_wrap::cusparse_csrilu02
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    float*                   csrValA_valM,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (    
        cusparseScsrilu02
        (
            handle,
            m,
            nnz,
            descrA,
            csrValA_valM,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )        

    );
}   
template<> inline
void cusparse_wrap::cusparse_csrilu02
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    double*                  csrValA_valM,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (    
        cusparseDcsrilu02
        (
            handle,
            m,
            nnz,
            descrA,
            csrValA_valM,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )        

    );
}  
template<> inline
void cusparse_wrap::cusparse_csrilu02
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    thrust::complex<float>*  csrValA_valM,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (    
        cusparseCcsrilu02
        (
            handle,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuComplex*>(csrValA_valM),
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )        

    );
}  
template<> inline
void cusparse_wrap::cusparse_csrilu02
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    thrust::complex<double>* csrValA_valM,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (    
        cusparseZcsrilu02
        (
            handle,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuDoubleComplex*>(csrValA_valM),
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )        

    );
}

template<> inline
void cusparse_wrap::cusparse_csrilu02_analysis
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    float*                       csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseScsrilu02_analysis
        (
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}

template<> inline
void cusparse_wrap::cusparse_csrilu02_analysis
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    double*                  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseDcsrilu02_analysis
        (
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrilu02_analysis
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    thrust::complex<float>*  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseCcsrilu02_analysis
        (
            handle,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrilu02_analysis
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    thrust::complex<double>*  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseZcsrilu02_analysis
        (
            handle,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuDoubleComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}


template<> inline
void cusparse_wrap::cusparse_csrilu02_bufferSize
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    float*                   csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    int*                     pBufferSizeInBytes
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseScsrilu02_bufferSize
        (
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            pBufferSizeInBytes
        )
    );    
}

template<> inline
void cusparse_wrap::cusparse_csrilu02_bufferSize
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    double*                  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    int*                     pBufferSizeInBytes
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseDcsrilu02_bufferSize
        (
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            pBufferSizeInBytes
        )
    );    
}
template<> inline
void cusparse_wrap::cusparse_csrilu02_bufferSize
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    thrust::complex<float>*  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    int*                     pBufferSizeInBytes
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseCcsrilu02_bufferSize
        (
            handle,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            pBufferSizeInBytes
        )
    );    
}
template<> inline
void cusparse_wrap::cusparse_csrilu02_bufferSize
(
    int                      m,
    int                      nnz,
    const cusparseMatDescr_t descrA,
    thrust::complex<double>*  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrilu02Info_t           info,
    int*                     pBufferSizeInBytes
)
{
    CUSPARSE_SAFE_CALL
    (
        cusparseZcsrilu02_bufferSize
        (
            handle,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuDoubleComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            pBufferSizeInBytes
        )
    );    
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_solve
(
    cusparseOperation_t      transA,
    int                      m,
    int                      nnz,
    float*                   alpha,
    const cusparseMatDescr_t descra,
    float*                   csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrsv2Info_t             info,
    float*                   x,
    float*                   y,
    cusparseSolvePolicy_t    policy,
    void*                   pBuffer
) const
{    
    CUSPARSE_SAFE_CALL
    (    
        cusparseScsrsv2_solve
        (
            handle,
            transA,
            m,
            nnz,
            alpha,
            descra,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            x,
            y,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_solve
(
    cusparseOperation_t      transA,
    int                      m,
    int                      nnz,
    double*                  alpha,
    const cusparseMatDescr_t descra,
    double*                  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrsv2Info_t             info,
    double*                  x,
    double*                  y,
    cusparseSolvePolicy_t    policy,
    void*                   pBuffer
) const
{    
    CUSPARSE_SAFE_CALL
    (    
        cusparseDcsrsv2_solve
        (
            handle,
            transA,
            m,
            nnz,
            alpha,
            descra,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            x,
            y,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_solve
(
    cusparseOperation_t      transA,
    int                      m,
    int                      nnz,
    cuComplex*               alpha,
    const cusparseMatDescr_t descra,
    cuComplex*               csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrsv2Info_t             info,
    cuComplex*               x,
    cuComplex*               y,
    cusparseSolvePolicy_t    policy,
    void*                   pBuffer
) const
{    
    CUSPARSE_SAFE_CALL
    (    
        cusparseCcsrsv2_solve
        (
            handle,
            transA,
            m,
            nnz,
            alpha,
            descra,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            x,
            y,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_solve
(
    cusparseOperation_t      transA,
    int                      m,
    int                      nnz,
    cuDoubleComplex*         alpha,
    const cusparseMatDescr_t descra,
    cuDoubleComplex*         csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrsv2Info_t             info,
    cuDoubleComplex*         x,
    cuDoubleComplex*         y,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
) const
{    
    CUSPARSE_SAFE_CALL
    (    
        cusparseZcsrsv2_solve
        (
            handle,
            transA,
            m,
            nnz,
            alpha,
            descra,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            x,
            y,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_solve
(
    cusparseOperation_t      transA,
    int                      m,
    int                      nnz,
    thrust::complex<float>*  alpha,
    const cusparseMatDescr_t descra,
    thrust::complex<float>*  csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrsv2Info_t             info,
    thrust::complex<float>*  x,
    thrust::complex<float>*  y,
    cusparseSolvePolicy_t    policy,
    void*                   pBuffer
) const
{    
    CUSPARSE_SAFE_CALL
    (    
        cusparseCcsrsv2_solve
        (
            handle,
            transA,
            m,
            nnz,
            reinterpret_cast<cuComplex*>(alpha),
            descra,
            reinterpret_cast<cuComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            reinterpret_cast<cuComplex*>(x),
            reinterpret_cast<cuComplex*>(y),
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_solve
(
    cusparseOperation_t      transA,
    int                      m,
    int                      nnz,
    thrust::complex<double>* alpha,
    const cusparseMatDescr_t descra,
    thrust::complex<double>* csrValA,
    const int*               csrRowPtrA,
    const int*               csrColIndA,
    csrsv2Info_t             info,
    thrust::complex<double>* x,
    thrust::complex<double>* y,
    cusparseSolvePolicy_t    policy,
    void*                    pBuffer
) const
{    
    CUSPARSE_SAFE_CALL
    (    
        cusparseZcsrsv2_solve
        (
            handle,
            transA,
            m,
            nnz,
            reinterpret_cast<cuDoubleComplex*>(alpha),
            descra,
            reinterpret_cast<cuDoubleComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            reinterpret_cast<cuDoubleComplex*>(x),
            reinterpret_cast<cuDoubleComplex*>(y),
            policy,
            pBuffer
        )
    );
}



template<> inline
void cusparse_wrap::cusparse_csrsv2_analysis
(
cusparseOperation_t      transA,
int                      m,
int                      nnz,
const cusparseMatDescr_t descrA,
float*             csrValA,
const int*               csrRowPtrA,
const int*               csrColIndA,
csrsv2Info_t             info,
cusparseSolvePolicy_t    policy,
void*                    pBuffer
) const
{
    CUSPARSE_SAFE_CALL
    (
        cusparseScsrsv2_analysis
        (
            handle,
            transA,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}

template<> inline
void cusparse_wrap::cusparse_csrsv2_analysis
(
cusparseOperation_t      transA,
int                      m,
int                      nnz,
const cusparseMatDescr_t descrA,
double*             csrValA,
const int*               csrRowPtrA,
const int*               csrColIndA,
csrsv2Info_t             info,
cusparseSolvePolicy_t    policy,
void*                    pBuffer
) const
{
    CUSPARSE_SAFE_CALL
    (
        cusparseDcsrsv2_analysis
        (
            handle,
            transA,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_analysis
(
cusparseOperation_t      transA,
int                      m,
int                      nnz,
const cusparseMatDescr_t descrA,
cuComplex*             csrValA,
const int*               csrRowPtrA,
const int*               csrColIndA,
csrsv2Info_t             info,
cusparseSolvePolicy_t    policy,
void*                    pBuffer
) const
{
    CUSPARSE_SAFE_CALL
    (
        cusparseCcsrsv2_analysis
        (
            handle,
            transA,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_analysis
(
cusparseOperation_t      transA,
int                      m,
int                      nnz,
const cusparseMatDescr_t descrA,
cuDoubleComplex*         csrValA,
const int*               csrRowPtrA,
const int*               csrColIndA,
csrsv2Info_t             info,
cusparseSolvePolicy_t    policy,
void*                    pBuffer
) const
{
    CUSPARSE_SAFE_CALL
    (
        cusparseZcsrsv2_analysis
        (
            handle,
            transA,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}

template<> inline
void cusparse_wrap::cusparse_csrsv2_analysis
(
cusparseOperation_t      transA,
int                      m,
int                      nnz,
const cusparseMatDescr_t descrA,
thrust::complex<float>*  csrValA,
const int*               csrRowPtrA,
const int*               csrColIndA,
csrsv2Info_t             info,
cusparseSolvePolicy_t    policy,
void*                    pBuffer
) const
{
    CUSPARSE_SAFE_CALL
    (
        cusparseCcsrsv2_analysis
        (
            handle,
            transA,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_analysis
(
cusparseOperation_t      transA,
int                      m,
int                      nnz,
const cusparseMatDescr_t descrA,
thrust::complex<double>* csrValA,
const int*               csrRowPtrA,
const int*               csrColIndA,
csrsv2Info_t             info,
cusparseSolvePolicy_t    policy,
void*                    pBuffer
) const
{
    CUSPARSE_SAFE_CALL
    (
        cusparseZcsrsv2_analysis
        (
            handle,
            transA,
            m,
            nnz,
            descrA,
            reinterpret_cast<cuDoubleComplex*>(csrValA),
            csrRowPtrA,
            csrColIndA,
            info,
            policy,
            pBuffer
        )
    );
}




template<> inline
void cusparse_wrap::cusparse_csrsv2_bufferSize(cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrsv2Info_t info, int* pBufferSizeInBytes) const
{
        CUSPARSE_SAFE_CALL
        (
            cusparseDcsrsv2_bufferSize(
                handle, 
                transA,
                m,
                nnz,
                descrA,
                csrValA, csrRowPtrA, csrColIndA,
                info,
                pBufferSizeInBytes
            )
        );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_bufferSize(cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrsv2Info_t info, int* pBufferSizeInBytes) const
{
        CUSPARSE_SAFE_CALL
        (
            cusparseScsrsv2_bufferSize(
                handle, 
                transA,
                m,
                nnz,
                descrA,
                csrValA, csrRowPtrA, csrColIndA,
                info,
                pBufferSizeInBytes
            )
        );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_bufferSize(cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, thrust::complex<float>* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrsv2Info_t info, int* pBufferSizeInBytes) const
{
        cuComplex* csrValA_ = reinterpret_cast<cuComplex*>(csrValA);
        CUSPARSE_SAFE_CALL
        (
            cusparseCcsrsv2_bufferSize(
                handle, 
                transA,
                m,
                nnz,
                descrA,
                csrValA_, csrRowPtrA, csrColIndA,
                info,
                pBufferSizeInBytes
            )
        );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_bufferSize(cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, thrust::complex<double>* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrsv2Info_t info, int* pBufferSizeInBytes) const
{
        cuDoubleComplex* csrValA_ = reinterpret_cast<cuDoubleComplex*>(csrValA);
        CUSPARSE_SAFE_CALL
        (
            cusparseZcsrsv2_bufferSize(
                handle, 
                transA,
                m,
                nnz,
                descrA,
                csrValA_, csrRowPtrA, csrColIndA,
                info,
                pBufferSizeInBytes
            )
        );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_bufferSize(cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrsv2Info_t info, int* pBufferSizeInBytes) const
{
        CUSPARSE_SAFE_CALL
        (
            cusparseCcsrsv2_bufferSize(
                handle, 
                transA,
                m,
                nnz,
                descrA,
                csrValA, csrRowPtrA, csrColIndA,
                info,
                pBufferSizeInBytes
            )
        );
}
template<> inline
void cusparse_wrap::cusparse_csrsv2_bufferSize(cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrsv2Info_t info, int* pBufferSizeInBytes) const
{
        
        CUSPARSE_SAFE_CALL
        (
            cusparseZcsrsv2_bufferSize(
                handle, 
                transA,
                m,
                nnz,
                descrA,
                csrValA, csrRowPtrA, csrColIndA,
                info,
                pBufferSizeInBytes
            )
        );
}

#endif