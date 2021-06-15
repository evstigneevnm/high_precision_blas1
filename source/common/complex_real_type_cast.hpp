#ifndef __COMPLEX_REAL_TYPE_CAST_HPP__
#define __COMPLEX_REAL_TYPE_CAST_HPP__

#include <thrust/complex.h>

namespace deduce_real_type_from_complex{
template<typename T>
struct recast_type
{
    using real = T;
};

template<>
struct recast_type< thrust::complex<float> >
{
    using real = float;
};

template<>
struct recast_type< thrust::complex<double> >
{
    using real = double;
};

}

#endif