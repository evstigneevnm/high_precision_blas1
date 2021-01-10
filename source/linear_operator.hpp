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

            x[i] /= (a/h - (T(1.f)/re)*(-T(2.f))/(h*h));

            // T num = 
            //x[i] /= ((T(1.f)/re_)*(-T(2.f))/(h_*h_));
        }
    }
};


#endif