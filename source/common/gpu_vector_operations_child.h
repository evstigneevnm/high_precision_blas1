#ifndef __GPU_VECTOR_OPERATIONS_CHILD_H__
#define __GPU_VECTOR_OPERATIONS_CHILD_H__

#include <fstream>
#include <string>
#include <common/gpu_vector_operations.h>

template <typename T, class ConditionNumberHelper, int BLOCK_SIZE = 1024>
class gpu_vector_operations_child: public gpu_vector_operations<T, BLOCK_SIZE>
{
public:
    using parent_t = gpu_vector_operations<T>;
    using scalar_type = typename parent_t::scalar_type;
    using vector_type = typename parent_t::vector_type;
    using TR = typename parent_t::norm_type;

    gpu_vector_operations_child(size_t sz_, cublas_wrap *cuBLAS_):
    gpu_vector_operations<T, BLOCK_SIZE>::gpu_vector_operations(sz_, cuBLAS_)
    {

    }
    ~gpu_vector_operations_child()
    {

    }

    void set_condition_number_helper(const ConditionNumberHelper* cn_help_)
    {
        cn_help = cn_help_;
        cn_helper_set = true;
    }
    scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        if(!cn_helper_set)
        {
            throw std::logic_error("calling a child to get a condition number but the class that implements it is not set!");
        }
        bool high_prec_ = parent_t::get_current_precision();
        scalar_type res = parent_t::scalar_prod(x, y);
        //... calculate the condition number
        TR cn_ = cn_help->condition_number_max(x, y);
        cond_numbs.push_back(cn_);
        //...done
        if(high_prec_)
        {
            parent_t::use_high_precision();
        }
        else
        {
            parent_t::use_standard_precision();
        }

        return(res);
    }

    void reset_condition_number_storage() const
    {
        cond_numbs.clear();
        cond_numbs.shrink_to_fit();
    }

    void write_condition_number_file(const std::string& file_name_)
    {
        cond_numbs.shrink_to_fit();
        std::ofstream f(file_name_.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("gpu_vector_operations_child: error while opening file " + file_name_);

        for(auto &val_: cond_numbs)
        {
            if (!(f << val_ << std::endl)) 
                throw std::runtime_error("write_convergency: error while writing to file " + file_name_);
        }
        f.close();
            

    }

private:
    mutable std::vector<TR> cond_numbs;
    mutable bool cn_helper_set = false;
    const ConditionNumberHelper* cn_help;

    
};




#endif