#ifndef __RANDOM_ADAPTER_HPP__
#define __RANDOM_ADAPTER_HPP__

#include <random>
#include <string>
#include <stdexcept>


template<class T>
class random_adapter
{
    using uniform_param_t = typename std::uniform_real_distribution<T>::param_type;

private:

    std::mt19937 gen;
    std::uniform_real_distribution<T> dis;
    uniform_param_t uniform_range;
    bool uniform_set = false;

public:
    random_adapter():
    gen(std::random_device()())
    {
    }
    ~random_adapter()
    {
    }
   
    void set_uniform_distribution(T min_val_, T max_val_)
    {
        uniform_range = uniform_param_t(min_val_, max_val_);
        uniform_set = true;
    } 
    T get_uniform()
    {
        if(!uniform_set)
            throw std::runtime_error("calling uniform distribution but it is not set");

        return T(float(dis(gen, uniform_range)));
    }


};

#endif