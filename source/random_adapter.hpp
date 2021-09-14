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