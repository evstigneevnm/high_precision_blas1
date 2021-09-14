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

#include <cstdarg>
#include <utils/logged_obj_base.h>
#include <utils/log.h>

using namespace utils;

class logged_obj : public logged_obj_base<log_std>
{
public:
    logged_obj(log_std *log_ = NULL) : logged_obj_base<log_std>(log_) {}

    void test()
    {
        info("logged_obj: test to log");
        error("logged_obj: test error to log");
    }
};

template<class Log>
class logged_obj_template : public logged_obj_base<Log>
{
    typedef logged_obj_base<Log> logged_obj_t;
public:
    logged_obj_template(log_std *log_ = NULL) : logged_obj_base<log_std>(log_) {}

    void test()
    {
        logged_obj_t::info("logged_obj_template: test to log");
        logged_obj_t::error("logged_obj_template: test error to log");
        logged_obj_t::set_log_msg_prefix("logged_obj_template: ");
        logged_obj_t::info_f("test prefixed log");
        logged_obj_t::set_log_msg_prefix("");
        logged_obj_t::info("logged_obj_template: test to log normal againg");
    }
};

int main()
{
    log_std                         l;
    logged_obj                      o1(&l);
    logged_obj_template<log_std>    o2(&l);

    o1.test();
    o2.test();

    return 0;
}