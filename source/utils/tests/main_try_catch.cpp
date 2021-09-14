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

#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <utils/log.h>
#include <utils/main_try_catch_macro.h>

int main(int argc, char **args)
{
    if (argc < 2) {
        std::cout << "USAGE: " << std::string(args[0]) << " <block_number>" << std::endl;
        return 0;
    }

    utils::log_std  log;
    USE_MAIN_TRY_CATCH(log)  

    int block_number = atoi(args[1]);

    MAIN_TRY("test block 1")
    if (block_number == 1) throw std::runtime_error("error block 1");
    MAIN_CATCH(1)

    MAIN_TRY("test block 2")
    if (block_number == 2) throw std::runtime_error("error block 2");
    MAIN_CATCH(2)

    MAIN_TRY("test block 3")
    if (block_number == 3) throw std::runtime_error("error block 3");
    MAIN_CATCH(3)

    return 0;
}