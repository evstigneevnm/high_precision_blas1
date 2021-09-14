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

#include <stdio.h>
#include <string>
#include <utils/InitCUDA.h>
#include <utils/cuda_safe_call.h>

int main(int argc, char **argv)
{
        bool    do_error = false;
        if ((argc >= 2)&&(std::string(argv[1]) == std::string("1"))) do_error = true;
        if (do_error) printf("you specified do error on purpose\n");
        try {
                if (!InitCUDA(0)) throw std::runtime_error("InitCUDA failed");

                int     *p;
                if (!do_error)
                        CUDA_SAFE_CALL( cudaMalloc((void**)&p, sizeof(int)*512) );
                else
                        CUDA_SAFE_CALL( cudaMalloc((void**)&p, -100 ) );

                return 0;

        } catch (std::runtime_error &e) {
                printf("%s\n", e.what());

                return 1;
        }
}