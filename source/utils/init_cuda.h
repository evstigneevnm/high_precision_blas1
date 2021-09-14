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

#ifndef __INITCUDA_H__
#define __INITCUDA_H__

#include <stdio.h>
#include <cuda_runtime.h>

namespace utils
{

#if __DEVICE_EMULATION__

inline bool init_cuda(int user_prefered_i = -1){ return true; }

#else
inline bool init_cuda(int user_prefered_i = -1)
{
    printf("Trying to init cuda\n");
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    if (user_prefered_i == -1) {
        for(i = 0; i < count; i++) {
            cudaDeviceProp prop;
            if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                if(prop.major >= 1) {
                    break;
                }
            }
        }
        if(i == count) {
            fprintf(stderr, "There is no device supporting CUDA.\n");
            return false;
        }
    } else i = user_prefered_i;
    if (cudaSetDevice(i) != cudaSuccess) {
        fprintf(stderr, "Error cudaSetDevice.\n");
        return false;
    }

    printf("CUDA initialized; device number %d.\n", i);
    return true;
}

#endif

}

#endif