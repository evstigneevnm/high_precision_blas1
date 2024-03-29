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


#ifndef __CUDA_SUPPORT_H__
#define __CUDA_SUPPORT_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <stdarg.h>
#include <utils/cuda_safe_call.h>
#include <cuda_runtime.h>


inline int init_cuda(int PCI_ID)
{

    
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0)
    {
        fprintf(stderr, "There is no compartable device found.\n");
        return -1;
    }
    
    int deviceNumber=0;
    int deviceNumberTemp=0;
    
    if(count>1)
    {

        if(PCI_ID==-1)
        {
            for(i = 0; i < count; i++) 
            {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                printf( "#%i:   %s, pci-bus id:%i %i %i \n", i, (char*)&deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
            }            
            printf("Device number for it to use>>>\n");
            int num_scanded = scanf("%i", &deviceNumberTemp);
            if(num_scanded == 0)
            {
                throw(std::runtime_error("incorrect Device number provided!"));
            }
            
        }
        else
        {
            cudaDeviceProp deviceProp;
            for(int j=0;j<count;j++)
            {
                cudaGetDeviceProperties(&deviceProp, j);
                if(deviceProp.pciBusID==PCI_ID)
                {
                    deviceNumberTemp = j;
                    break;
                }
            }

            printf("Using %s@[%i:%i:%i]\n",(char*)&deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
        }
        deviceNumber=deviceNumberTemp;
    
    }
    else
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceNumber);
        printf( "#%i:   %s, pci-bus id:%i %i %i \n", deviceNumber, (char*)&deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
        printf( "       using it...\n");    
    }

    cudaSetDevice(deviceNumber);
    
    return deviceNumber;
}

//
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
//
//__host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) 
//

template <class T>
void host_2_device_cpy(T* device, T* host, int Nx, int Ny, int Nz)
{
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice));

}

template <class T>
void device_2_host_cpy(T* host, T* device, size_t size)
{
    CUDA_SAFE_CALL(cudaMemcpy(host, device, sizeof(T)*size, cudaMemcpyDeviceToHost));
}

template <class T>
void host_2_device_cpy(T* device, T* host, size_t size)
{
    CUDA_SAFE_CALL(cudaMemcpy(device, host, sizeof(T)*size, cudaMemcpyHostToDevice));
}

template <class T>
void device_2_device_cpy(const T* device_from, T* device_to, size_t size)
{
    CUDA_SAFE_CALL(cudaMemcpy(device_to, device_from, sizeof(T)*size, cudaMemcpyDeviceToDevice));
}


template <class T>
void device_2_host_cpy(T* host, T* device, int Nx, int Ny, int Nz)
{
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost));
}


template <class T>
T* device_allocate(int Nx, int Ny, int Nz)
{
    T* m_device;
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_device, mem_size));
    return m_device;    
}


template <class T>
T* device_allocate(size_t size)
{
    T* m_device;
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_device, sizeof(T)*size));
    return m_device;    
}

template <class T>
T* device_allocate_host(int Nx, int Ny, int Nz)
{
    T* m_device;
    int mem_size=sizeof(T)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&m_device, mem_size));
    return m_device;    
}


template <class T>
T* device_allocate_host(size_t size)
{
    T* m_device;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&m_device, sizeof(T)*size));
    return m_device;    
}

template <class T>
void device_allocate_all(int Nx, int Ny, int Nz, int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T** value=va_arg(ap, T**); /* Increments ap to the next argument. */
        T* temp=device_allocate<T>(Nx, Ny, Nz);
        value[0]=temp;      
    }
    va_end(ap);

}

template <class T>
void device_deallocate(T* array)
{
    CUDA_SAFE_CALL(cudaFree(array));
}

template <class T>
void device_deallocate_host(T* array)
{
    CUDA_SAFE_CALL(cudaFreeHost(array));
}


template <class T>
void device_deallocate_all(int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T* value=va_arg(ap, T*); /* Increments ap to the next argument. */
        CUDA_SAFE_CALL(cudaFree(value));
    }
    va_end(ap);
}


// host operations
template <class T>
T* host_allocate(int Nx, int Ny, int Nz)
{
    
    int size=(Nx)*(Ny)*(Nz);
    T* array;
    array=(T*)malloc(sizeof(T)*size);
    if ( !array )
    {
        throw std::runtime_error(std::string("host memory allocation failed"));
    }
    for(int j=0;j<size;j++)
        array[j]=(T)0;

    return array;
}

// host operations
template <class T>
T* host_allocate(size_t size)
{
    
    T* array;
    array=(T*)malloc(sizeof(T)*size);
    if ( !array )
    {
        throw std::runtime_error(std::string("host memory allocation failed"));
    }
    for(int j=0;j<size;j++)
        array[j]=(T)0;

    return array;
}


template <class T>
void host_allocate_all(int Nx, int Ny, int Nz, int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T** value= va_arg(ap, T**); /* Increments ap to the next argument. */
        T* temp=host_allocate<T>(Nx, Ny, Nz);
        value[0]=temp;      
    }
    va_end(ap);

}


template <class T>
void host_deallocate(T* array)
{
    free(array);
}

template <class T>
void host_deallocate_all(int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        T* value= va_arg(ap, T*); /* Increments ap to the next argument. */
        free(value);
    }
    va_end(ap);

}


#endif