DTYPE = -DTYPE=double
FTYPE = -DTYPE=float
NVCC = /usr/local/cuda/bin/nvcc
GCC = /usr/bin/gcc
GPP = /usr/bin/g++
GCC_kern = /usr/bin/gcc-5

NVCCFLAGS = -Wno-deprecated-gpu-targets -g -arch=sm_35 -std=c++11 -ccbin=g++-5
LIBFLAGS = --compiler-options -fPIC
GCCFLAGS = -g -std=c++11 -pthread
GCCFLAGS_R = -O3 -std=c++11 -pthread
GCCFLAGS_RN = -O3 -std=c++11 -Xcompiler -pthread
ICUDA = -I/usr/local/cuda/include
IPROJECT = -I source/
IBOOST = -I/home/noctum/boost_1_70/include/
IGMP = -I gmp/install/include

LCUDA = -L/usr/local/cuda/lib64
LBOOST = -L/home/noctum/boost_1_70/lib/
LIBS1 = -lcublas -lcurand 
LIBS2 = -lcufft -lcublas -lcurand 
LIBBOOST = -lboost_serialization
LLAPACK = -L/opt/OpenBLAS/lib -lopenblas
LGMP = -L gmp/install/lib -lgmp -lgmpxx


deb_d:
	$(GPP) $(GCCFLAGS) $(DTYPE) $(IPROJECT) source/test_linear_solver.cpp -o test_linear_solver.bin

deb_f:
	$(GPP) $(GCCFLAGS) $(FTYPE) $(IPROJECT) source/test_linear_solver.cpp -o test_linear_solver.bin

t_deb_d:
	$(GPP) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(IGMP) source/test.cpp $(LGMP) -o test.bin

t_deb_f:
	$(GPP) $(GCCFLAGS) $(FTYPE) $(IPROJECT) $(IGMP) source/test.cpp $(LGMP) -o test.bin

dot_F:
	$(GPP) $(GCCFLAGS_R) $(FTYPE) $(IPROJECT) $(IGMP) source/test_threads.cpp $(LGMP) -o test_threads.bin
dot_D:
	$(GPP) $(GCCFLAGS_R) $(DTYPE) $(IPROJECT) $(IGMP) source/test_threads.cpp $(LGMP) -o test_threads.bin
vec_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/gpu_vector_operations_kernels.cu -c -o ./gpu_vec_kers.o
vec_kern1:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/cuda_dot_product_kernels.cu -c -o ./cuda_dot_prod.o
reduction_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/gpu_reduction_kernels.cu -c -o ./cuda_reduction_kers.o

reduction_ogita_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/testing/gpu_reduction_ogita_kernels.cu -c -o ./cuda_reduction_ogita_kers.o


helper_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/generate_vector_pair_kernels_helper.cu -c -o ./cuda_helper_kers.o

vec_D:
	$(NVCC) $(GCCFLAGS_RN) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/test_vector_operations.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_dot_prod.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o test_vector_operations.bin
vec_F:
	$(NVCC) $(GCCFLAGS_RN) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/test_vector_operations.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_dot_prod.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o test_vector_operations.bin
bench_D:
	$(NVCC) $(GCCFLAGS_RN) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/obtain_condition_precision.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_dot_prod.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o benchmark.bin
bench_F:
	$(NVCC) $(GCCFLAGS_RN) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/obtain_condition_precision.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_dot_prod.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o benchmark.bin

