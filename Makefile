DTYPE = -DTYPE=double
FTYPE = -DTYPE=float
NVCC = /usr/local/cuda/bin/nvcc
GCC = /usr/bin/gcc
GPP = /usr/bin/g++
GCC_kern = /usr/bin/g++-5
SM_CUDA = sm_35
CPP_STANDARD = c++11

NVCCFLAGS = -Wno-deprecated-gpu-targets -g -arch=$(SM_CUDA) -std=$(CPP_STANDARD) -ccbin=$(GCC_kern)
NVCCFLAGS_R = -Wno-deprecated-gpu-targets -O3 -arch=$(SM_CUDA) -std=$(CPP_STANDARD) -ccbin=$(GCC_kern)
LIBFLAGS = --compiler-options -fPIC
GCCFLAGS = -g -std=$(CPP_STANDARD) -pthread
GCCFLAGS_R = -O3 -std=$(CPP_STANDARD) -pthread
GCCFLAGS_RN = -O3 -std=$(CPP_STANDARD) -Xcompiler -pthread
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


test_batch2:
	$(GPP) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(IGMP) source/test_batch_bounds.cpp $(LGMP) -o test_batch_bounds.bin

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
# vec_kern1:
# 	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/cuda_dot_product_kernels.cu -c -o ./cuda_dot_prod.o
reduction_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/gpu_reduction_kernels.cu -c -o ./cuda_reduction_kers.o

reduction_ogita_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/testing/gpu_reduction_ogita_kernels.cu -c -o ./cuda_reduction_ogita_kers.o


helper_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/generate_vector_pair_kernels_helper.cu -c -o ./cuda_helper_kers.o

vec_D:
	$(NVCC) $(NVCCFLAGS_R) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/test_vector_operations.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o test_vector_operations.bin
vec_F:
	$(NVCC) $(NVCCFLAGS_R) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/test_vector_operations.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o test_vector_operations.bin

vecC_D:
	$(NVCC) $(NVCCFLAGS_R) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/test_vector_operations_complex.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o test_vector_operations_complex.bin
vecC_F:
	$(NVCC) $(NVCCFLAGS_R) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/test_vector_operations_complex.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o test_vector_operations_complex.bin

bench_D:
	$(NVCC) $(GCCFLAGS_RN) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/obtain_condition_precision.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o benchmark.bin
bench_F:
	$(NVCC) $(GCCFLAGS_RN) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/obtain_condition_precision.cpp $(LGMP) $(LCUDA) $(LIBS2)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o -o benchmark.bin

