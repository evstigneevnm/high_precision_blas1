DTYPE = -DTYPE=double
FTYPE = -DTYPE=float
NVCC = /usr/local/cuda/bin/nvcc
GCC = /usr/bin/gcc
GPP = /usr/bin/g++
GCC_kern = /usr/bin/g++
SM_CUDA = sm_35
CPP_STANDARD = c++14
TARGET = -g

NVCCFLAGS = -Wno-deprecated-gpu-targets $(TARGET) -arch=$(SM_CUDA) -std=$(CPP_STANDARD) -ccbin=$(GCC_kern)
LIBFLAGS = --compiler-options -fPIC
GCCFLAGS = $(TARGET) -std=$(CPP_STANDARD)
ICUDA = -I/usr/local/cuda/include
IPROJECT = -I source/
IBOOST = -I/home/noctum/boost_1_70/include/
IGMP = -I/opt/gmp/include
ICUMP = -I/opt/cump/include

LCUDA = -L/usr/local/cuda/lib64
LBOOST = -L/home/noctum/boost_1_70/lib/
LIBS1 = -lcublas -lcurand 
LIBS2 = -lcufft -lcublas -lcurand 
LIBCUSPARSE = -lcusparse
LIBBOOST = -lboost_serialization
LLAPACK = -L/opt/OpenBLAS/lib -lopenblas
LGMP = -L/opt/gmp/lib -lgmp -lgmpxx
LCUMP = -L/opt/cump/lib -lcump
LPTHREAD = -lpthread

test_batch2:
	$(GPP) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(IGMP) source/test_batch_bounds.cpp $(LGMP) -o test_batch_bounds.bin

linsolver_D:
	$(GPP) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(ICUDA) source/test_linear_solver.cpp $(LPTHREAD) -o test_linear_solver_D.bin

linsolver_F:
	$(GPP) $(GCCFLAGS) $(FTYPE) $(IPROJECT) $(ICUDA) source/test_linear_solver.cpp $(LPTHREAD) -o test_linear_solver_F.bin

linsolver:
	make linsolver_D linsolver_F

t_deb_d:
	$(GPP) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(IGMP) source/test.cpp $(LGMP) -o test.bin

t_deb_f:
	$(GPP) $(GCCFLAGS) $(FTYPE) $(IPROJECT) $(IGMP) source/test.cpp $(LGMP) -o test.bin

dot_F:
	$(GPP) $(GCCFLAGS) $(FTYPE) $(IPROJECT) $(IGMP) source/test_threads.cpp $(LGMP) -o test_threads.bin
dot_D:
	$(GPP) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(IGMP) source/test_threads.cpp $(LGMP) -o test_threads.bin
vec_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/gpu_vector_operations_kernels.cu -c -o ./gpu_vec_kers.o
reduction_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/gpu_reduction_kernels.cu -c -o ./cuda_reduction_kers.o

reduction_ogita_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/common/testing/gpu_reduction_ogita_kernels.cu -c -o ./cuda_reduction_ogita_kers.o

cump_kern:
	$(NVCC) -Wno-deprecated-declarations $(NVCCFLAGS) $(IPROJECT) $(ICUMP) source/high_prec/cump_blas_kernels.cu -c -o ./cump_kers.o

helper_kern:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/generate_vector_pair_kernels_helper.cu -c -o ./cuda_helper_kers.o
helper_kernC:
	$(NVCC) $(NVCCFLAGS) $(IPROJECT) source/generate_vector_pair_kernels_helper_complex.cu -c -o ./cuda_helper_kers_complex.o
vec_D:
	$(NVCC) $(NVCCFLAGS) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/test_vector_operations.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP) gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o cump_kers.o -o test_vector_operations_D.bin
vec_F:
	$(NVCC) $(NVCCFLAGS) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/test_vector_operations.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP) gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o  cump_kers.o -o test_vector_operations_F.bin

vecC_D:
	$(NVCC) $(NVCCFLAGS) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/test_vector_operations_complex.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers_complex.o cuda_reduction_ogita_kers.o  cump_kers.o -o test_vector_operations_complex_D.bin
vecC_F:
	$(NVCC) $(NVCCFLAGS) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/test_vector_operations_complex.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP) gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers_complex.o cuda_reduction_ogita_kers.o cump_kers.o -o test_vector_operations_complex_F.bin

bench_D:
	$(NVCC) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/obtain_condition_precision.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP)  gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o  cump_kers.o -o benchmark_D.bin
bench_F:
	$(NVCC) $(GCCFLAGS) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/obtain_condition_precision.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP) gpu_vec_kers.o cuda_reduction_kers.o cuda_helper_kers.o cuda_reduction_ogita_kers.o  cump_kers.o -o benchmark_F.bin

benchC_D:
	$(NVCC) $(GCCFLAGS) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/obtain_condition_precision_complex.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP)  gpu_vec_kers.o  cuda_reduction_kers.o cuda_reduction_ogita_kers.o cuda_helper_kers_complex.o  cump_kers.o -o benchmark_complex_D.bin
benchC_F:
	$(NVCC) $(GCCFLAGS) $(FTYPE) $(IPROJECT) $(ICUDA) $(IGMP) $(ICUMP) source/obtain_condition_precision_complex.cpp $(LGMP) $(LCUDA) $(LIBS2) $(LCUMP) gpu_vec_kers.o cuda_reduction_kers.o cuda_reduction_ogita_kers.o cuda_helper_kers_complex.o  cump_kers.o -o benchmark_complex_F.bin

test_reduction:
	$(NVCC) $(NVCCFLAGS) $(DTYPE) $(IPROJECT) $(ICUDA) $(IGMP) source/test_reduction.cpp $(LCUDA) $(LIBS1)  gpu_vec_kers.o cuda_reduction_kers.o cuda_reduction_ogita_kers.o  cump_kers.o -o test_reduction.bin

mat_csr_D:
	$(NVCC) $(DTYPE) $(NVCCFLAGS) $(IPROJECT) $(ICUDA) source/test_csr_matrix.cpp $(LCUDA) $(LIBS1) $(LIBCUSPARSE) gpu_vec_kers.o cuda_reduction_ogita_kers.o cuda_reduction_kers.o -o test_csr_matrix.bin
mat_pointers_D:
	$(NVCC) $(DTYPE) $(NVCCFLAGS) $(IPROJECT) $(ICUDA) source/test_csr_matrix_pointers.cu $(LCUDA) $(LIBS1) $(LIBCUSPARSE) -o test_csr_matrix_pointers.bin
lin_solver_csr_D:
	$(NVCC) $(DTYPE) $(NVCCFLAGS) $(IPROJECT) $(ICUDA) source/test_csr_linear_solver.cpp $(LCUDA) $(LIBS1) $(LIBCUSPARSE) gpu_vec_kers.o cuda_reduction_ogita_kers.o cuda_reduction_kers.o -o test_csr_linear_solver.bin
lin_solver_csr_F:
	$(NVCC) $(FTYPE) $(NVCCFLAGS) $(IPROJECT) $(ICUDA) source/test_csr_linear_solver.cpp $(LCUDA) $(LIBS1) $(LIBCUSPARSE) gpu_vec_kers.o cuda_reduction_ogita_kers.o cuda_reduction_kers.o -o test_csr_linear_solver.bin


