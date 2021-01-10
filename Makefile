deb_d:
	g++ -g -std=c++11 -DTYPE=double -I source/ source/test_linear_solver.cpp -o test_linear_solver.bin

deb_f:
	g++ -g -std=c++11 -DTYPE=float -I source/ source/test_linear_solver.cpp -o test_linear_solver.bin


t_deb_d:
	g++ -g -std=c++11 -DTYPE=double -I source/ -I gmp/install/include/ source/test.cpp -L gmp/install/lib -lgmp -lgmpxx -o test.bin

t_deb_f:
	g++ -g -std=c++11 -DTYPE=float -I source/ -I gmp/install/include/ source/test.cpp -L gmp/install/lib -lgmp -lgmpxx -o test.bin


dot_d:
	g++ -g -std=c++14 -pthread -I source/ -DTYPE=double source/test_threads.cpp -o test_threads.bin

dot_F:
	g++ -O3 -std=c++14 -pthread -I source/ -I gmp/install/include -DTYPE=float source/test_threads.cpp -L gmp/install/lib -lgmp -lgmpxx -o test_threads.bin
dot_D:
	g++ -O3 -std=c++14 -pthread -I source/ -I gmp/install/include -DTYPE=double source/test_threads.cpp -L gmp/install/lib -lgmp -lgmpxx -o test_threads.bin
