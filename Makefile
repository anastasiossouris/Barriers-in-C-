CC=g++
CFLAGS= -c -std=c++11 -Wall -Wextra -g -O3 -fno-extern-tls-init
LIBS= -lpthread -latomic
INCLUDES=

all: intel_i7_benchmark_suite

intel_i7_benchmark_suite: intel_i7_benchmark_suite.o centralized_sense_reversing_barrier.o xorshift.o meanconf.o
	$(CC) -Wl,--no-as-needed -o intel_i7_benchmark_suite meanconf.o xorshift.o intel_i7_benchmark_suite.o centralized_sense_reversing_barrier.o $(LIBS)

meanconf.o: meanconf.cpp
	$(CC) $(CFLAGS) $(INCLUDES) meanconf.cpp -o meanconf.o


xorshift.o: xorshift.cpp
	$(CC) $(CFLAGS) $(INCLUDES) xorshift.cpp -o xorshift.o

intel_i7_benchmark_suite.o: intel_i7_benchmark_suite.cpp
	$(CC) $(CFLAGS) $(INCLUDES) intel_i7_benchmark_suite.cpp -o intel_i7_benchmark_suite.o

centralized_sense_reversing_barrier.o: centralized_sense_reversing_barrier.cpp
	$(CC) $(CFLAGS) $(INCLUDES) centralized_sense_reversing_barrier.cpp -o centralized_sense_reversing_barrier.o

clean:
	rm -rf *.o
