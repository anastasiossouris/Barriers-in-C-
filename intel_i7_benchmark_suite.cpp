/**
 * This file implements a benchmark suite for the Intel i7 Sandy Bridge 2600K machine.
 *
 * The benchmark is invoked as:
 *	./prog_name BarrierClass OutFile
 * The benchmark then proceeds in using the barrier specified by BarrierClass and writing the results to a file named Outfile.
 *
 * The benchmark run is:
 *	For number of threads from 1 to 8
 *		for random workload from 1 to 1000000 [1,10,100,1000,10000,100000,1000000]
 *			Experiment: Have the threads perform 10.000 barrier episodes where each one performs random worlkoad in between as specified
 *			Repeat the experiment  30 times to measure the latency
 *
 * Thus the result of the above experiment is the latency of N threads performing 10.000 barrier episodes with a random workload of W. The output in the OutFile is: 
 *	NumThreads/Workload 1 10 100 ... 1000000
 *	1 ... (lower,mean,upper)
 *	2 ...
 *	3 ...
 * 	. ...
 *	8 ...
 */
#include <cassert>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <functional>
#include <utility>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <type_traits>
#include "cache_line_size.hpp"
#include "meanconf.hpp"
#include "profile.hpp"
#include "affinity.hpp"
#include "centralized_sense_reversing_barrier.hpp"
#include "static_tree_barrier.hpp"
#include "static_tree_barrier_global_departure.hpp"

/**
 * A helper object to simulate random workload.
 */
struct random_workload{
	using size_type = std::size_t;

	const size_type W; // the workload parameter

	std::uniform_int_distribution<size_type> dis;
	std::mt19937 gen;

	// random workload generation in the range [1,workload]. Random numbers start with the given seed.
 	// This is needed for reproducability of the results.
	random_workload(size_type workload, std::mt19937::result_type seed) : W{workload}, dis{1,W}, gen{seed} {}

	// produce random workload
	void operator()(){
		const size_type rnd_workload{dis(gen)};

		// volatile is needed to disable compiler optimizations
		for (volatile size_type i = 0; i < rnd_workload; ++i){}
	}
};

// The function returns a vector of vectors that contain the (lower,mean,upper) latencies. 
// The first vector denotes the number of threads and the inner vector the workload parameter
std::vector<std::vector<std::tuple<double,double,double> > > run_experiment_centralized_sense_reversing_barrier(){
	std::vector<std::vector<std::tuple<double,double,double> > > data;

	const std::size_t workloads [] = {1,10,100};
	const std::size_t workload_size = sizeof(workloads)/sizeof(workloads[0]);

	data.resize(8);
	
	for (std::size_t i = 0; i < data.size(); ++i){
		data[i].resize(workload_size);
	}

	auto thread_job = [](barrier::centralized_sense_reversing_barrier& barrier, std::size_t workload, std::mt19937::result_type seed, std::atomic<bool>& start_flag){	
		const std::size_t num_episodes = 10000;

		random_workload work{workload, seed};

		// wait until we are told to start
		while (!start_flag.load()){}

		for (std::size_t i = 0; i < num_episodes; ++i){
			work();
			barrier.await();
		}
	};

	std::cout << "Starting the experiment" << std::endl;

	barrier::internal::affinity aff_setter;

	for (std::size_t num_threads = 1; num_threads <= 8; ++num_threads){
		for (std::size_t workload_index = 0; workload_index < workload_size; ++workload_index){
			const std::size_t workload = workloads[workload_index];
			std::cout << "Executing experiment with " << num_threads << " threads and " << workload << " workload parameter." << std::endl;

			// with a confidence interval
			const std::size_t num_times{30};

			barrier::internal::confidence_interval mean(num_times);


			// create the random seeds for the threads. Each of the num_times times each thread must start with the same seed!
			// this is a requirement for reproducability
			std::vector<std::mt19937::result_type> seeds;
			 			
			std::mt19937 rnd(1337);

			for (std::size_t i = 0; i < num_threads; ++i){
				seeds.push_back(rnd());
			}

			for (std::size_t i = 0; i < num_times; ++i){
				std::cout << "\t..." << i;


				// create the barrier instance
				std::aligned_storage<sizeof(barrier::centralized_sense_reversing_barrier),CACHE_LINE_SIZE>::type barrier;
				
				new(&barrier) barrier::centralized_sense_reversing_barrier(num_threads); 

				// clear the caches
				{
					std::cout << "\tClearing caches" << std::endl;
					barrier::internal::cache_wiper cw;

					cw.clear_caches();
				}


				// create the threads
				std::vector<std::thread> threads;
				std::atomic<bool> start_flag{false};

				for (int j = 0; j < num_threads; ++j){
					std::thread t = std::thread{thread_job,std::ref(*static_cast<barrier::centralized_sense_reversing_barrier*>(
								static_cast<void*>(&barrier))), 
								workload, seeds[j], std::ref(start_flag)};
					std::thread::native_handle_type t_handle = t.native_handle();
					threads.push_back(std::move(t));

					aff_setter(num_threads, j, t_handle);
				}
					
				auto start_time = std::chrono::steady_clock::now();
				start_flag = true;
				// wait for the threads to finish
				std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
				auto end_time = std::chrono::steady_clock::now();
				double elapsed_time = std::chrono::duration<double,std::nano>(end_time-start_time).count();

				mean.add(elapsed_time);				
			}


			// now record the result
			data[num_threads-1][workload_index] = mean.mean();
		}
	}

	return data;
}


void write_data_to_file(std::vector<std::vector<std::tuple<double,double,double>>> data, std::string out_file){
	std::cout << "Writing data to file " << out_file << std::endl;

	std::ofstream out;

	out.open(out_file);

	try{
		out << "NumberOfThreads\\Workload 1\t\t10\t\t100\t\t1000\t\t10000\t\t100000\t\t1000000\n";

		for (auto i = 0; i < data.size(); ++i){
			out << i+1;

			for (const auto& m : data[i]){
				out << "\t" << std::get<0>(m) << " " << std::get<1>(m) << " " << std::get<2>(m);
			}

			out << "\n";
		}

		std::cout << "Data file was written successfully!" << std::endl;
	}
	catch(...){
		out.close();
		throw;
	}	
}
/*
void test(){
	// create the barrier instance
	std::aligned_storage<sizeof(barrier::centralized_sense_reversing_barrier),CACHE_LINE_SIZE>::type barrier;
				
	new(&barrier) barrier::centralized_sense_reversing_barrier(10000);

	barrier::centralized_sense_reversing_barrier& b =  *static_cast<barrier::centralized_sense_reversing_barrier*>(
								static_cast<void*>(&barrier));

	std::cout << "Address of barrier at " << std::addressof(b) << std::endl;
	std::cout << "Address of counter instance at " << std::addressof(b.counter) << std::endl;
	std::cout << "Address of sense instance at " << std::addressof(b.sense) << std::endl;

	exit(1);
}*/
/*
void test_thread_mapping(){
	auto thread_job = [](std::atomic<bool>& start_flag){
		while (!start_flag){}
		volatile int cnt = 0;
		for (volatile int i = 0; i < 1000000000; ++i){
			++cnt;
		}
	};

	barrier::internal::affinity aff_setter;

	for (int i = 1; i <= 8; ++i){
		std::vector<std::thread> threads;

		std::cout << "Starting " << i << " threads" << std::endl;
		
		std::atomic<bool> start_flag{false};

		for (int j = 0; j < i; ++j){
			std::thread t = std::thread{thread_job, std::ref(start_flag)};
			std::thread::native_handle_type t_handle = t.native_handle();
			threads.push_back(std::move(t));
			aff_setter(i, j, t_handle);
		}

		start_flag = true;
		// wait to finish
		std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	}

	exit(1);
}*/

void perf_friendly_version(){
	auto thread_job = [](barrier::centralized_sense_reversing_barrier& barrier, std::size_t workload, std::mt19937::result_type seed, std::atomic<bool>& start_flag){	
		const std::size_t num_episodes = 10000000;

		random_workload work{workload, seed};

		// wait until we are told to start
		while (!start_flag.load()){}

		for (std::size_t i = 0; i < num_episodes; ++i){
			work();
			barrier.await();
		}
	};

	std::cout << "Starting the experiment" << std::endl;

	barrier::internal::affinity aff_setter;

	const std::size_t num_threads = 8;

	const std::size_t workload = 100;
	std::cout << "Executing experiment with " << num_threads << " threads and " << workload << " workload parameter." << std::endl;

	// create the random seeds for the threads. Each of the num_times times each thread must start with the same seed!
	// this is a requirement for reproducability
	std::vector<std::mt19937::result_type> seeds;
			 			
	std::mt19937 rnd(1337);

	for (std::size_t i = 0; i < num_threads; ++i){
		seeds.push_back(rnd());
	}

	// create the barrier instance
	std::aligned_storage<sizeof(barrier::centralized_sense_reversing_barrier),CACHE_LINE_SIZE>::type barrier;
				
	new(&barrier) barrier::centralized_sense_reversing_barrier(num_threads); 

	// clear the caches
	{
		std::cout << "\tClearing caches" << std::endl;
		barrier::internal::cache_wiper cw;

		cw.clear_caches();
	}


	// create the threads
	std::vector<std::thread> threads;
	std::atomic<bool> start_flag{false};

	for (int j = 0; j < num_threads; ++j){
		std::thread t = std::thread{thread_job,std::ref(*static_cast<barrier::centralized_sense_reversing_barrier*>(
						static_cast<void*>(&barrier))), 
						workload, seeds[j], std::ref(start_flag)};
		std::thread::native_handle_type t_handle = t.native_handle();
		threads.push_back(std::move(t));

		aff_setter(num_threads, j, t_handle);
	}

	start_flag = true;
	// wait for the threads to finish
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));	


	std::exit(1);	
}

// the purpose of these 2 functions is to create the layout of the tree. 
// First: i know the mapping of thread identifiers to the cores (aff_setter).
// Thus, this funtion creates the nodes and returns a verctor of pointers to those nodes such that vec[0] is a pointer to the node
// that should be used by thread with logical id 0. Also the function connects the nodes together.
//
// The first version static_tree_layout_good_locality() makes a good locality whereas the static_tree_layout_bad_locality() makes a bad locality.

std::vector<barrier::static_tree_barrier::node* > static_tree_layout_good_locality(std::size_t num_threads){
	using raw_cache_aligned_node = std::aligned_storage<sizeof(barrier::static_tree_barrier::node),CACHE_LINE_SIZE>::type;
	std::vector<raw_cache_aligned_node*> cache_aligned_data;

	for (std::size_t i = 0; i < num_threads; ++i){
		cache_aligned_data.push_back(new raw_cache_aligned_node());

		// and construct a node there
		new (cache_aligned_data[i]) barrier::static_tree_barrier::node();
	}

	// now create the vector to return to the clients
	std::vector<barrier::static_tree_barrier::node* > nodes;

	for (std::size_t i = 0; i < num_threads; ++i){
		nodes.push_back(static_cast<barrier::static_tree_barrier::node*>(static_cast<void*>(cache_aligned_data[i])));
	}


	// perform initialization of the data regardless of the layout
	for (std::size_t i = 0; i < num_threads; ++i){
		nodes[i]->sense = true;
		nodes[i]->local_sense = false;
	}

	// do the layout.. and do it the hard way!!
	switch(num_threads){
	case 1:
		nodes[0]->arrival_parent = nullptr;
		break;
	case 2:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(1);
		nodes[0]->departure_children.push_back(&nodes[1]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		break;
	case 3:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[1]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		break;
	case 4:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[1]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);
		nodes[2]->departure_children.push_back(&nodes[3]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		break;
	case 5:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);
		nodes[0]->departure_children.push_back(&nodes[4]->sense);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(1);
		nodes[4]->departure_children.push_back(&nodes[1]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);
		nodes[2]->departure_children.push_back(&nodes[3]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		
		break;
	case 6:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);
		nodes[0]->departure_children.push_back(&nodes[4]->sense);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(2);
		nodes[4]->departure_children.push_back(&nodes[1]->sense);
		nodes[4]->departure_children.push_back(&nodes[5]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);
		nodes[2]->departure_children.push_back(&nodes[3]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];

		// for node 5
		nodes[5]->arrival_parent = &nodes[4]->arrival_children_flag[1];

		break;
	case 7:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);
		nodes[0]->departure_children.push_back(&nodes[4]->sense);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(2);
		nodes[4]->departure_children.push_back(&nodes[1]->sense);
		nodes[4]->departure_children.push_back(&nodes[5]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(2);
		nodes[2]->departure_children.push_back(&nodes[3]->sense);
		nodes[2]->departure_children.push_back(&nodes[6]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];

		// for node 5
		nodes[5]->arrival_parent = &nodes[4]->arrival_children_flag[1];

		// for node 6
		nodes[6]->arrival_parent = &nodes[2]->arrival_children_flag[1];
		break;
	case 8:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);
		nodes[0]->departure_children.push_back(&nodes[4]->sense);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(2);
		nodes[4]->departure_children.push_back(&nodes[1]->sense);
		nodes[4]->departure_children.push_back(&nodes[5]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(2);
		nodes[2]->departure_children.push_back(&nodes[3]->sense);
		nodes[2]->departure_children.push_back(&nodes[6]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		nodes[3]->arrival_children_flag.resize(1);
		nodes[3]->departure_children.push_back(&nodes[7]->sense);

		// for node 5
		nodes[5]->arrival_parent = &nodes[4]->arrival_children_flag[1];

		// for node 6
		nodes[6]->arrival_parent = &nodes[2]->arrival_children_flag[1];

		// for node 7
		nodes[7]->arrival_parent = &nodes[3]->arrival_children_flag[0];
		break;
	default:
		assert(0);
	};

	// perform initialization of shared flags
	for (std::size_t i = 0; i < num_threads; ++i){
		for (std::size_t j = 0; j < nodes[i]->arrival_children_flag.size(); ++j){
			nodes[i]->arrival_children_flag[j].flag = true;
		}
	}

	return std::move(nodes);
}

std::vector<barrier::static_tree_barrier::node* > static_tree_layout_bad_locality(std::size_t num_threads){
	using raw_cache_aligned_node = std::aligned_storage<sizeof(barrier::static_tree_barrier::node),CACHE_LINE_SIZE>::type;
	std::vector<raw_cache_aligned_node*> cache_aligned_data;

	for (std::size_t i = 0; i < num_threads; ++i){
		cache_aligned_data.push_back(new raw_cache_aligned_node());

		// and construct a node there
		new (cache_aligned_data[i]) barrier::static_tree_barrier::node();
	}

	// now create the vector to return to the clients
	std::vector<barrier::static_tree_barrier::node* > nodes;

	for (std::size_t i = 0; i < num_threads; ++i){
		nodes.push_back(static_cast<barrier::static_tree_barrier::node*>(static_cast<void*>(cache_aligned_data[i])));
	}


	// perform initialization of the data regardless of the layout
	for (std::size_t i = 0; i < num_threads; ++i){
		nodes[i]->sense = true;
		nodes[i]->local_sense = false;
	}

	// do the layout.. and do it the hard way!!
	switch(num_threads){
	case 1:
		nodes[0]->arrival_parent = nullptr;
		break;
	case 2:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(1);
		nodes[0]->departure_children.push_back(&nodes[1]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		break;
	case 3:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[1]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		break;
	case 4:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[3]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);
		nodes[2]->departure_children.push_back(&nodes[1]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[0]->arrival_children_flag[0];

		// for node 1
		nodes[1]->arrival_parent = &nodes[2]->arrival_children_flag[0];

		break;
	case 5:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[3]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);
		nodes[2]->departure_children.push_back(&nodes[1]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[3]->arrival_children_flag.resize(1);
		nodes[3]->departure_children.push_back(&nodes[4]->sense);		

		// for node 1
		nodes[1]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		
		// for node 4
		nodes[4]->arrival_parent = &nodes[3]->arrival_children_flag[0];

		break;
	case 6:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[3]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(2);
		nodes[2]->departure_children.push_back(&nodes[1]->sense);
		nodes[2]->departure_children.push_back(&nodes[5]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[3]->arrival_children_flag.resize(1);
		nodes[3]->departure_children.push_back(&nodes[4]->sense);		

		// for node 1
		nodes[1]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		
		// for node 4
		nodes[4]->arrival_parent = &nodes[3]->arrival_children_flag[0];

		// for node 5
		nodes[5]->arrival_parent = &nodes[2]->arrival_children_flag[1];
		
		break;
	case 7:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[3]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(2);
		nodes[2]->departure_children.push_back(&nodes[1]->sense);
		nodes[2]->departure_children.push_back(&nodes[5]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[3]->arrival_children_flag.resize(1);
		nodes[3]->departure_children.push_back(&nodes[4]->sense);		

		// for node 1
		nodes[1]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		
		// for node 4
		nodes[4]->arrival_parent = &nodes[3]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(1);
		nodes[4]->departure_children.push_back(&nodes[6]->sense);

		// for node 5
		nodes[5]->arrival_parent = &nodes[2]->arrival_children_flag[1];

		// for node 6
		nodes[6]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		break;
	case 8:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);
		nodes[0]->departure_children.push_back(&nodes[3]->sense);
		nodes[0]->departure_children.push_back(&nodes[2]->sense);

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(2);
		nodes[2]->departure_children.push_back(&nodes[1]->sense);
		nodes[2]->departure_children.push_back(&nodes[5]->sense);

		// for node 3
		nodes[3]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[3]->arrival_children_flag.resize(1);
		nodes[3]->departure_children.push_back(&nodes[4]->sense);		

		// for node 1
		nodes[1]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		
		// for node 4
		nodes[4]->arrival_parent = &nodes[3]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(2);
		nodes[4]->departure_children.push_back(&nodes[6]->sense);
		nodes[4]->departure_children.push_back(&nodes[7]->sense);

		// for node 5
		nodes[5]->arrival_parent = &nodes[2]->arrival_children_flag[1];

		// for node 6
		nodes[6]->arrival_parent = &nodes[4]->arrival_children_flag[0];
		
		// for node 7
		nodes[7]->arrival_parent = &nodes[4]->arrival_children_flag[1];

		break;
	default:
		assert(0);
	};

	// perform initialization of shared flags
	for (std::size_t i = 0; i < num_threads; ++i){
		for (std::size_t j = 0; j < nodes[i]->arrival_children_flag.size(); ++j){
			nodes[i]->arrival_children_flag[j].flag = true;
		}
	}

	return std::move(nodes);
}

std::vector<std::vector<std::tuple<double,double,double> > > run_experiment_static_tree_barrier(){
	std::vector<std::vector<std::tuple<double,double,double> > > data;

	const std::size_t workloads [] = {1,10,100};
	const std::size_t workload_size = sizeof(workloads)/sizeof(workloads[0]);

	data.resize(8);
	
	for (std::size_t i = 0; i < data.size(); ++i){
		data[i].resize(workload_size);
	}

	auto thread_job = [](barrier::static_tree_barrier& barrier, 
			    barrier::static_tree_barrier::node* node,
			    std::size_t workload, std::mt19937::result_type seed, std::atomic<bool>& start_flag){	
		const std::size_t num_episodes = 10000;

		random_workload work{workload, seed};

		// wait until we are told to start
		while (!start_flag.load()){}

		for (std::size_t i = 0; i < num_episodes; ++i){
			work();
			barrier.await(node);
		}
	};

	std::cout << "Starting the experiment" << std::endl;

	barrier::internal::affinity aff_setter;

	for (std::size_t num_threads = 1; num_threads <= 8; ++num_threads){
		for (std::size_t workload_index = 0; workload_index < workload_size; ++workload_index){
			const std::size_t workload = workloads[workload_index];
			std::cout << "Executing experiment with " << num_threads << " threads and " << workload << " workload parameter." << std::endl;

			// with a confidence interval
			const std::size_t num_times{30};

			barrier::internal::confidence_interval mean(num_times);


			// create the random seeds for the threads. Each of the num_times times each thread must start with the same seed!
			// this is a requirement for reproducability
			std::vector<std::mt19937::result_type> seeds;
			 			
			std::mt19937 rnd(1337);

			for (std::size_t i = 0; i < num_threads; ++i){
				seeds.push_back(rnd());
			}

			for (std::size_t i = 0; i < num_times; ++i){
				std::cout << "\t..." << i;


				// create the barrier instance
				std::aligned_storage<sizeof(barrier::static_tree_barrier),CACHE_LINE_SIZE>::type barrier;
				
				new(&barrier) barrier::static_tree_barrier(); 

				// clear the caches
				{
					std::cout << "\tClearing caches" << std::endl;
					barrier::internal::cache_wiper cw;

					cw.clear_caches();
				}

				// creating the nodes
				std::cout << "\t...Creating nodes..." << std::endl;
				std::vector<barrier::static_tree_barrier::node*> nodes = static_tree_layout_good_locality(num_threads);				
				//std::vector<barrier::static_tree_barrier::node*> nodes = static_tree_layout_bad_locality(num_threads);	

				// create the threads
				std::cout << "\t...Creating threads..." << std::endl;
				std::vector<std::thread> threads;
				std::atomic<bool> start_flag{false};

				for (int j = 0; j < num_threads; ++j){
					std::thread t = std::thread{thread_job,std::ref(*static_cast<barrier::static_tree_barrier*>(
								static_cast<void*>(&barrier))), 
								nodes[j],
								workload, seeds[j], std::ref(start_flag)};
					std::thread::native_handle_type t_handle = t.native_handle();
					threads.push_back(std::move(t));

					aff_setter(num_threads, j, t_handle);
				}
					
				auto start_time = std::chrono::steady_clock::now();
				start_flag = true;
				// wait for the threads to finish
				std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
				auto end_time = std::chrono::steady_clock::now();
				double elapsed_time = std::chrono::duration<double,std::nano>(end_time-start_time).count();

				mean.add(elapsed_time);	

				// release the nodes
				for (std::size_t i = 0; i < num_threads; ++i){
					delete nodes[i];
				}			
			}


			// now record the result
			data[num_threads-1][workload_index] = mean.mean();
		}
	}

	return data;
}

std::vector<barrier::static_tree_barrier_global_departure::node* > static_tree_global_departure_layout_good_locality(std::size_t num_threads){
	using raw_cache_aligned_node = std::aligned_storage<sizeof(barrier::static_tree_barrier_global_departure::node),CACHE_LINE_SIZE>::type;
	std::vector<raw_cache_aligned_node*> cache_aligned_data;

	for (std::size_t i = 0; i < num_threads; ++i){
		cache_aligned_data.push_back(new raw_cache_aligned_node());

		// and construct a node there
		new (cache_aligned_data[i]) barrier::static_tree_barrier_global_departure::node();
	}

	// now create the vector to return to the clients
	std::vector<barrier::static_tree_barrier_global_departure::node* > nodes;

	for (std::size_t i = 0; i < num_threads; ++i){
		nodes.push_back(static_cast<barrier::static_tree_barrier_global_departure::node*>(static_cast<void*>(cache_aligned_data[i])));
	}


	// perform initialization of the data regardless of the layout
	for (std::size_t i = 0; i < num_threads; ++i){
		nodes[i]->local_sense = false;
	}

	// do the layout.. and do it the hard way!!
	switch(num_threads){
	case 1:
		nodes[0]->arrival_parent = nullptr;
		break;
	case 2:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(1);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		break;
	case 3:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		break;
	case 4:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);

		// for node 1
		nodes[1]->arrival_parent = &nodes[0]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		break;
	case 5:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(1);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		
		break;
	case 6:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(2);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(1);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];

		// for node 5
		nodes[5]->arrival_parent = &nodes[4]->arrival_children_flag[1];

		break;
	case 7:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(2);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(2);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];

		// for node 5
		nodes[5]->arrival_parent = &nodes[4]->arrival_children_flag[1];

		// for node 6
		nodes[6]->arrival_parent = &nodes[2]->arrival_children_flag[1];
		break;
	case 8:
		// for node 0
		nodes[0]->arrival_parent = nullptr;
		nodes[0]->arrival_children_flag.resize(2);

		// for node 4
		nodes[4]->arrival_parent = &nodes[0]->arrival_children_flag[0];
		nodes[4]->arrival_children_flag.resize(2);

		// for node 1
		nodes[1]->arrival_parent = &nodes[4]->arrival_children_flag[0];

		// for node 2
		nodes[2]->arrival_parent = &nodes[0]->arrival_children_flag[1];
		nodes[2]->arrival_children_flag.resize(2);

		// for node 3
		nodes[3]->arrival_parent = &nodes[2]->arrival_children_flag[0];
		nodes[3]->arrival_children_flag.resize(1);

		// for node 5
		nodes[5]->arrival_parent = &nodes[4]->arrival_children_flag[1];

		// for node 6
		nodes[6]->arrival_parent = &nodes[2]->arrival_children_flag[1];

		// for node 7
		nodes[7]->arrival_parent = &nodes[3]->arrival_children_flag[0];
		break;
	default:
		assert(0);
	};

	// perform initialization of shared flags
	for (std::size_t i = 0; i < num_threads; ++i){
		for (std::size_t j = 0; j < nodes[i]->arrival_children_flag.size(); ++j){
			nodes[i]->arrival_children_flag[j].flag = true;
		}
	}

	return std::move(nodes);
}


std::vector<std::vector<std::tuple<double,double,double> > > run_experiment_static_tree_barrier_global_departure(){
	std::vector<std::vector<std::tuple<double,double,double> > > data;

	const std::size_t workloads [] = {1,10,100};
	const std::size_t workload_size = sizeof(workloads)/sizeof(workloads[0]);

	data.resize(8);
	
	for (std::size_t i = 0; i < data.size(); ++i){
		data[i].resize(workload_size);
	}

	auto thread_job = [](barrier::static_tree_barrier_global_departure& barrier, 
			    barrier::static_tree_barrier_global_departure::node* node,
			    std::size_t workload, std::mt19937::result_type seed, std::atomic<bool>& start_flag){	
		const std::size_t num_episodes = 10000;

		random_workload work{workload, seed};

		// wait until we are told to start
		while (!start_flag.load()){}

		for (std::size_t i = 0; i < num_episodes; ++i){
			work();
			barrier.await(node);
		}
	};

	std::cout << "Starting the experiment" << std::endl;

	barrier::internal::affinity aff_setter;

	for (std::size_t num_threads = 1; num_threads <= 8; ++num_threads){
		for (std::size_t workload_index = 0; workload_index < workload_size; ++workload_index){
			const std::size_t workload = workloads[workload_index];
			std::cout << "Executing experiment with " << num_threads << " threads and " << workload << " workload parameter." << std::endl;

			// with a confidence interval
			const std::size_t num_times{30};

			barrier::internal::confidence_interval mean(num_times);


			// create the random seeds for the threads. Each of the num_times times each thread must start with the same seed!
			// this is a requirement for reproducability
			std::vector<std::mt19937::result_type> seeds;
			 			
			std::mt19937 rnd(1337);

			for (std::size_t i = 0; i < num_threads; ++i){
				seeds.push_back(rnd());
			}

			for (std::size_t i = 0; i < num_times; ++i){
				std::cout << "\t..." << i;


				// create the barrier instance
				std::aligned_storage<sizeof(barrier::static_tree_barrier_global_departure),CACHE_LINE_SIZE>::type barrier;
				
				new(&barrier) barrier::static_tree_barrier_global_departure(); 

				// clear the caches
				{
					std::cout << "\tClearing caches" << std::endl;
					barrier::internal::cache_wiper cw;

					cw.clear_caches();
				}

				// creating the nodes
				std::cout << "\t...Creating nodes..." << std::endl;
				std::vector<barrier::static_tree_barrier_global_departure::node*> nodes = static_tree_global_departure_layout_good_locality(num_threads);				
				
				// create the threads
				std::cout << "\t...Creating threads..." << std::endl;
				std::vector<std::thread> threads;
				std::atomic<bool> start_flag{false};

				for (int j = 0; j < num_threads; ++j){
					std::thread t = std::thread{thread_job,std::ref(*static_cast<barrier::static_tree_barrier_global_departure*>(
								static_cast<void*>(&barrier))), 
								nodes[j],
								workload, seeds[j], std::ref(start_flag)};
					std::thread::native_handle_type t_handle = t.native_handle();
					threads.push_back(std::move(t));

					aff_setter(num_threads, j, t_handle);
				}
					
				auto start_time = std::chrono::steady_clock::now();
				start_flag = true;
				// wait for the threads to finish
				std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
				auto end_time = std::chrono::steady_clock::now();
				double elapsed_time = std::chrono::duration<double,std::nano>(end_time-start_time).count();

				mean.add(elapsed_time);	

				// release the nodes
				for (std::size_t i = 0; i < num_threads; ++i){
					delete nodes[i];
				}			
			}


			// now record the result
			data[num_threads-1][workload_index] = mean.mean();
		}
	}

	return data;
}

int main(int argc, const char* argv[]){	
	std::string out_file = "StaticTreeBarrierGlobalDepartureRelaxedWithGoodLocality";

	auto data = run_experiment_static_tree_barrier_global_departure();
	
	write_data_to_file(data, out_file);
	
	return (0);
}
