#ifndef __STATIC_TREE_BARRIER_HPP_IS_INCLUDED__
#define __STATIC_TREE_BARRIER_HPP_IS_INCLUDED__ 1

#include <cassert>
#include <vector>
#include <atomic>
#include "cache_line_size.hpp"

namespace barrier{

	/**
	 * Static Tree Barrier:
	 * --------------------
	 *
	 * In this barrier each thread has assigned a unique static_tree_barrier_node object. Those nodes form two trees: a arrival tree and a departure tree. 
	 * The shape of those trees, the assignment of threads to those nodes as well as the placement of those nodes depends on the architecture and must be implemented
	 * for example using the Portable Hardware Locality Library.
	 *
	 * Each thread has the following data:
	 *	(1) Which parent should i notify upon my arrival? atomic_bool* arrival_parent
	 *	(2) How many children should i expect for the arrival stage and where will they inform me?
	 *		intmax_t num_arrival_children; 
	 *		atomic_bool* arrival_children_flag; // array of booleans to be used by my children
	 *	(3) Where should my parent notify me about the departure stage?
	 *		atomic_bool sense;
	 *	(4) Which children should i notify for the departure?
	 *		intmax_t num_departure_children;
	 *		struct static_tree_barrier_node** departure_children;
	 *	(5) Which is my local sense? bool local_sense
	 *
	 * Data Packing:
	 * ------------
	 * I pack all those data in the same structure static_tree_barrier_node. The static tree barrier itslef is not represented by any structure because the location of the nodes
	 * must be made carefully and is up to the client. The await() function takes as parameter a pointer to the node for the calling thread.
	 *
	 * Alignment Requirements:
	 * ----------------------
	 * The static_tree_barrier_node itself should be allocated in cache-aligned storage. Also, each node should be allocated to the closest memory module for the thread that will
	 * use that node. Also, avoid having the nodes allocated in an array because a hardware prefetcher may prefetch more cache lines and thus introduce false-sharing.
	 * Inside each node now i took care not to introduce false-sharing. With the assumption that each node is cache-aligned then the atomic sense value is free of false-sharing.
	 * The flags, however, arrival_children_flag is another story. Each one must be cache-aligned to avoid false-sharing between the children that are notifying the node of
	 * their arrival. The implementation uses a struct shared_flag that is padded to a cache line and then allocates an array of those shared_flags. A hardware prefetcher now 
	 * could be an issue and tests must be made to validate the hypothesis.
	 *
	 * Usage:
	 * -----
	 */

	class static_tree_barrier{
	public:
		using size_type = unsigned int;

		struct shared_flag{
			std::atomic<bool> flag;
			char _padding[CACHE_LINE_SIZE-sizeof(flag)];

			shared_flag(){
			 	flag = true;
			}

			shared_flag(const shared_flag& other){
				// just do nothing.. this is needed so that i can use the shared_flag in vectors
				// since atomics have a deleted copy constructor
			}
		};
 
		// each node should be allocated in cache-line boundaries
		struct node{
			// where i expect my parent to signal me departure
			std::atomic<bool> sense;
			char _sense_padding[CACHE_LINE_SIZE-sizeof(sense)];
			// which parent should i notify upon arrival?
			shared_flag* arrival_parent;
			// i need to give each of the children that i expect to arrive one flag
			// this needs care with hardware prefetchers
			std::vector<shared_flag> arrival_children_flag; 
			// which children must i notify upon departure?
			std::vector<std::atomic<bool>* > departure_children;
			bool local_sense; // my local sense value
			char _local_sense_padding[CACHE_LINE_SIZE-sizeof(local_sense)];
		};


		#if 1
		void await(node* n){
			// relaxed version
			assert(n != nullptr);
			// wait until my children have arrived
			for (auto& flag : n->arrival_children_flag){
				while (flag.flag.load(std::memory_order_relaxed) != n->local_sense){}
				flag.flag.load(std::memory_order_acquire); // sync memory
			}

			// note: in the version presented in Shared Memory Synchronization Synthesis Lectures, here the thread re-sets the children flags to true. I instead
			// use the local sense value to avoid having to perform those stores and reducing perhaps the overall latency for the thread.

			// Inform my parent of my subtree's arrival and pass it the memory
			if (n->arrival_parent){
				n->arrival_parent->flag.store(n->local_sense, std::memory_order_release);

				// wait now until my parent signals departure
				while (n->sense.load(std::memory_order_relaxed) != n->local_sense){}
				n->sense.load(std::memory_order_acquire); // sync memory
			}

			// now its time to signal children on departure tree
			for (auto sig : n->departure_children){
				sig->store(n->local_sense, std::memory_order_release); // also sync memory
			}

			n->local_sense = !n->local_sense;
		}
		#endif

		#if 0
		void await(node* n){
			// seq-cst version
			assert(n != nullptr);
			// wait until my children have arrived
			for (auto& flag : n->arrival_children_flag){
				while (flag.flag.load() != n->local_sense){}
			}

			// note: in the version presented in Shared Memory Synchronization Synthesis Lectures, here the thread re-sets the children flags to true. I instead
			// use the local sense value to avoid having to perform those stores and reducing perhaps the overall latency for the thread.

			// Inform my parent of my subtree's arrival and pass it the memory
			if (n->arrival_parent){
				n->arrival_parent->flag.store(n->local_sense);

				// wait now until my parent signals departure
				while (n->sense.load() != n->local_sense){}
			}

			// now its time to signal children on departure tree
			for (auto sig : n->departure_children){
				sig->store(n->local_sense); // also sync memory
			}

			n->local_sense = !n->local_sense;
		}
		#endif

	};
	

} // namespace barrier

#endif
