#ifndef __STATIC_TREE_BARRIER_GLOBAL_DEPARTURE_HPP_IS_INCLUDED__
#define __STATIC_TREE_BARRIER_GLOBAL_DEPARTURE_HPP_IS_INCLUDED__

#include <atomic>

/**
 * Static Tree Barrier With Global Departure Flag:
 * -----------------------------------------------
 *
 * This barrier uses the static tree barrier for the arrival part and spinning on a global atomic boolean flag in order to perform the departure stage.
 */

namespace barrier{

	// this must be aligned to cache-line boundaries
	class static_tree_barrier_global_departure{
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
			// which parent should i notify upon arrival?
			shared_flag* arrival_parent;
			// i need to give each of the children that i expect to arrive one flag
			// this needs care with hardware prefetchers
			std::vector<shared_flag> arrival_children_flag; 
			bool local_sense; // my local sense value
			char _local_sense_padding[CACHE_LINE_SIZE-sizeof(local_sense)];
		};

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

				// wait now until the root signals departure
				while (sense.load(std::memory_order_relaxed) != n->local_sense){}
				sense.load(std::memory_order_acquire); // sync memory
			}
			else{
				// i am the root signal the global departure
				sense.store(n->local_sense, std::memory_order_release);
			}

			n->local_sense = !n->local_sense;
		}
		
	private:
		std::atomic<bool> sense{true}; // the global sense value 
		char _sense_padding[CACHE_LINE_SIZE-sizeof(sense)];
	};
	

} // namespace barrier


#endif

