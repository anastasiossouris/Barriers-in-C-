#ifndef __AFFINITY_HPP_IS_INCLUDED__
#define __AFFINITY_HPP_IS_INCLUDED__ 1

#include <cassert>
#include <stdexcept>
#include <unistd.h>
#include <pthread.h>

namespace barrier{

namespace internal{

	struct affinity{		
		/**
		 * Set's the affinity of the thread with the given id to the passed core.
		 *
		 * \param core The core to which to set the affinity of the current thread
		 * \param id The identifier of the thread.
		 * \throw runtime_error If the affinity cannot be set
		 */
		void operator()(int core, pthread_t id){
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(core, &cpuset);

			if (pthread_setaffinity_np(id, sizeof(cpu_set_t), &cpuset)){
				throw std::runtime_error("failed to set affinity: call to pthread_setaffinity_np() failed");
			}
		}

		void operator()(int num_threads, int core, pthread_t id){
			// first fill cores then fill second threads (from left to right)
			// it just happens that the mapping is ok
			(*this)(core, id);
		}
	};

}

}
#endif
