#include "centralized_sense_reversing_barrier.hpp"

namespace barrier{

	thread_local bool centralized_sense_reversing_barrier::local_sense = false;

} // namespace barrier
