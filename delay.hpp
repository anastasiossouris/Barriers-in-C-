#ifndef __DELAY_HPP_IS_INCLUDED__
#define __DELAY_HPP_IS_INCLUDED__ 1

#include <cstddef>

namespace barrier{

namespace internal{

		inline void delay(std::size_t amount){
			for (std::size_t i = 0; i < amount; ++i){
				__asm__ __volatile__("pause;");
			}
		}

} // namespace internal

} // namespace barrier

#endif
