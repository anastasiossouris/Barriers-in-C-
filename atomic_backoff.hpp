#ifndef __ATOMIC_BACKOFF_HPP_IS_INCLUDED__
#define __ATOMIC_BACKOFF_HPP_IS_INCLUDED__ 1

#include <cstddef>
#include <random>
#include <atomic>
#include <thread>
#include "delay.hpp"

namespace barrier{

namespace internal{

	template<class BackoffDerived>
	class backoff_base{
	public:
		using derived_type = BackoffDerived;

		void operator()() const{
			if (tries <= MAX_TRIES){
				self()->delay(tries);
				tries *= 2;
			}
			else{	
				std::this_thread::yield();
			}
		}

		void reset(){
			tries = 1;
		}

	private:
		static const std::size_t MAX_TRIES = 16;
		mutable std::size_t tries{1}; 

		const derived_type* self() const{ return static_cast<const derived_type*>(this); }
		derived_type* self(){ return static_cast<derived_type*>(this); }
	};

	//! No backoff case
	class no_backoff : public backoff_base<no_backoff>{
	public:
		void delay(std::size_t) const{}
	};

	//! Backoff for a fixed amount of iterations regardless of number of failures.
	class constant_backoff : public backoff_base<constant_backoff>{
	public:
		void delay(std::size_t) const{
			 barrier::internal::delay(CONSTANT_DELAY);		
		}
	private:
		static const std::size_t CONSTANT_DELAY = 16;
	};

	//! Backoff for as many iterations as the number of failures.
	class exponential_backoff : public backoff_base<exponential_backoff>{
	public:
		void delay(std::size_t tries) const{ barrier::internal::delay(tries); }
	};

	//! The default atomic backoff policy.
	using default_atomic_backoff = exponential_backoff;

} // namespace internal

} // namespace barrier

#endif
