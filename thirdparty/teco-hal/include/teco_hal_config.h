#ifndef TECO_HAL_INCLUDE_TECO_HAL_CONFIG_H_
#define TECO_HAL_INCLUDE_TECO_HAL_CONFIG_H_

#define HAL_DEVICE __device__
#define FORCE_INLINE inline __attribute__((always_inline))
#define NO_INLINE inline __attribute__((noinline))
#define WEAK __attribute__((weak))

#endif  // TECO_HAL_INCLUDE_TECO_HAL_CONFIG_H_
