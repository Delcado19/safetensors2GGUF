/* b3962 relied on transitive chrono includes; its build also forwards C++ flags
 * to C sources, so the forced include must be harmless in both languages. */
#ifdef __cplusplus
#include <chrono>
#endif
