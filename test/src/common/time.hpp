/**
 * time.h
 * @author: HPE
 */

#ifndef UAL_COMMON_TIME_HPP_
#define UAL_COMMON_TIME_HPP_

#include <chrono>

struct TimeRecorder {
    using TimeStamp = std::chrono::high_resolution_clock::time_point;
    static TimeStamp now() { return std::chrono::high_resolution_clock::now(); }

    static int32_t nowMilliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(now().time_since_epoch())
                   .count() %
               1000;
    }

    static int32_t nowMicroseconds() {
        return std::chrono::duration_cast<std::chrono::microseconds>(now().time_since_epoch())
                   .count() %
               1000000;
    }

    static long long duration(const TimeStamp &start, const TimeStamp &end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
};

#endif  // UAL_COMMON_TIME_HPP_
