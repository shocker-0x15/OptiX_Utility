#pragma once

#include <chrono>
#include <vector>
#include <stdint.h>

using namespace std::chrono;

template <typename res, uint32_t MaxNumMeasurements = 512>
class StopWatchTemplate {
    std::vector<typename res::duration> m_measurements;
    std::vector<typename res::time_point> m_startTPStack;

public:
    enum class DurationType : uint32_t {
        Nanoseconds,
        Microseconds,
        Milliseconds,
        Seconds,
    };

    uint64_t durationCast(const typename res::duration &duration, DurationType dt) const {
        switch (dt) {
        case DurationType::Nanoseconds:
            return duration_cast<nanoseconds>(duration).count();
        case DurationType::Microseconds:
            return duration_cast<microseconds>(duration).count();
        case DurationType::Milliseconds:
            return duration_cast<milliseconds>(duration).count();
        case DurationType::Seconds:
            return duration_cast<seconds>(duration).count();
        default:
            break;
        }
        return UINT64_MAX;
    }

    void start() {
        m_startTPStack.push_back(res::now());
    }

    uint32_t stop() {
        uint32_t mIdx = 0xFFFFFFFF;
        if (m_measurements.size() < MaxNumMeasurements) {
            mIdx = static_cast<uint32_t>(m_measurements.size());
            m_measurements.push_back(res::now() - m_startTPStack.back());
        }
        m_startTPStack.pop_back();
        return mIdx;
    }

    uint64_t getMeasurement(uint32_t index, DurationType dt = DurationType::Milliseconds) const {
        if (index >= m_measurements.size())
            return UINT64_MAX;
        return durationCast(m_measurements[index], dt);
    }

    uint64_t getElapsed(DurationType dt = DurationType::Milliseconds) {
        typename res::duration duration = res::now() - m_startTPStack.back();
        return durationCast(duration, dt);
    }

    uint64_t getElapsedFromRoot(DurationType dt = DurationType::Milliseconds) {
        typename res::duration duration = res::now() - m_startTPStack.front();
        return durationCast(duration, dt);
    }

    void clearAllMeasurements() {
        m_measurements.clear();
    }
};

template <uint32_t MaxNumMeasurements = 512>
using StopWatch = StopWatchTemplate<system_clock, MaxNumMeasurements>;
template <uint32_t MaxNumMeasurements = 512>
using StopWatchHiRes = StopWatchTemplate<high_resolution_clock, MaxNumMeasurements>;
