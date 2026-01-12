#include "perfcli/cuda/Stream.hpp"
#include "perfcli/util/Error.hpp"

namespace perfcli {

Stream::Stream() : stream_(nullptr), is_external_(false) {
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

Stream::Stream(cudaStream_t stream) : stream_(stream), is_external_(true) {}

Stream::~Stream() {
  if (!is_external_ && stream_) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      (void)err;
    }
  }
}

Stream::Stream(Stream&& other) noexcept
    : stream_(other.stream_), is_external_(other.is_external_) {
  other.stream_ = nullptr;
  other.is_external_ = false;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    if (!is_external_ && stream_) {
      cudaStreamDestroy(stream_);
    }
    stream_ = other.stream_;
    is_external_ = other.is_external_;
    other.stream_ = nullptr;
    other.is_external_ = false;
  }
  return *this;
}

void Stream::sync() const {
  if (stream_) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
}

Event::Event() : event_(nullptr) {
  CUDA_CHECK(cudaEventCreate(&event_));
}

Event::Event(unsigned int flags) : event_(nullptr) {
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

Event::~Event() {
  if (event_) {
    cudaError_t err = cudaEventDestroy(event_);
    if (err != cudaSuccess) {
      (void)err;
    }
  }
}

Event::Event(Event&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    if (event_) {
      cudaEventDestroy(event_);
    }
    event_ = other.event_;
    other.event_ = nullptr;
  }
  return *this;
}

void Event::record(cudaStream_t stream) const {
  CUDA_CHECK(cudaEventRecord(event_, stream));
}

void Event::sync() const {
  CUDA_CHECK(cudaEventSynchronize(event_));
}

float Event::elapsed_time_ms(const Event& start) const {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
  return ms;
}

EventTimer::EventTimer()
    : stream_(0), running_(false), elapsed_ms_(0.0f) {}

EventTimer::EventTimer(cudaStream_t stream)
    : stream_(stream), running_(false), elapsed_ms_(0.0f) {}

void EventTimer::start() {
  start_event_.record(stream_);
  running_ = true;
}

void EventTimer::stop() {
  stop_event_.record(stream_);
  running_ = false;
}

void EventTimer::sync() {
  if (!running_) {
    stop_event_.sync();
    elapsed_ms_ = stop_event_.elapsed_time_ms(start_event_);
  }
}

float EventTimer::elapsed_milliseconds() const {
  return elapsed_ms_;
}

float EventTimer::elapsed_microseconds() const {
  return elapsed_ms_ * 1000.0f;
}

std::vector<StreamPtr> create_streams(int count) {
  std::vector<StreamPtr> streams;
  streams.reserve(count);
  for (int i = 0; i < count; ++i) {
    streams.push_back(std::make_unique<Stream>());
  }
  return streams;
}

}
