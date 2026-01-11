#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace perfcli {

class Stream {
public:
  Stream();
  explicit Stream(cudaStream_t stream);
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  [[nodiscard]] bool is_external() const noexcept { return is_external_; }

  void sync() const;

private:
  cudaStream_t stream_;
  bool is_external_;
};

class Event {
public:
  Event();
  explicit Event(unsigned int flags);
  ~Event();

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

  void record(cudaStream_t stream) const;
  void sync() const;
  [[nodiscard]] float elapsed_time_ms(const Event& start) const;

private:
  cudaEvent_t event_;
};

class EventTimer {
public:
  EventTimer();
  explicit EventTimer(cudaStream_t stream);

  void start();
  void stop();
  void sync();
  [[nodiscard]] float elapsed_milliseconds() const;
  [[nodiscard]] float elapsed_microseconds() const;

private:
  cudaStream_t stream_;
  Event start_event_;
  Event stop_event_;
  bool running_;
  float elapsed_ms_;
};

using StreamPtr = std::unique_ptr<Stream>;

[[nodiscard]] std::vector<StreamPtr> create_streams(int count);

}
