#ifndef __CAMP_DEVICES_HPP
#define __CAMP_DEVICES_HPP

#include <cstring>
#include <memory>
#include <mutex>

#include <cuda_runtime.h>

#ifdef CAMP_ENABLE_HIP
#include <hip/hip_runtime.hpp>
#endif //#ifdef CAMP_ENABLE_HIP

namespace camp
{
namespace devices
{
inline namespace v1
{
  enum class Platform {
    undefined = 0,
    host = 1,
    omp = 2,
    tbb = 4,
    cuda = 8,
    hip = 16
  };

  class CudaEvent
  {
    public:
      CudaEvent(cudaStream_t stream){ 
	cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming); 
	cudaEventRecord(m_event, stream);
      }
      bool check() const { return (cudaEventQuery(m_event) == cudaSuccess); }
      void wait() const { cudaEventSynchronize(m_event); }
    private:
      cudaEvent_t m_event;
  };

#ifdef CAMP_ENABLE_HIP
  class HipEvent
  {
    public:
      HipEvent(hipStream_t stream){
	hipEventCreateWithFlags(&m_event, hipEventDisableTiming);
	hipEventRecord(m_event, stream);
      }
      bool check() const { return (hipEventQuery(m_event) == hipSuccess); }
      void wait() const { hipEventSynchronize(m_event); }
  };
#endif //#ifdef CAMP_ENABLE_HIP

  class HostEvent
  {
    public:
      HostEvent() {}
      bool check() const { return true; }
      void wait() const {}
    private:
  };

  class Event
  {
    public:
      template<typename T>
      Event(T&& value){ m_value.reset(new EventModel<T>(value));}

      bool check() const { return m_value->check(); }
      void wait() const { m_value->wait(); }

      template<typename T>
      T get() {
	auto result = dynamic_cast<EventModel<T>*>(m_value.get());
	if (result ==nullptr)
	{
	  std::runtime_error("Incompatible Event type get cast.");
	}
	return result->get();
      }

    private:
      class EventInterface {
	public:
	  virtual ~EventInterface(){}
	  virtual bool check() const = 0;
	  virtual void wait() const = 0;
      };

      template<typename T>
      class EventModel : public EventInterface {
	public:
	  EventModel(T const& modelVal) : m_modelVal(modelVal) {}
	  bool check() const override { return m_modelVal.check(); }
	  void wait() const override { m_modelVal.wait(); }
	  T get() { return m_modelVal; }
	private:
	  T m_modelVal;
      };

      std::shared_ptr<EventInterface> m_value;
  };

  class Cuda 
  {
    static cudaStream_t get_a_stream(int num)
    {
      static cudaStream_t streams[16] = {};
      static int previous = 0;

      static std::once_flag m_onceFlag;
      static std::mutex m_mtx;

      std::call_once(m_onceFlag,
	[] {
	  if (streams[0] == nullptr) {
	    for (auto &s : streams) {
	      cudaStreamCreate(&s);
	    }
	  }
	});

      if (num < 0) {
	m_mtx.lock();
        previous = (previous + 1) % 16;
	m_mtx.unlock();
        return streams[previous];
      }

      return streams[num % 16];
    }

  public:
    Cuda(int device = 0, int group = -1) : stream(get_a_stream(group)) {}

    // Methods
    Platform get_platform() { return Platform::cuda; }
    static Cuda &get_default()
    {
      static Cuda h;
      return h;
    }
    CudaEvent get_event() { 
      return CudaEvent(get_stream());
    }
    Event get_event_erased() {
      return Event{CudaEvent(get_stream())};
    }
    void wait() { cudaStreamSynchronize(stream); }
    void wait_on(Event *e) { e->wait(); }

    // Memory
    template <typename T>
    T *allocate(size_t size)
    {
      T *ret = nullptr;
      cudaMallocManaged(&ret, sizeof(T) * size);
      return ret;
    }
    void *calloc(size_t size)
    {
      void *p = allocate<char>(size);
      this->memset(p, 0, size);
      return p;
    }
    void free(void *p) { cudaFree(p); }
    void memcpy(void *dst, const void *src, size_t size)
    {
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
    }
    void memset(void *p, int val, size_t size)
    {
      cudaMemsetAsync(p, val, size, stream);
    }

    cudaStream_t get_stream() { return stream; }

  private:
    cudaStream_t stream;
  };

#ifdef CAMP_ENABLE_HIP
  class Hip
  {
    static hipStream_t get_a_stream(int num)
    {
      static hipStream_t streams[16] = {};
      static int previous = 0;

      static std::once_flag m_onceFlag;
      static std::mutex m_mtx;

      std::call_once(m_onceFlag,
	[] {
	  if (streams[0] == nullptr) {
	    for (auto &s : streams) {
	      hipStreamCreate(&s);
	    }
	  }
	});

      if (num < 0) {
	m_mtx.lock();
        previous = (previous + 1) % 16;
	m_mtx.unlock();
        return streams[previous];
      }

      return streams[num % 16];
    }

  public:
    Hip(int device = 0, int group = -1) : stream(get_a_stream(group)) {}

    // Methods
    Platform get_platform() { return Platform::hip; }
    static Hip &get_default()
    {
      static Hip h;
      return h;
    }
    HipEvent get_event() {
      return HipEvent(get_stream());
    }
    Event get_event_erased() {
      return Event{HipEvent(get_stream())};
    }
    void wait() { hipStreamSynchronize(stream); }
    void wait_on(Event *e) { e->wait(); }

    // Memory
    template <typename T>
    T *allocate(size_t size)
    {
      T *ret = nullptr;
      hipMallocManaged(&ret, sizeof(T) * size);
      return ret;
    }
    void *calloc(size_t size)
    {
      void *p = allocate<char>(size);
      this->memset(p, 0, size);
      return p;
    }
    void free(void *p) { hipFree(p); }
    void memcpy(void *dst, const void *src, size_t size)
    {
      hipMemcpyAsync(dst, src, size, hipMemcpyDefault, stream);
    }
    void memset(void *p, int val, size_t size)
    {
      hipMemsetAsync(p, val, size, stream);
    }

    hipStream_t get_stream() { return stream; }

  private:
    hipStream_t stream;
  };
#endif //#ifdef CAMP_ENABLE_HIP

  class Host
  {
  public:
    Host(int device = 0, int group = -1) {}

    // Methods
    Platform get_platform() { return Platform::host; }
    static Host &get_default()
    {
      static Host h;
      return h;
    }
    HostEvent get_event() { return HostEvent(); }
    Event get_event_erased() {
      Event e{HostEvent()};
      return e;
    }
    void wait() {}
    void wait_on(Event *e) { }

    // Memory
    template <typename T>
    T *allocate(size_t n)
    {
      return (T*)malloc(sizeof(T) * n);
    }
    void *calloc(size_t size)
    {
      void *p = allocate<char>(size);
      this->memset(p, 0, size);
      return p;
    }
    void free(void *p) { free(p); }
    void memcpy(void *dst, const void *src, size_t size)
    {
      memcpy(dst, src, size);
    }
    void memset(void *p, int val, size_t size) { std::memset(p, val, size); }
  };

  class Context
  {
    public:
      template<typename T>
      Context(T&& value){ m_value.reset(new ContextModel<T>(value));}
      template<typename T>
      T get() {
	auto result = dynamic_cast<ContextModel<T>*>(m_value.get()); 
	if (result ==nullptr)
	{
	  std::runtime_error("Incompatible Context type get cast.");
	}
	return result->get();
      }
      Platform get_platform() { return m_value->get_platform(); }
      template <typename T>
      T *allocate(size_t size)
      {
	return (T *)m_value->calloc(size * sizeof(T));
      }
      void *calloc(size_t size) { return m_value->calloc(size); }
      void free(void *p) { m_value->free(p); }
      void memcpy(void *dst, const void *src, size_t size)
      {
	m_value->memcpy(dst, src, size);
      }
      void memset(void *p, int val, size_t size) { m_value->memset(p, val, size); }
      Event get_event() { return m_value->get_event(); }
      void wait_on(Event *e) { m_value->wait_on(e); }

    private:
      class ContextInterface {
	public:
	  virtual ~ContextInterface(){}
	  virtual Platform get_platform() = 0;
	  virtual void *calloc(size_t size) = 0;
	  virtual void free(void *p) = 0;
	  virtual void memcpy(void *dst, const void *src, size_t size) = 0;
	  virtual void memset(void *p, int val, size_t size) = 0;
	  virtual Event get_event() = 0;
	  virtual void wait_on(Event *e) = 0;
      };

      template<typename T>
      class ContextModel : public ContextInterface {
	public:
	  ContextModel(T const& modelVal) : m_modelVal(modelVal) {}
          Platform get_platform() override { return m_modelVal.get_platform(); }
	  void *calloc(size_t size) override { return m_modelVal.calloc(size); }
	  void free(void *p) override { m_modelVal.free(p); }
	  void memcpy(void *dst, const void *src, size_t size) override
	  {
	    m_modelVal.memcpy(dst, src, size);
	  }
	  void memset(void *p, int val, size_t size) override
	  {
	    m_modelVal.memset(p, val, size);
	  }
	  Event get_event() { return m_modelVal.get_event_erased(); }
	  void wait_on(Event *e) { m_modelVal.wait_on(e); }
	  T get() { return m_modelVal; }
	private:
	  T m_modelVal;
      };

      std::shared_ptr<ContextInterface> m_value;
  };
}  // inline namespace v1
}  // namespace devices
}  // namespace camp
#endif /* __CAMP_DEVICES_HPP */
