#ifndef __CAMP_DEVICES_HPP
#define __CAMP_DEVICES_HPP

#include <cstring>
#include <memory>

#include <cuda_runtime.h>
namespace camp
{
namespace devices
{

  enum class Platform {
    undefined = 0,
    host = 1,
    omp = 2,
    tbb = 4,
    cuda = 8,
    // omp_target = 16, not sure this is a meaningful difference
    hip = 32
  };

  class Host
  {
  public:
    Host(int device = 0, int group = -1) {}

    // Methods
    Platform get_platform() { return Platform::host; }
    Host &get_default()
    {
      static Host h;
      return h;
    }
    void wait()
    {
      // nothing to wait for, sequential/simd host is always synchronous
    }
    // Memory
    template <typename T>
    T *allocate(size_t n)
    {
      return reinterpret_cast<T *>(a.allocate(n * sizeof(T)));
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
  class Omp : public Host
  {
    // TODO: see if using fake addresses is an issue
    char *dep = nullptr;

  public:
    Omp(int device = 0, int group = -1) : dep((char *)group) {}

    // Methods
    Platform get_platform() { return Platform::omp; }
    Omp &get_default()
    {
      static Omp h;
      return h;
    }
    void wait()
    {
// TODO: see if taskwait depend has wide enough support
#pragma omp task if (0) depend(dep[0])
      {
      }
      // #pragma omp taskwait depend(dep[0])
    }
    char *get_dep() { return dep; }
    // Memory: inherited from Host
    void memset(void *p, int val, size_t size)
    {
      if (omp_get_level() != 0) {
        ::std::memset(p, val, size);
      } else {
        char *c = (char *)p;
#pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
          c[i] = val;
        }
      }
    }
  };

  class Cuda
  {
    static cudaStream_t get_a_stream(int num)
    {
      // TODO consider pool size
      static cudaStream_t streams[16] = {};
      static int previous = 0;
      // TODO deal with parallel init
      if (streams[0] == nullptr) {
        for (auto &s : streams) {
          cudaStreamCreate(&s);
        }
      }

      if (num < 0) {
        previous = (previous + 1) % 16;
        return streams[previous];
      }

      return streams[num % 16];
    }

  public:
    Cuda(int device = 0, int group = -1) : stream(get_a_stream(group)) {}

    // Methods
    Platform get_platform() { return Platform::cuda; }
    Cuda &get_default()
    {
      static Cuda h;
      return h;
    }
    void wait() { cudaStreamSynchronize(stream); }
    template <typename T>
    T *allocate(size_t size)
    {
      void *ret = nullptr;
      cudaMalloc(&ret, sizeof(T) * size);
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

  class Device
  {
    class dev_wrapper_base
    {
    public:
      virtual Platform get_platform();
      // virtual Cuda &get_default(); // not sure how to do this
      virtual void wait();
      virtual void *calloc(size_t size);
      virtual void free(void *p);
      virtual void memcpy(void *dst, const void *src, size_t size);
      virtual void memset(void *p, int val, size_t size);
    };
    template <typename D>
    class dev_wrapper : public dev_wrapper_base
    {
      D dev;

    public:
      dev_wrapper(D d) : dev(d) {}
      Platform get_platform() override { return dev.get_platform(); }
      void wait() override { dev.wait(); }
      void *calloc(size_t size) override { return dev.calloc(size); }
      void free(void *p) override { dev.free(p); }
      void memcpy(void *dst, const void *src, size_t size) override
      {
        dev.memcpy(dst, src, size);
      }
      void memset(void *p, int val, size_t size) override
      {
        dev.memset(p, val, size);
      }
    };

    std::shared_ptr<dev_wrapper_base> d;

  public:
    template <typename T>
    Device(T dev) : d(std::make_shared(dev_wrapper<T>(dev)))
    {
    }
    Platform get_platform() { return d->get_platform(); }
    void wait() { d->wait(); }
    template <typename T>
    T *allocate(size_t size)
    {
      return (T *)d->calloc(size * sizeof(T));
    }
    void *calloc(size_t size) { return d->calloc(size); }
    void free(void *p) { d->free(p); }
    void memcpy(void *dst, const void *src, size_t size)
    {
      d->memcpy(dst, src, size);
    }
    void memset(void *p, int val, size_t size) { d->memset(p, val, size); }
  };

}  // namespace devices
}  // namespace camp
#endif /* __CAMP_DEVICES_HPP */
