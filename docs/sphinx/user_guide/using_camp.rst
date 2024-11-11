.. _using_camp-label: 

**********
Using Camp
**********

Since Camp is a metaprogramming library that is primarily header-only, it is hard
to find isolated code examples that show Camp's capability. Instead, this page
shows several real application examples of how Camp is being used currently.

Camp Used in Umpire
===================

Umpire is a resource management library that allows the discovery, provision, and management of memory on machines 
with multiple memory devices like NUMA and GPUs. Umpire is also part of the RAJA Portability Suite.
Umpire's Operations provide an abstract interface to modifying and moving data between Umpire allocators.
Camp makes these operations easy to port regardless of the underlying hardware. For example:

.. code-block:: cpp

  camp::resources::EventProxy<camp::resources::Resource> CudaMemsetOperation::apply_async(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value, std::size_t length,
    camp::resources::Resource& ctx)
  {
    auto device = ctx.try_get<camp::resources::Cuda>();
    if (!device) {
      UMPIRE_ERROR(resource_error,
                 fmt::format("Expected resources::Cuda, got resources::{}", platform_to_string(ctx.get_platform())));
    }
    auto stream = device->get_stream();

    cudaError_t error = ::cudaMemsetAsync(src_ptr, value, length, stream);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(
        runtime_error,
        fmt::format("cudaMemsetAsync( src_ptr = {}, value = {}, length = {}, stream = {}) failed with error: {}",
                    src_ptr, value, length, cudaGetErrorString(error), (void*)stream));
    }

    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

See the full example `here <https://github.com/LLNL/Umpire/blob/5bf5bc182f1e6ee3f6be1d953b68451d3ddc35f5/src/umpire/op/CudaMemsetOperation.cpp>`_.

.. note::

   The new ``ResourceAwarePool`` feature in Umpire will be using both Camp resources and Camp events to
   keep track of the state of different allocations within the pool. Without these Camp features, it would
   be much harder to keep track of the state of the allocations in a portable, hardware-agnostic way.

Camp Used in RAJA
=================

RAJA is a library of C++ software abstractions, primarily developed at Lawrence Livermore National Laboratory (LLNL), that enables 
architecture and programming model portability for HPC applications.
One of many examples of using Camp in RAJA is the following code block which helps RAJA determine which backend
is being used in the ``RAJA::Launch`` abstraction.

.. code-block:: cpp

   // Helper function to retrieve a resource based on the run-time policy - if a device is active
   #if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_SYCL)
     template<typename T, typename U>
     RAJA::resources::Resource Get_Runtime_Resource(T host_res, U device_res, RAJA::ExecPlace device){
       if(device == RAJA::ExecPlace::DEVICE) {return RAJA::resources::Resource(device_res);}
       else { return RAJA::resources::Resource(host_res); }
       }
   #endif

See the full example `here <https://github.com/LLNL/RAJA/blob/develop/include/RAJA/pattern/launch/launch_core.hpp>`_.

Camp Used in RAJAPerf
=====================

RAJAPerf is RAJA's Performance Suite designed to explore the performance of loop-based computational kernels found in HPC applications.
Specifically, it is used to assess and monitor runtime performance of kernels implemented using RAJA and compare those to variants 
implemented using common parallel programming models, such as OpenMP and CUDA, directly.
Camp has also been included in RAJAPerf as a way to easily determine which stream to run a RAJA kernel:

.. code-block:: cpp

   #if defined(RAJA_ENABLE_CUDA)
     camp::resources::Cuda getCudaResource()
     {
       if (run_params.getGPUStream() == 0) {
         return camp::resources::Cuda::CudaFromStream(0);
       }
       return camp::resources::Cuda::get_default();
     }
   #endif

See the full example `here <https://github.com/LLNL/RAJAPerf/blob/abb07792a899f7417e77ea40015e7e1dfd52716e/src/common/KernelBase.hpp>`_.

Camp Used in CHAI
=================

CHAI is a library that handles automatic data migration to different memory spaces behind an array-style interface. It was designed to 
work with RAJA and integrates well with it, though CHAI could be used with other C++ abstractions as well.
Just like Camp and Umpire, CHAI is part of the RAJA Portability Suite and uses Camp for operations like move and copy. Below
is an example of Camp used in CHAI's ``ArrayManager``:

.. code-block:: cpp

   static void copy(void * dst_pointer, void * src_pointer, umpire::ResourceManager & manager, ExecutionSpace dst_space, ExecutionSpace src_space) {

   #ifdef CHAI_ENABLE_CUDA
     camp::resources::Resource device_resource(camp::resources::Cuda::get_default());
   #elif defined(CHAI_ENABLE_HIP)
     camp::resources::Resource device_resource(camp::resources::Hip::get_default());
   #else
     camp::resources::Resource device_resource(camp::resources::Host::get_default());
   #endif

     camp::resources::Resource host_resource(camp::resources::Host::get_default());
     if (dst_space == GPU || src_space == GPU) {
       // Do the copy using the device resource
       manager.copy(dst_pointer, src_pointer, device_resource);
     } else {
       // Do the copy using the host resource
       manager.copy(dst_pointer, src_pointer, host_resource);
     }
     // Ensure device to host copies are synchronous
     if (dst_space == CPU && src_space == GPU) {
       device_resource.wait();
     }
   }

See the full example `here <https://github.com/LLNL/CHAI/blob/7ba2ba89071bf836071079929af7419da475ba27/src/chai/ArrayManager.cpp#L246>`_.

Many codes at LLNL use the different libraries within the RAJA Portability Suite. Camp plays a vital role
in the compiler abstractions that make the RAJA Portability Suite possible.
