

.. _using_camp-label: 

**********
Using Camp
**********

Since Camp is a metaprogramming library that is primarily header-only, it is hard
to find isolated code examples that show Camp's capability. Instead, this page
shows several real application examples of how Camp is being used currently.

Camp Used in Umpire
===================

Umpire's Operations provide an abstract interface to modifying and moving data between Umpire allocators.
Camp makes these operations easy to port regardless of the underlying hardware. For example:

.. code-block:: bash

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

.. note::

   The new ``ResourceAwarePool`` feature in Umpire will be using both Camp resources and Camp events to
   keep track of the state of different allocations within the pool. Without these Camp features, it would
   be much harder to keep track of the state of the allocations in a portable, hardware-agnostic way.

Camp Used in RAJA
=================

One example of using Camp in RAJA is the following code block which helps RAJA determine which backend
is being used in the ``RAJA::Launch``.

.. code-block:: bash

   // Helper function to retrieve a resource based on the run-time policy - if a device is active
   #if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_SYCL)
     template<typename T, typename U>
     RAJA::resources::Resource Get_Runtime_Resource(T host_res, U device_res, RAJA::ExecPlace device){
       if(device == RAJA::ExecPlace::DEVICE) {return RAJA::resources::Resource(device_res);}
       else { return RAJA::resources::Resource(host_res); }
       }
   #endif

See the full example `here<https://github.com/LLNL/RAJA/blob/develop/include/RAJA/pattern/launch/launch_core.hpp>`_.

Camp Used in CHAI
=================

Camp is also used within CHAI, another library in the RAJA Portability Suite, for operations like move and copy. Below
is an example of Camp used in Chai's ``ArrayManager``:

.. code-block:: bash

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

See the full example `here<https://github.com/LLNL/CHAI/blob/7ba2ba89071bf836071079929af7419da475ba27/src/chai/ArrayManager.cpp#L246>`_.

Many codes at LLNL use the different libraries within the RAJA Portability Suite. Camp plays a vital role
in the compiler abstractions that make the RAJA Portability Suite possible.
