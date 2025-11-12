.. # Copyright (c) 2018-2025, Lawrence Livermore National Security, LLC and
.. # other Camp project contributors. See the camp/LICENSE file for details.
.. #
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _using_camp-label: 

**********
Using Camp
**********

Since Camp is a metaprogramming library that is primarily header-only, it is hard
to find isolated code examples that show Camp's capability. Instead, this page
shows several real application examples of how Camp is being used currently. Since Camp
is a metaprogramming library and is very low-level, these examples show detailed implementation code that is not
user-facing. However, these examples should be useful for understanding how Camp functionality can be
properly utilized.

More information about Camp the ``Resource`` and ``EventProxy`` can be found on the :ref:`resources-label` page.

There are additional, more basic examples of other Camp features on the :ref:`features-label` pages.

Camp Used in Umpire
===================

`Umpire <https://github.com/LLNL/Umpire>`_ is a resource management library that allows the discovery, provision, and management of memory on machines 
with multiple memory devices like NUMA and GPUs. Umpire is also part of the RAJA Portability Suite.
Umpire's Operations provide an abstract interface to modifying and moving data between Umpire Allocators.
Camp makes these operations easy to port regardless of the underlying hardware. For simplicity, the below example
just shows Umpire's ``CudaMemsetOperation``, but the code for Umpire's Hip, OpenMP, etc. variations of the Memset
operation have very few code changes.

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

The above code snippet shows how Umpire uses Camp's ``EventProxy`` and ``Resource`` features. This function is passed
the ``ctx`` parameter which is a Camp ``Resource``. It then uses the ``try_get`` method in an attempt to get the typed
``Resource`` and if it can't, it throws an error. From there, we can call other method functions like ``get_stream()``
on the typed ``Resource``.

See the full example `here <https://github.com/LLNL/Umpire/blob/5bf5bc182f1e6ee3f6be1d953b68451d3ddc35f5/src/umpire/op/CudaMemsetOperation.cpp>`_.

.. note::

   The new ``ResourceAwarePool`` feature in Umpire will be using both Camp resources and Camp events to
   keep track of the state of different allocations within the pool. Without these Camp features, it would
   be much harder to keep track of the state of the allocations in a portable, hardware-agnostic way.

Camp Used in RAJA
=================

`RAJA <https://github.com/LLNL/RAJA>`_ is a library of C++ software abstractions that enables 
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

In this example, RAJA is using a Camp ``Resource`` to determine where to execute the ``RAJA::Launch``. Depending on 
the value of ``device`` it will return a Camp device or host ``Resource``.

See the full example `here <https://github.com/LLNL/RAJA/blob/develop/include/RAJA/pattern/launch/launch_core.hpp>`_.

RAJA also uses Camp ``tuple`` features to set up RAJA Views:

.. code-block:: cpp

   template<typename IdxLin, typename...DimTypes>
   struct add_offset<RAJA::TypedLayout<IdxLin,camp::tuple<DimTypes...>>>
   {
     using type = RAJA::TypedOffsetLayout<IdxLin,camp::tuple<DimTypes...>>;
   };

Learn more about the Camp ``tuple`` feature on the :ref:`tuple-label` page.

See the full example `here <https://github.com/LLNL/RAJA/blob/0aef7cc44348d82e94e73e12f77c27ea306e47b8/include/RAJA/util/View.hpp>`_.

RAJA also uses Camp for error checking and generating error strings for cuda
and hip API functions.

.. code-block:: cpp

   CAMP_HIP_API_INVOKE_AND_CHECK(hipLaunchKernel, func, gridDim, blockDim, args, shmem, res.get_stream());
   // C++ exception with description "HIP error: invalid configuration argument hipLaunchKernel(func=0x273ff0, gridDim={1,2,3}, blockDim={3,2,1}, args=0x7fffffff76f8, sharedMem=0, stream=0x7e0860) /path/to/RAJA/install/RAJA/include/RAJA/policy/hip/MemUtils_HIP.hpp:273" thrown in the test body.

In this example, RAJA uses a CAMP error checking and reporting macro when
launching a hip kernel. If the kernel launch fails it will generate a string
containing the error message, the function called, the function arguments, and
the location of the call.

RAJA has many examples of using Camp. In fact, so many internal RAJA implementations use Camp that RAJA has a
`Camp alias page <https://github.com/LLNL/RAJA/blob/0aef7cc44348d82e94e73e12f77c27ea306e47b8/include/RAJA/util/camp_aliases.hpp>`_ which
creates RAJA aliases for many Camp features.

Camp Used in RAJAPerf
=====================

`RAJAPerf <https://github.com/LLNL/RAJAPerf>`_ is RAJA's Performance Suite designed to explore the performance of loop-based computational kernels found in HPC applications.
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

This RAJAPerf example creates a typed Camp ``Resource`` and then returns either the default stream or a different stream depending on
the value of ``run_params.getGPUStream()``. This example shows member functions of the typed resource such as ``get_default()``
for getting the default stream and ``CudaFromStream()`` for selecting a specific stream.

See the full example `here <https://github.com/LLNL/RAJAPerf/blob/abb07792a899f7417e77ea40015e7e1dfd52716e/src/common/KernelBase.hpp>`_.

Camp Used in CHAI
=================

`CHAI <https://github.com/LLNL/CHAI>`_ is a library that handles automatic data migration to different memory spaces behind an array-style interface. It was designed to 
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

This CHAI example shows how to construct a generic ``Resource`` from the default stream of a typed ``Resource``. Later, the
example shows how to create a barrier with that ``Resource`` by calling the ``wait()`` method.

See the full example `here <https://github.com/LLNL/CHAI/blob/7ba2ba89071bf836071079929af7419da475ba27/src/chai/ArrayManager.cpp#L246>`_.

Many codes at LLNL and elsewhere use the different libraries within the RAJA Portability Suite. Camp plays a vital role
in the software abstractions that make the RAJA Portability Suite possible.
