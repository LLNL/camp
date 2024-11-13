.. _resources-label:

=========
Resources
=========

Camp Resources allow users to keep track of `streams of execution`. A single `stream of execution` on the device 
(e.g. a single CUDA stream) corresponds to a single Camp device resource. Similarly, when we are executing on the 
host, this corresponds to a separate `stream of execution` and therefore a separate Camp host resource.

Typically, we deal with multiple Camp resources. This includes a single resource for the host and one or more for 
the device, depending on how many (CUDA, Hip, etc.) streams we have in use. While we can have multiple Camp resources 
for the device (e.g. multiple CUDA streams), we can only have one resource for the host because the host only has one `stream of execution`.

In addition to the Host, Camp provides a resource for several device backends:

* CUDA
* HIP
* OpenMP Target
* SYCL

Generic vs. Specific Camp Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Camp has two different types of Resources: generic (aka. type-erased) and concrete (aka. typed). 
The generic (type-erased) resources work by holding a shared pointer to the base implementation of a resource 
with virtual methods that call the same methods on the concrete (typed) resource.

A specific resource is created with:

.. code-block:: cpp

   camp::resources::Host h1;
   camp::resources::Cuda c1;

This will create a Host resource and a CUDA resource. With either ``c1`` or ``h1`` we can call different methods 
like ``get_platform()`` or ``get_stream()``. See the Doxygen information for more details. On the other hand, a generic 
resource is created with:

.. code-block:: cpp

   camp::resources::Resource h{h1};
   camp::resources::Resource r{c1};

This way of creating a generic resource uses the specific resource created above, ``h1`` or ``c1``, to construct it.
We can also create a generic resource with:

.. code-block:: cpp

   camp::resources::Resource h{camp::resources::Host()};
   camp::resources::Resource r{camp::resources::Cuda()};

Having the ability to use both a typed and a typed-erased resource allows users more flexibility as they use Camp resources.
The compiler can implicitly convert between the generic and concrete resources for ease of use.

Using Resources
~~~~~~~~~~~~~~~

Below, a few key use cases for Camp resources are shown.

Using Events
^^^^^^^^^^^^

Camp resources allow users a hardware-agnostic way of interacting with the underlying hardware. For example, using
a Camp resource, users can create an event with the resource with:

.. code-block:: cpp

   camp::resources::Cuda c1;
   c1.get_event();

The ``get_event()`` method will create and record a CUDA event. From here, users can check the event:

.. code-block:: cpp

   if(c1.get_event().check()) {
     // If we get here, the event has completed
   }

Or explicitly wait on the event:

.. code-block:: cpp

   c1.get_event().wait(); //Explicitly wait for the event to complete
   // Do some work

Users can also use events to synchronize on the device:

.. code-block:: cpp

   #if defined(ENABLE_CUDA)
     using resource_type = camp::resources::Cuda; // Create the (Specific) Camp resource
   #elif defined(ENABLE_HIP)
     using resource_type = camp::resources::Hip; // Create the (Specific) Camp resource
   #endif

   ...
   auto resource = camp::resources::Resource{resource_type{}}; // Create a (Generic) Camp resource 
   my_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(my_data); // Do some work on the device
   resource.get_event().wait(); // Use the resource to synchronize the device after the kernel
   ...

Comparing Resources
^^^^^^^^^^^^^^^^^^^

It may be handy to be able to compare two different resources to see if they are the same or not.
One common use case is when dealing with two different device streams where each stream corresponds
to a separate Camp resource.

.. code-block:: cpp

   camp::resources::Cuda c1, c2; // Create two different Cuda resources
   ...  
   my_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, c1.get_stream()>>>(my_data);
   if(c1 != c2) { // Compare device resources
     c1.get_event().wait(); // Synchronize streams if resources are not the same
   }
   my_other_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, c2.get_stream()>>>(my_data);
   ...


Comparison of resources must be of the same type. In other words, you can compare two generic resources
for equality OR two specific (or typed) resources for equality. If you need to compare a generic resource
with a specific resource, you have to convert the specific (typed) resource to a generic one. For example:

.. code-block:: cpp

   camp::resources::Cuda c1; //This is a typed resource
   camp::resources::Resource other_res = get_other_resource(...); //This is my generic resource
   
   if (other_res == Resource{c1}) {
     std::cout << "They are equal!" << std::endl;
   }

The above code works because we converted the typed resource, ``c1``, to a generic resource with ``Resource{c1}``.

While it is possible for two device resources to be different since each resource refers to a different
device stream, all ``Host`` Camp resources will be the same since there is only one `stream of execution` 
for the Host.

Whether users are using a CUDA or HIP backend, the Camp resources require no code changes and provide
a hardware-agnostic interface. Because of the way Camp resources were built, the compiler can implicitly
convert between the generic and concrete resources for ease of use.

Find more examples of using Camp resources in the Using Camp section :ref:`using_camp-label`.
