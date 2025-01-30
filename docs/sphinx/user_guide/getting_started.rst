.. # Copyright (c) 2018-2025, Lawrence Livermore National Security, LLC and
.. # other Camp project contributors. See the camp/LICENSE file for details.
.. #
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _getting_started-label:

*************************
Getting Started with Camp
*************************

This page provides information on how to quickly get up and running with Camp

------------
Installation
------------

Camp is hosted on GitHub `here <https://github.com/LLNL/Camp>`_.
To clone the repo into your local working space, type:

.. code-block:: bash

  $ git clone --recursive https://github.com/LLNL/Camp.git


The ``--recursive`` argument is required to ensure that the *BLT* submodule is
also checked out. `BLT <https://github.com/LLNL/BLT>`_ is the build system we
use for Camp.


^^^^^^^^^^^^^
Building Camp
^^^^^^^^^^^^^

Camp uses CMake and BLT to handle builds. Make sure that you have a modern
compiler loaded and the configuration is as simple as:

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake ../

By default, Camp will only support the ``Host`` backend. Additional backends for
device support can be enabled using the options such as ``-DENABLE_CUDA=On``.
CMake will provide output about which compiler
is being used and the values of other options. Once CMake has completed, Camp
can be built with Make:

.. code-block:: bash

  $ make

-----------
Basic Usage
-----------

The most common Camp usage revolves around Camp ``Resource`` types. A quick way to get started
using Camp resources is to do the following:

.. code-block:: cpp

   #include "camp/camp.hpp"

   using namespace camp::resources;

   #if defined(UMPIRE_ENABLE_CUDA)
     using resource_type = Cuda;
   #elif defined(UMPIRE_ENABLE_HIP)
     using resource_type = Hip;
   #else
     using resource_type = Host;
   #endif

   resource_type d1, d2;
   ...
   // Use the d1 and d2 resources to launch kernels, etc.
   ...
   d2.get_event().wait(); //Synchronize the resources
   ...

Find more examples of using Camp resources in the Using Camp section :ref:`using_camp-label`.
