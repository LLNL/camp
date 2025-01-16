.. ##
.. ## Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
.. ## and Camp project contributors. See the camp/LICENSE file for details.
.. ##
.. ## Part of the LLVM Project, under the Apache License v2.0 with LLVM
.. ## exceptions.
.. ## See https://llvm.org/LICENSE.txt for license information.
.. ## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
.. ##
.. ## See the LLVM_LICENSE file at http://github.com/llnl/camp for the
.. ## full license text.       
.. ##

.. _features-label:

*************
Camp Features
*************

The following sections describe the main Camp features. These sections are designed
to familiarize users with Camp and it's capabilities, as well as provide syntactic
examples of how to use Camp. 

Camp has many features that mimic the C++ standard library. The benefit of using Camp is 
that containers and structures, and the operations performed on them, can be run on 
a device, instead of just on the host, as the C++ standard library is limited to. 

For more information about specific Camp features, visit the pages below. 

.. toctree::
  :maxdepth: 3

  feature/array
  feature/list
  feature/resource
  feature/number
  feature/tuple
