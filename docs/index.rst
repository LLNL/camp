.. ##
.. ## Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the CAMP/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

################
CAMP
################

The CAMP (Concepts and MetaProgramming, or Compiler Agnostic MetaProgramming) Library is a C++ Library that provides standard-library-like structures and algorithms that can be portably used on a variety of backend architectures. CAMP is developed at Lawrence Livermore National Laboratory (LLNL). This project collects a variety of macros and metaprogramming facilities for C++ projects. It's in the direction of projects like metal (a major influence) but with a focus on wide compiler compatibility across HPC-oriented systems. 

Camp aims to be primarily header-only, with only one small implementation file. 

=================================
Git Repository and Issue Tracking
=================================

The main interaction hub for CAMP is on `GitHub <https://github.com/LLNL/camp>`_
There you will find the Git source code repository, issue tracker, release 
history, and other information about the project.

========================================
CAMP Copyright and License Information
========================================

Please see :ref:`camp-copyright`.

.. toctree::
   :hidden: 
   :caption: User Documentation

   sphinx/user_guide/index
   doxygen/html/index

.. toctree::
   :hidden: 
   :caption: Developer Documentation

   sphinx/camp_license
