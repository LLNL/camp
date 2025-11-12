.. ##
.. ## Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
.. ## and Camp project contributors. See the camp/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

################
Camp
################

Camp (Compiler Agnostic MetaProgramming) is a portable C++ Library
that provides modern C++ structures and algorithms that can be used on a variety of 
backend architectures. Camp is part of the RAJA Portability Suite and is developed 
at Lawrence Livermore National Laboratory (LLNL). This project 
collects a variety of macros and metaprogramming facilities for C++ projects.  
Camp focuses on wide compiler compatibility across HPC systems. 

Camp aims to be primarily header-only, with only one small implementation file. 

=================================
Contributions
=================================

To contribute, go to Camp's `GitHub <https://github.com/LLNL/camp>`_.
There you can find the Git source code repository, issue tracker, release 
history, and other information about the project.

Make a `pull request <https://github.com/LLNL/camp/compare>`_ with ``main`` as the destination branch. 
We have several Docker tests that must all passed in order for your branch to be merged.

Please see :ref:`camp-dev-guide-label` for more details.

=======
Authors
=======

Thanks to all of Camp's `contributors <https://github.com/LLNL/camp/graphs/contributors>`_.

=========
Questions
=========

If you have a question, file an `issue <https://github.com/LLNL/camp/issues/new/choose>`_ or send an email to raja-dev@llnl.gov.

Please see :ref:`camp-copyright`.

.. toctree::
   :hidden: 
   :caption: User Documentation

   sphinx/user_guide/index
   doxygen/html/index

.. toctree::
   :hidden: 
   :caption: Developer Documentation

   sphinx/dev_guide/index
   sphinx/camp_license
