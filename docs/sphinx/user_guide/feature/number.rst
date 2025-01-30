.. # Copyright (c) 2018-2025, Lawrence Livermore National Security, LLC and
.. # other Camp project contributors. See the camp/LICENSE file for details.
.. #
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _number-label:

=============
Numeric types 
=============

------------------
The ``num<>`` type
------------------

Camp includes constructs for compile time numerical `types` which can be used in almost any other Camp construct. 
The ``num<>`` type is an integral constant `type`, rather than a constant value. Let's look at the difference: 

.. code-block:: cpp 

  constexpr int myVal = 0; 
  // This is a numeric value
  using myNum = camp::num<0>;
  // This is a type, which holds onto the value 0 as part of its template type information,
  // so it can be used with other Camp features (such as list) which act on types rather than values

---------------
Index Sequences
---------------

Camp provides compile time methods to generate sequences of index types. The most basic is ``make_idx_seq``, which 
constructs a sequence of integers starting from zero, and ending at the integral value specified. 

.. code-block:: cpp 

  using seqType = make_idx_seq_t<4>;
  // seqType is idx_seq<0, 1, 2, 3>

Index sequences can also be created using a list (or other sized construct) that already contains data using the ``make_idx_seq`` template.

.. code-block:: cpp 

  using listType = camp::list<int, char> 
  using seqType = make_idx_seq_t<listType>;
  // seqType is idx_seq<0, 1>

