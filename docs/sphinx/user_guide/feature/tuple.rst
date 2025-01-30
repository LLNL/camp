.. # Copyright (c) 2018-2025, Lawrence Livermore National Security, LLC and
.. # other Camp project contributors. See the camp/LICENSE file for details.
.. #
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _tuple-label:

======
Tuples
======

------------
Introduction
------------

A tuple is a fixed-size collection of heterogeneous values (i.e., it may contain different types of values). 
Camp includes a ``tuple`` structure that closely mimics the one present in the C++ Standard Template Library (STL), 
with the added benefit of working on GPUs. 

Camp has two tuple implementations, ``tuple`` and ``tagged_tuple``. ``tuple`` is the base 
class of ``tagged_tuple``. The biggest difference between the two is that ``tuple`` uses 
an ordered integer sequence for its indexes, whereas a ``tagged_tuple`` uses a set of index 
`tags` provided by the user. 

-----
Usage
-----

Constructing ``tuples``
^^^^^^^^^^^^^^^^^^^^^^^

Using a tuple in Camp is very similar to the C++ standard template library (STL) version.

.. code-block:: cpp 
  
  camp::tuple<int, double> myTuple(5, 3.142);
  
  std::tuple<int, double> stdTuple(6, 6.022);

As we can see from the above example, the construction method of each tuple is very similar. 
Camp also contains a ``tagged_tuple`` where one must provide an index sequence, as well as values. 
This index sequence may be a list of any type, including :ref:`number-label`. 

Constructing ``tagged_tuples``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp 

  camp::tagged_tuple<camp::list<float, char>, int, double> myTaggedTuple(7, 9.023);
  camp::tagged_tuple<camp::list<num<1>, num<3>>, int, double> myOtherTaggedTuple(8, 3.342);

Assignment
^^^^^^^^^^

Camp's tuple implementation allows for assignment of one tuple to another, provided the parameters are compatible (i.e. convertible). 

.. code-block:: cpp 

  const camp::tuple<int, char> t(5, 'a');
  camp::tagged_tuple<camp::list<int, char>, long long, char> t2;

  t2 = t;
  // t2 = [5, 'a']

``make_tuple`` and ``make_tagged_tuple``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Camp provides some convenience functions for making tuples and tagged tuples from items. Consider the following: 

.. code-block:: cpp 

  auto myTuple = camp::make_tuple(5, 'a');
  auto myTaggedTuple = camp::make_tagged_tuple<camp::list<int, char>>(5, 'a');

Accessing Elements using ``get<>()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sequence provided to the ``tagged_tuple`` constructor is what is used to access elements using the ``get<>`` method. 
In a normal ``tuple``, we can use the ``get<>`` method to access elements by index, and by type. 
The elements of a ``tagged_tuple`` can only be accessed via the values of the index sequence provided at the time the 
``tagged_tuple`` was created. 

Let's see an example of the ``get<>`` method now:

.. code-block:: cpp 

  // Let's start with the basic camp::tuple: 
  camp::tuple<int, double> myTuple(5, 3.142);
  
  // get by index
  auto var = camp::get<0>(myTuple);
  // var is 5

  // get by type
  auto var2 = camp::get<double>(myTuple);
  // var2 is 3.142 

  // Now let's move on to the tagged_tuple
  camp::tagged_tuple<camp::list<float, char>, int, double> myTaggedTuple(7, 9.023);

  // we can only use the type from the first list (either float or char)
  auto var3 = camp::get<float>(myTaggedTuple);
  // var3 is 7, and is type int (not float). The tag list is only for indexing purposes

Helper methods
^^^^^^^^^^^^^^

``tuple_size``
""""""""""""""

The ``tuple_size`` method returns the number of elements in the tuple. In the examples below, we assume the existence of the Camp tuple ``myTuple`` in the previous example.

.. code-block:: cpp 

  auto size = camp::tuple_size<myTuple>; 
  // size is num<2>

``tuple_element``
"""""""""""""""""

Camp provides ``tuple_element`` and ``tuple_element_t`` methods to obtain the type of the 
tuple element. 

.. code-block:: cpp 

  using type = camp::tuple_element_t<0, myTuple>;
  // type is int
