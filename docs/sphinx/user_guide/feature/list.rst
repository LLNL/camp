.. # Copyright (c) 2018-2025, Lawrence Livermore National Security, LLC and
.. # other Camp project contributors. See the camp/LICENSE file for details.
.. #
.. # SPDX-License-Identifier: (BSD-3-Clause)

.. _list-label:

=====
Lists
=====

The ``list`` type in the Camp library is a powerful tool for type manipulation and 
metaprogramming in C++. It allows you to create and operate on sequences of types 
at compile time, enabling various operations such as accessing, filtering, and 
transforming types.

Some important notes: 

.. note:: All Camp structures and classes are within the namespace ``camp``.

------------
Introduction
------------

In C++, templates provide a way to define functions and classes that operate on types. 
The ``list`` type serves as a container for types, similar to how a standard container 
holds values. This allows for advanced type-level programming, where you can perform 
operations on types as if they were values.

The ``list`` type is defined as a variadic template, allowing you to store any number 
of types. You can use it to create type sequences, manipulate them, and apply various 
algorithms to them.

Common use cases for ``list`` include:

* Storing a collection of types for type traits or type transformations.
* Implementing compile-time algorithms that operate on types.
* Creating type-safe interfaces or APIs that leverage type information.

----------------------
Usage and Example Code
----------------------

The ``list`` structure is literally just a variadic template struct: 

.. code-block:: cpp

    template <typename... Ts>
    struct list {
      using type = list;
    };

Thus, the ``list`` type is incredibly versatile as it can handle any number of types. Since ``list`` is just a structure with a single 
member ``type``, there are no methods that are directly part of ``list``; however, Camp provides many helper methods that act on lists. 
Here are some examples demonstrating how to use the ``list`` type and its associated functions. 

The base ``list`` structure, the list-specific ``at`` construct, and the ``find_if`` construct are 
all included in ``camp/include/camp/list.hpp`` as well as all of the other helper methods described below.  

Creating a List
^^^^^^^^^^^^^^^

You can create a list of types using the ``list`` template:

.. code-block:: cpp

    using MyList = camp::list<int, double, char>;

Alternatively, you can turn a different type into a list using the ``as_list`` construct. In this example, we will use a :ref:`tuple-label`: 

.. code-block:: cpp 

    camp::tuple<int, double> myTuple(5, 2.22);
    using MyList = camp::as_list<decltype(myTuple)>
    // MyList is list<int, double>;


Accessing Types with ``at``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To access a specific type in the list, use the ``at`` template:

.. code-block:: cpp

    using MyList = camp::list<int, double, char>;

    using FirstType = camp::at<MyList, camp::num<0>>::type; 
    // FirstType is int
    using SecondType = camp::at<MyList, camp::num<1>>::type; 
    // SecondType is double
    using ThirdType = camp::at_v<MyList, 2>; 
    // ThirdType is char

The ``at`` functionality also contains two convenience templates for accessing the first and second elements of a list (in a tuple-like manner):

.. code-block:: cpp 

    using myList = camp::list<int, double, char>;

    using firstType = camp::first<myList>; 
    // firstType is int
    using secondType = camp::second<myList>; 
    // secondType is double 

In special circumstances, the ``at_v`` method can be used to store and retrieve values from a list, as well as types. This requires a little 
bit of prep work, and a special structure to actually hold the value within a type. 

.. code-block:: cpp 

    template<int VALUE>
    struct Value {
        static constexpr int value = VALUE;
    };

    using myList = camp::list< Value<8>, Value<4>, Value<2> >;

    auto val = camp::at_v<myList, 1>::value;
    // val is 4. 

Since ``Value`` is a templated struct, the actual value that we want to retrieve is encoded into the type information for the struct. 
So every ``Value`` we create is a unique type. The static ``value`` method of the ``Value`` struct allows us to retrieve the value 
information with just the type being needed. So when we call ``camp::at_v<myList, 1>``, the type that is returned is our unique ``Value`` struct, 
which has been templated with the value that we wish to store, and thus we can access the static ``value`` member of that struct to recieve the stored 
value in such a way that we can use it in our code.  


Finding a Type with ``find_if``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find the first type in the list that satisfies a condition using ``find_if``:

.. code-block:: cpp

    using myList = list<float, double, int*>;
    using FoundType = camp::find_if<is_double, MyList>::type; 
    // FoundType is double

If the condition in ``find_if`` cannot be met, it will return ``nil``. 

Combining Lists with ``extend`` , ``prepend``, and ``append``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can combine two lists into one using either the ``extend`` method, ``prepend`` method, or ``append`` method. Like in python, ``extend`` 
will add the elements of one list into the back of the other list. ``prepend`` and ``append`` will add the specified list type to either 
the front or back of an existing list, respectively. Let's look at some examples:

.. code-block:: cpp 

    using list1 = camp::list<float, double, double>;
    using list2 = camp::list<int, int, char>;

    // extend
    using list3 = camp::extend<list1, list2>::type; 
    // list3 is type camp::list<float, double, double, int, int, char>

    // append
    using list4 = camp::append<list1, list2>::type; 
    // list4 is type camp::list<float, double, double, list<int, int, char>>

    // prepend
    using list5 = camp::prepend<list1, list2>::type; 
    // list5 is type camp::list<list<int, int, char>, float, double, double>

``Extend`` requires two lists to be given as inputs, whereas ``prepend`` and ``append`` can take any type:

.. code-block:: cpp 

    using list1 = camp::list<int, int, char>;

    using list2 = camp::append<list1, double>::type; 
    // list2 is type camp::list<int, int, char, double>

Flattening Nested Lists
^^^^^^^^^^^^^^^^^^^^^^^

Nested lists can be flattened into a single dimension using the ``flatten`` construct. 

.. code-block:: cpp 

    using list1 = camp::list<int, list<char, double>, list<list<list<float>>>>;

    using list2 = camp::flatten<list1>::type;
    // list2 is of type list<int, char, double, float>;

Performing transformations on elements of a list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Camp provides a ``transform`` construct to perform operations on the types contained in a list:

.. code-block:: cpp 

    using list1 = camp::list<int&, int&>;

    using list2 = camp::transform<std::remove_reference, list1>;
    // list2 is of type camp::list<int, int>;

Operating on lists with the ``accumulate`` construct
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``accumulate`` construct can be used to apply a given operation to a list. ``accumulate`` 
takes an operation, an initial value, and a list. It applies the operation across the list, starting with 
the initial value.

.. code-block:: cpp 

    using myNewList = accumulate<append, list<>, list<int, float, double>>; 
    // myNewList is of type list<int, float, double>;

Cartesian products of lists (an application of accumulate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Camp provides a method to evaluate the cartesian product of two lists. The ``cartesian_product`` method is simply an 
application of the accumulate method. 

.. code-block:: cpp 

    struct a;
    struct b;
    struct c;
    struct d;
    struct e;

    using listA = list<a, b>;
    using listB = list<c, d, e>

    using prod = cartesian_product<listA, listB>;
    // prod is of type list<list<a, c>,
    //                      list<a, d>,
    //                      list<a, e>,
    //                      list<b, c>,
    //                      list<b, d>,
    //                      list<b, e>>

Finding the first index of a type within a list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``index_of`` method can be used on a list to find the first index in the list where
a given type appears. If the type is not found in a list, ``nil`` is returned. 

.. code-block:: cpp 

    using myList = list<int, double, char, float>

    using firstChar = index_of<char, myList>::type
    // fistChar is num<2>
    using firstBool = index_of<bool, myList>::type
    // firstBool is nil


Filtering Types with ``filter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Camp provides a way to filter a list such that only the desired types remain. 

.. code-block:: cpp

    using myList = list<int, float*, double, short*>;

    using ptrsOnly = filter<std::is_pointer, myList>;
    // ptrsOnly is of type list<float*, short*>

------------------------
Using Lists to make Maps
------------------------

Camp provides a ``map.hpp`` header which can be combined with associative lists 
to create a map-type structure. Using the ``at_key`` method, we do a lookup on the maps "keys" to access its "values". This works 
because the lists  act as key value pairs. 

.. code-block:: cpp  

    using myMap = list<list<int, num<0>>, list<char, num<1>>>;

    using val = at_key<myList, int>;

    // val is num<0>

