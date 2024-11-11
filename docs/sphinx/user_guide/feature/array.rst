
.. _array-label:

======
Arrays
======

Camp has its own Array implementation very similar to C++ std::array. Array functionality is
included in the ``array.hpp`` file within ``include/camp/``. 

Some important notes: 

.. note:: All Camp structures and classes are in the namespace ``camp``.
          
----------------
Array Philosophy
----------------

In C++, an array is a constant-sized, statically allocated slice of contiguous memory. 
While C++ arrays are conceptually equivalent to C-style arrays--in that their size must be known 
at compile time--they have many quality-of-life features that make them easier and more convenient to use. 

For example, in C, one might allocate memory in as follows:: 
  
  #define N 100
  double* arr = (double *)malloc(sizeof(double) * N);

This is the super clunky and old-school way to do it, but in the end, we get a slice of 
memory that can hold ``N`` double values. But alas, we must do all of our own memory management,
implement our own operators to iterate over the values, everything. 

C++ introduces a `new` way to do the same thing, at the language level, using the ``new`` keyword:

.. code-block:: c

    constexpr size_t N = 100;

    double* arr = new double[N];

This example is functionally equivalent to the C example in basically every way, the only difference 
being that our call to ``new`` obfuscates the call to ``malloc``. 

Finally, the C++ standard library (STL) implemented an ``array`` class, which provides access functions, 
iterators, size information, and automatic memory management!

.. code-block:: cpp

    constexpr size_t N = 100; // size must be known at compile time

    std::array<double, N> arr;

The only drawback from the STL array implementation is that it can only be used on the ``host``, not any ``device``. 
That's where Camp comes in. The ``camp::array`` retains all of the features from the STL array, but can also be used in GPU device code.

-----------------------
The camp::array methods
-----------------------

The Camp array syntax is exactly the same as the C++ standard library syntax, the only 
difference being that we use the ``camp`` namespace instead of ``std``.

.. code-block:: cpp

    constexpr size_t N = 100;

    camp::array<double, N> arr;

It's that simple!

.. important:: 
  While the Camp ``Array`` tries to emulate the STL array as closely as it can, there are some key differences 
  that users should be aware of: 
  
  * no ``__device__ "at"`` method. The ``at`` method has the potential to throw exceptions, which is not GPU friendly.
  * Calling ``front()`` or ``back()`` is a compile time error when ``size() == 0``, 
    as opposed to undefined  behaviour set by the C++ standard.
  * Camp arrays do not have reverse iterators implemented. 
  * Camp arrays do not have the ``swap`` method implemented.   

Constructors
^^^^^^^^^^^^

Camp arrays, like their STL counterparts, require the size and data type to be known at compile time. The array constructor is as follows:
  
.. code-block:: cpp

    camp::array<class T, std::size_t N>;

where T and N are provided by the user. In our previous example, we created an array of 100 ``doubles``:
  
.. code-block:: cpp

    constexpr size_t N = 100; 

    camp::array<double, N> arr;

We can also initialize our array at compile time, if we know the values:

.. code-block:: cpp
  
    camp::array<int, 3> arr = {1, 2, 3};

Element Access
^^^^^^^^^^^^^^

To access an element of an array, we can either use the ``operator[]``, or the ``at`` methods. The only real difference between these 
two methods is that the ``at`` method performs bounds checking on the input, and will throw an exception if the index requested is out of 
the bounds of the array.

.. code-block:: cpp

    camp::array<int, 3> arr  = {1, 2, 3};
    std::cout << arr[1] << std::endl; // prints 2
    std::cout << arr.at(2) << std::endl; // prints 3
    std::cout << arr.at(4) << std::endl; // throws std::out_of_range
    std::cout << arr[4] << std::endl; // undefined behaviour; accesses illegal memory

The ``front()`` and ``back()`` methods can be used to obtain a reference (const or non-const) to the first and last element in the array, respectively:

.. code-block:: cpp

    camp::array<int, 3> arr  = {1, 2, 3};
    std::cout << "front is " << arr.front() << ", back is " << arr.back() <<std::endl;
    // output: front is 1, back is 3

    arr.front() = 4;
    // arr is now {4, 2, 3}

A pointer to the underlying data can be obtained using the ``data()`` method. camp::array implements ``begin(), end(), cbegin(),`` and ``cend()`` iterator functions, which allow it to be used interchangeably in many C++ algorithms in the standard library, and beyond:

.. code-block:: cpp
  
    camp::array<int, 3> arr  = {1, 2, 3};
    for (const auto elem : arr) { 
      std::cout << elem << "\n";
    }

Camp provides multiple ``get`` methods that can be used for constexpr element access and moving:

.. code-block:: cpp

    camp::array<int, 3> arr  = {1, 2, 3};
    return camp::get<1>(arr); // returns 2

``get`` can return const and non-const lvalue references, and rvalue references. 

Size methods
^^^^^^^^^^^^

The Camp array contains a ``size()`` method which can be used to find the number of elements contained in the array. 
This is the same number that is passed into the array when it is constructed. The ``max_size()`` method does the same thing, as the 
size of an array is necessarily the maximum size, since an array has constant size. The ``empty()`` method returns a bool indicating 
whether the array has any elements. Since the array's  size is determined at compile time, the value of ``empty()`` will be true, unless 
an array is constructed with ``size  == 0``, which would not be very useful. 

The Camp array contains a ``fill`` method, which can be used to set all of the values of the array to one value:

.. code-block:: cpp

    camp::array<int, 3> arr  = {1, 2, 3};
    // array is {1, 2, 3}
    arr.fill(0);
    // array is {0, 0, 0}

Array Comparisons
^^^^^^^^^^^^^^^^^

The Camp array defines all of the standard comparison operators: ``==, !=, <, <=, >, >=``. Note that ``<, <=, >, >=`` use a 
lexographical check to determine which one is greater or less than. 

