#ifndef camp_array_HPP__
#define camp_array_HPP__

#include "camp/defines.hpp"
#include "camp/helpers.hpp"
#include "camp/number.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace camp {
   ///
   /// Provides a portable std::array-like class.
   /// 
   /// Differences from std::array are listed below:
   /// - No __device__ "at" method (exceptions are not GPU friendly)
   /// - Calling front or back is a compile error when size() == 0
   ///   (instead of undefined behavior at run time)
   /// - No reverse iterators have been implemented yet
   /// - Swap is yet to be implemented
   ///
   template <class T, std::size_t N>
   struct array {
      using value_type = T;
      using size_type = std::size_t;
      using difference_type = std::ptrdiff_t;
      using reference = value_type&;
      using const_reference = const value_type&;
      using pointer = value_type*;
      using const_pointer = const value_type*;
      using iterator = pointer;
      using const_iterator = const_pointer;

      // TODO: Investigate device trap
      constexpr reference at(size_type pos) {
         if (pos >= N) {
            throw std::out_of_range{"camp::array::at detected out of range access"};
         }

         return elements[pos];
      }

      constexpr const_reference at(size_type pos) const {
         if (pos >= N) {
            throw std::out_of_range{"camp::array::at detected out of range access"};
         }

         return elements[pos];
      }

      CAMP_HOST_DEVICE constexpr reference operator[](size_type pos) {
         return elements[pos];
      }

      CAMP_HOST_DEVICE constexpr const_reference operator[](size_type pos) const {
         return elements[pos];
      }

      CAMP_HOST_DEVICE constexpr reference front() {
         static_assert(N > 0, "Calling camp::array::front on an empty array is not allowed.");
         return elements[0];
      }

      CAMP_HOST_DEVICE constexpr const_reference front() const {
         static_assert(N > 0, "Calling camp::array::front on an empty array is not allowed.");
         return elements[0];
      }

      CAMP_HOST_DEVICE constexpr reference back() {
         static_assert(N > 0, "Calling camp::array::back on an empty array is not allowed.");
         return elements[N - 1];
      }

      CAMP_HOST_DEVICE constexpr const_reference back() const {
         static_assert(N > 0, "Calling camp::array::back on an empty array is not allowed.");
         return elements[N - 1];
      }

      CAMP_HOST_DEVICE constexpr pointer data() noexcept {
         return elements;
      }

      CAMP_HOST_DEVICE constexpr const_pointer data() const noexcept {
         return elements;
      }

      CAMP_HOST_DEVICE constexpr iterator begin() noexcept {
         return elements;
      }

      CAMP_HOST_DEVICE constexpr const_iterator begin() const noexcept {
         return elements;
      }

      CAMP_HOST_DEVICE constexpr const_iterator cbegin() const noexcept {
         return elements;
      }

      CAMP_HOST_DEVICE constexpr iterator end() noexcept {
         return elements + N;
      }

      CAMP_HOST_DEVICE constexpr const_iterator end() const noexcept {
         return elements + N;
      }

      CAMP_HOST_DEVICE constexpr const_iterator cend() const noexcept {
         return elements + N;
      }

      CAMP_HOST_DEVICE constexpr bool empty() const noexcept {
         return N == 0;
      }

      CAMP_HOST_DEVICE constexpr size_type size() const noexcept {
         return N;
      }

      CAMP_HOST_DEVICE constexpr size_type max_size() const noexcept {
         return N;
      }

      CAMP_HOST_DEVICE constexpr void fill(const T& value) {
         for (std::size_t i = 0; i < N; ++i) {
            elements[i] = value;
         }
      }

      value_type elements[N];
   };

   template <class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr bool operator==(const array<T, N>& lhs,
                                                     const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (!(lhs[i] == rhs[i])) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr bool operator!=(const array<T, N>& lhs,
                                                     const array<T, N>& rhs) {
      return !(lhs == rhs);
   }

   template <class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr bool operator<(const array<T, N>& lhs,
                                                    const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] < rhs[i]) {
            return true;
         }
         else if (rhs[i] < lhs[i]) {
            return false;
         }
      }

      return false;
   }

   template <class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr bool operator<=(const array<T, N>& lhs,
                                                     const array<T, N>& rhs) {
      return !(rhs < lhs);
   }

   template <class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr bool operator>(const array<T, N>& lhs,
                                                    const array<T, N>& rhs) {
      return rhs < lhs;
   }

   template <class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr bool operator>=(const array<T, N>& lhs,
                                                     const array<T, N>& rhs) {
      return !(lhs < rhs);
   }

   template <std::size_t I, class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr T& get(array<T, N>& a) noexcept {
      static_assert(I < N, "Index out of bounds in camp::get<> (camp::array&)");
      return a[I];
   }

   template <std::size_t I, class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr T&& get(array<T, N>&& a) noexcept {
      static_assert(I < N, "Index out of bounds in camp::get<> (camp::array&&)");
      return move(a[I]);
   }

   template <std::size_t I, class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr const T& get(const array<T, N>& a) noexcept {
      static_assert(I < N, "Index out of bounds in camp::get<> (const camp::array&)");
      return a[I];
   }

   template <std::size_t I, class T, std::size_t N>
   CAMP_HOST_DEVICE inline constexpr const T&& get(const array<T, N>&& a) noexcept {
      static_assert(I < N, "Index out of bounds in camp::get<> (const camp::array&&)");
      return move(a[I]);
   }

#if defined(__cplusplus) && __cplusplus >= 201703L
   // The use of auto for non-type template parameters is not supported
   // until C++17
   namespace detail {
      template <class T, auto N, auto... I>
      CAMP_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N>
         to_array_impl(T (&a)[N], idx_seq<I...>)
      {
         return {{a[I]...}};
      }

      template <class T, auto N, auto... I>
      CAMP_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N>
         to_array_impl(T (&&a)[N], idx_seq<I...>)
      {
         return {{move(a[I])...}};
      }
   }
 
   template <class T, auto N>
   CAMP_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N>
      to_array(T (&a)[N])
   {
      return detail::to_array_impl(a, make_idx_seq_t<N>{});
   }

   template <class T, auto N>
   CAMP_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N>
      to_array(T (&&a)[N])
   {
      return detail::to_array_impl(move(a), make_idx_seq_t<N>{});
   }
#else
   template <class T, std::size_t N>
   CAMP_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N> to_array(T (&a)[N]) {
      array<std::remove_cv_t<T>, N> result;

      for (std::size_t i = 0; i < N; ++i) {
         result[i] = a[i];
      }

      return result;
   }

   template <class T, std::size_t N>
   CAMP_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N> to_array(T (&&a)[N]) {
      array<std::remove_cv_t<T>, N> result;

      for (std::size_t i = 0; i < N; ++i) {
         result[i] = move(a[i]);
      }

      return result;
   }
#endif

#if defined(__cplusplus) && __cplusplus >= 201703L
   ///
   /// Deduction guide
   ///
   /// TODO: Find a way to make sure all U's are the same:
   ///       https://github.com/llvm/llvm-project/blob/main/libcxx/include/array#L383
   template <class T, class... U>
   CAMP_HOST_DEVICE array(T, U...) -> array<T, 1 + sizeof...(U)>;
#endif
} // namespace camp

#if defined(__cplusplus) && __cplusplus >= 201703L
// For structured bindings
namespace std {
   template <class T, std::size_t N>
   struct tuple_size<camp::array<T, N>> :
      std::integral_constant<std::size_t, N>
   { };

   template <std::size_t I, class T, std::size_t N>
   struct tuple_element<I, camp::array<T, N>> {
      using type = T;
   };
}
#endif

#endif // !defined(CAMP_ARRAY_H)

