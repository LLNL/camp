// Define CAMP_CONFIG_OVERRIDE to change this on a per-file basis
#if !defined(CAMP_CONFIG_OVERRIDE)
#cmakedefine CAMP_ENABLE_OPENMP
#cmakedefine CAMP_ENABLE_TARGET_OPENMP
#cmakedefine CAMP_ENABLE_CUDA
#cmakedefine CAMP_ENABLE_HIP
#cmakedefine CAMP_ENABLE_SYCL
#endif

#cmakedefine CAMP_VERSION_MAJOR
#cmakedefine CAMP_VERSION_MINOR
#cmakedefine CAMP_VERSION_PATCH

#define CAMP_VERSION (CAMP_VERSION_MAJOR * 10000) \
                    +(CAMP_VERSION_MINOR * 100) \
                    +(CAMP_VERSION_PATCH)
