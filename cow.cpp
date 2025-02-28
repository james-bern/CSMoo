// <!> Begin playground.cpp <!> 
#include <cstring>
#include <stdarg.h>

// <!> Begin basics.cpp <!> 
// detect operating system
#if defined(__APPLE__) || defined(__MACH__)
#define OPERATING_SYSTEM_APPLE
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
#define OPERATING_SYSTEM_WINDOWS
#else
#pragma message("ERROR: INVALID OPERATING SYSTEM")
#endif


#define _CRT_SECURE_NO_WARNINGS
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <time.h>
#include <chrono>
// #include<sys/time.h>


// // types
// basic types
typedef uint32_t uint;
typedef float real;
typedef uint8_t u8;
typedef uint64_t u64;
// vecD
template <uint D> struct Vector;
typedef Vector<2> vec2;
typedef Vector<3> vec3;
typedef Vector<4> vec4;
#define vecD Vector<D>
#define tuD  template <uint D>
#define tuDv tuD vecD
// matD
template <uint D> struct Matrix;
typedef Matrix<2> mat2;
typedef Matrix<3> mat3;
typedef Matrix<4> mat4;
#define matD Matrix<D>
#define tuDm tuD matD

// // standard macros
// ASSERT
#define ASSERT(b) do { if (!(b)) { \
    printf("ASSERT("); \
    printf(STR(b)); \
    printf("); <- "); \
    printf("Line %d in %s\n", __LINE__, __FILE__); \
    *((volatile int *) 0) = 0; \
} } while (0)
// STATIC_ASSERT
#define STATIC_ASSERT(cond) static_assert(cond, "STATIC_ASSERT");
// FORNOW_UNUSED
// - (suppresses compiler warning about unused variable)
#define FORNOW_UNUSED(expr) do { (void)(expr); } while (0)
#define _IS_INDEXABLE(arg) (sizeof(arg[0]))
#define _IS_ARRAY(arg) (_IS_INDEXABLE(arg) && (((void *) &arg) == ((void *) arg)))
#define ARRAY_LENGTH(arr) (_IS_ARRAY(arr) ? (sizeof(arr) / sizeof(arr[0])) : 0)

// // unorthodox macros
// for_
// - makes general-purpose for loops more readable (to me)
#define for_(i, N) for (uint i = 0; i < N; ++i)
#define _for_sign_ for (int sign = -1; sign <= 1; sign += 2)
// do_once
// - code inside do_once { ... } will run the first time it's hit, then never again
#define STR(foo) #foo
#define XSTR(foo) STR(foo)
#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)
#define do_once static bool CONCAT(_do_once_, __LINE__) = false; bool CONCAT(_prev_do_once_, __LINE__) = CONCAT(_do_once_, __LINE__); CONCAT(_do_once_, __LINE__) = true; if (!CONCAT(_prev_do_once_, __LINE__) && CONCAT(_do_once_, __LINE__))
// defer
// - code inside defer { ... }; will run when we leave the defer's enclosing scope
// - https://handmade.network/forums/t/1273-post_your_c_c++_macro_tricks/3
#if defined(OPERATING_SYSTEM_WINDOWS)
#include <utility>
#endif
template <typename F> struct Defer {
    F f;
    Defer(F _f) : f(_f) { }
    ~Defer() { f(); }
};
template <typename F> Defer<F> makeDefer(F f) { return Defer<F>(f); };
struct defer_dummy {};
template <typename F> Defer<F> operator + (defer_dummy, F &&f) { return makeDefer<F>(std::forward<F>(f)); }
#define defer auto CONCAT(defer_, __COUNTER__) = defer_dummy() + [&]()
// run_before_main
// - code inside run_before_main { ... }; runs before main
struct run_before_main_dummy {};
template <typename F> bool operator + (run_before_main_dummy, F &&f) { f(); return true; }
#define run_before_main static bool CONCAT(run_before_main_, __COUNTER__) = run_before_main_dummy() + []()

// // math
// constants
#define TINY_VAL real(1e-5)
#undef HUGE_VAL
#define HUGE_VAL real(1e6)
#define PI (real(3.14159265359))
#define TAU (2 * PI)
// conversions
real RAD(real degrees) { return (PI / 180 * (degrees)); }
real DEG(real radians) { return (180 / PI * (radians)); }
real INCHES(real mm) { return ((mm) / real(25.4)); }
real MM(real inches) { return ((inches) * real(25.4)); }
// trig
#define SIN sinf
#define COS cosf
#define TAN tanf
#define ATAN2 atan2f
// POW, SQRT
#define POW powf
#define SQRT sqrtf
int ROUND(float a) { return (int)(roundf(a)); }
tuDv ROUND(vecD a) { for_(d, D) a[d] = ROUND(a[d]); return a; }

// SGN
int SGN(  int a) { return (a < 0) ? -1 : 1; }
int SGN(float a) { return (a < 0) ? -1 : 1; }
// ABS
int  ABS( int a) { return (a < 0) ? -a : a; }
real ABS(real a) { return (a < 0) ? -a : a; }
tuDv ABS(vecD a) { return cwiseAbs(a); }
// MIN
// int  MIN( int a,  int b) { return (a < b) ? a : b; } // TODO: do we ever use this?
uint MIN(uint a, uint b) { return (a < b) ? a : b; }
real MIN(real a, real b) { return (a < b) ? a : b; }
// MAX
// int  MAX( int a,  int b) { return (a > b) ? a : b; } // TODO: do we ever use this?
uint MAX(uint a, uint b) { return (a > b) ? a : b; }
real MAX(real a, real b) { return (a > b) ? a : b; }
// floating-poiut comparisons
bool IS_ZERO(real a) { return (ABS(a) < TINY_VAL); }
tuD bool IS_ZERO(vecD a) { for_(d, D) if (!IS_ZERO(a[d])) return false; return true; }
bool ARE_EQUAL(real a, real b) { return IS_ZERO(ABS(a - b)); }
tuD bool ARE_EQUAL(vecD a, vecD b) { return IS_ZERO((a - b)); }

bool IS_BETWEEN_LOOSE(real p, real a, real b) { return (((a - TINY_VAL) < p) && (p < (b + TINY_VAL))); }
bool IS_BETWEEN_TIGHT(real p, real a, real b) { return (((a + TINY_VAL) < p) && (p < (b - TINY_VAL))); }

tuD bool ARE_PARALLEL(vecD a, vecD b) {
    return IS_ZERO(cross(normalized(a), normalized(b)));
}

// CLAMP
uint CLAMP(uint t, uint a, uint b) { return MIN(MAX(t, a), b); }
real CLAMP(real t, real a, real b) { return MIN(MAX(t, a), b); }
real MAG_CLAMP(real t, real a) {
    ASSERT(a > 0.0f);
    return CLAMP(t, -ABS(a), ABS(a));
}
// LERP
real LERP(real t, real a, real b) { return ((1.0f - t) * a) + (t * b); }
tuDv LERP(real t, vecD a, vecD b) { return ((1.0f - t) * a) + (t * b); }
real AVG(real a, real b) { return LERP(0.5f, a, b); }
tuD real AVG(vecD a) { real tmp = 0.0f; for_(d, D) tmp += a[d]; return tmp/ D; }
tuDv AVG(vecD a, vecD b) { return LERP(0.5f, a, b); }
real INVERSE_LERP(real p, real a, real b) { return (p - a) / (b - a); }
real LINEAR_REMAP(real p, real a, real b, real c, real d) { return LERP(INVERSE_LERP(p, a, b), c, d); }
// CLAMPED_LERP
real CLAMPED_LERP(real t, real a, real b) { return LERP(CLAMP(t, 0.0f, 1.0f), a, b); }
tuDv CLAMPED_LERP(real t, vecD a, vecD b) { return LERP(CLAMP(t, 0.0f, 1.0f), a, b); }
real CLAMPED_INVERSE_LERP(real p, real a, real b) { return CLAMP(INVERSE_LERP(p, a, b), 0.0f, 1.0f); }
real CLAMPED_LINEAR_REMAP(real p, real a, real b, real c, real d) { return LERP(CLAMPED_INVERSE_LERP(p, a, b), c, d); }
tuDv CLAMPED_LINEAR_REMAP(real p, real a, real b, vecD c, vecD d) { return LERP(CLAMPED_INVERSE_LERP(p, a, b), c, d); }
// MODULO
// - works for negative N
int MODULO(int x, int N) { return ((x % N) + N) % N; }

// // OS-specific
// TODO: MILLIS
// DEBUGGER
#ifdef OPERATING_SYSTEM_APPLE
#include <signal.h>
#define DEBUGBREAK() raise(SIGTRAP)
#elif defined(OPERATING_SYSTEM_WINDOWS)
#define DEBUGBREAK() __debugbreak()
#endif
// SLEEP
#ifdef OPERATING_SYSTEM_APPLE
#include <unistd.h>
#define SLEEP(x) usleep((x)*1000)
#elif defined(OPERATING_SYSTEM_WINDOWS)
#define SLEEP Sleep
#endif
// IS_NAN
#ifdef OPERATING_SYSTEM_APPLE
#include <unistd.h>
#define IS_NAN isnan
#elif defined(OPERATING_SYSTEM_WINDOWS)
// #include <windows.h>
#define IS_NAN isnan
#endif
// SWAP
template <typename T> void SWAP(T *a, T *b) {
    T tmp = *a;
    *a = *b;
    *b = tmp;
}
// MILLIS
long MILLIS() {
    // using namespace std::chrono;
    // milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    // return (long) ms.count();
    // struct timeval tv;
    // gettimeofday(&tv,NULL);
    // return (((long long) tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
    return 0L; // FORNOW
}
// don't buffer printf
run_before_main { setvbuf(stdout, NULL, _IONBF, 0); };
// seed random number generator
run_before_main { srand((unsigned int) time(NULL)); };
// GUARDED_free
void GUARDED_free(void *pointer) {
    if (pointer) free(pointer);
}

// <!> End basics.cpp <!>

// <!> Begin linalg.cpp <!> 
////////////////////////////////////////////////////////////////////////////////
// vectors and matrices ////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

tuD struct Vector {
    // real data[D];
    // real &operator [](uint index) { return data[index]; }
};

template <> struct Vector<2> {
    struct { real x, y; };
    real &operator [](uint index) {
        ASSERT(index < 2);
        if (index == 0) return x;
        return y;
    }
};

template <> struct Vector<3> {
    struct { real x, y, z; };
    real &operator [](uint index) {
        ASSERT(index < 3);
        if (index == 0) return x;
        if (index == 1) return y;
        return z;
    }
};

template <> struct Vector<4> {
    struct { real x, y, z, w; };
    real &operator [](uint index) {
        ASSERT(index < 4);
        if (index == 0) return x;
        if (index == 1) return y;
        if (index == 2) return z;
        return w;
    }
};


tuD struct Matrix {
    real data[D * D];
    real &operator ()(uint row, uint col) {
        ASSERT(row < D);
        ASSERT(col < D);
        return data[D * row + col];
    }
    const real &operator ()(uint row, uint col) const {
        ASSERT(row < D);
        ASSERT(col < D);
        return data[D * row + col];
    }
};



tuD union SnailTupleOfUnsignedInts {
    uint data[D];
    uint &operator [](uint index) { return data[index]; }
};

template <> union SnailTupleOfUnsignedInts<2> {
    uint data[2];
    struct { uint i, j; };
    uint &operator [](uint index) { return data[index]; }
};

template <> union SnailTupleOfUnsignedInts<3> {
    uint data[3];
    struct { uint i, j, k; };
    uint &operator [](uint index) { return data[index]; }
};

typedef SnailTupleOfUnsignedInts<2> uint2;
typedef SnailTupleOfUnsignedInts<3> uint3;
typedef SnailTupleOfUnsignedInts<4> uint4;

tuD bool operator == (SnailTupleOfUnsignedInts<D> A, SnailTupleOfUnsignedInts<D> B) {
    for_(d, D) {
        if (A[d] != B[d]) return false;
    }
    return true;
}

tuD bool operator != (SnailTupleOfUnsignedInts<D> A, SnailTupleOfUnsignedInts<D> B) {
    return (!(A == B));
}

// "constructors" //////////////////////////////////////////////////////////////

vec2 V2(real x, real y) { return { x, y }; }
vec3 V3(real x, real y, real z) { return { x, y, z }; }
vec4 V4(real x, real y, real z, real w) { return { x, y, z, w }; }
vec3 V3(vec2 xy, real z) { return { xy.x, xy.y, z }; }
vec4 V4(vec3 xyz, real w) { return { xyz.x, xyz.y, xyz.z, w }; }
vec2 V2(real x) { return { x, x }; }
vec3 V3(real x) { return { x, x, x }; }
vec4 V4(real x) { return { x, x, x, x }; }
vec2 _V2(vec3 xyz) { return { xyz.x, xyz.y }; }
vec3 _V3(vec4 xyzw) { return { xyzw.x, xyzw.y, xyzw.z }; }

mat2 M2(real a0, real a1, real a2, real a3) {
    return { a0, a1, a2, a3 };
}
mat3 M3(real a0, real a1, real a2, real a3, real a4, real a5, real a6, real a7, real a8) {
    return { a0, a1, a2, a3, a4, a5, a6, a7, a8 };
}
mat4 M4(real a0, real a1, real a2, real a3, real a4, real a5, real a6, real a7, real a8, real a9, real a10, real a11, real a12, real a13, real a14, real a15) {
    return { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 };
}
mat3 _M3(mat4 M) {
    return { M(0, 0), M(0, 1), M(0, 2), M(1, 0), M(1, 1), M(1, 2), M(2, 0), M(2, 1), M(2, 2) };
}

mat2 hstack(vec2 col0, vec2 col1) { return { col0.x, col1.x, col0.y, col1.y }; }
mat3 hstack(vec3 col0, vec3 col1, vec3 col2) { return { col0.x, col1.x, col2.x, col0.y, col1.y, col2.y, col0.z, col1.z, col2.z }; }
mat4 hstack(vec4 col0, vec4 col1, vec4 col2, vec4 col3) { return { col0.x, col1.x, col2.x, col3.x, col0.y, col1.y, col2.y, col3.y, col0.z, col1.z, col2.z, col3.z, col0.w, col1.w, col2.w, col3.w }; }

// arithmetic operators ////////////////////////////////////////////////////////

// vectors
tuDv  operator +  (vecD A, vecD B) {
    vecD result;
    for_(i, D) {
        result[i] = A[i] + B[i];
    }
    return result;
}
tuDv &operator += (vecD &A, vecD B) {
    A = A + B;
    return A;
}

tuDv  operator -  (vecD A, vecD B) {
    vecD result;
    for_(i, D) {
        result[i] = A[i] - B[i];
    }
    return result;
}
tuDv &operator -= (vecD &A, vecD B) {
    A = A - B;
    return A;
}

tuDv  operator *  (real scalar, vecD A) {
    vecD result;
    for_(i, D) {
        result[i]  = scalar * A[i];
    }
    return result;
}
tuDv  operator *  (vecD A, real scalar) {
    vecD result = scalar * A;
    return result;
}
tuDv &operator *= (vecD &A, real scalar) {
    A = scalar * A;
    return A;
}
tuDv  operator -  (vecD A) {
    return -1 * A;
}

tuDv  operator /  (vecD A, real scalar) {
    vecD result;
    for_(i, D) {
        result[i]  = A[i] / scalar;
    }
    return result;
}
tuDv  operator /  (real scalar, vecD A) {
    vecD result;
    for_(i, D) {
        result[i]  = scalar / A[i];
    }
    return result;
}
tuDv &operator /= (vecD &v, real scalar) {
    v = v / scalar;
    return v;
}

// matrices
tuDm  operator +  (matD A, matD B) {
    matD result = {};
    for_(k, D * D) {
        result.data[k] = A.data[k] + B.data[k];
    }
    return result;
}
tuDm &operator += (matD &A, matD B) {
    A = A + B;
    return A;
}

tuDm  operator -  (matD A, matD B) {
    matD result = {};
    for_(i, D * D) {
        result.data[i] = A.data[i] - B.data[i];
    }
    return result;
}
tuDm &operator -= (matD &A, matD B) {
    A = A + B;
    return A;
}

tuDm  operator *  (matD A, matD B) {
    matD result = {};
    for_(row, D) {
        for_(col, D) {
            for_(i, D) {
                result(row, col) += A(row, i) * B(i, col);
            }
        }
    }
    return result;
}
tuDm &operator *= (matD &A, matD B) {
    A = A * B;
    return A;
}
tuDv  operator *  (matD A, vecD b) { // A b
    vecD result = {};
    for_(row, D) {
        for_(col, D) {
            result[row] += A(row, col) * b[col];
        }
    }
    return result;
}
tuDv  operator *  (vecD b, matD A) { // b^D A
    vecD result = {};
    for_(row, D) {
        for_(col, D) {
            result[row] += A(col, row) * b[col];
        }
    }
    return result;
}
tuDm  operator *  (real scalar, matD M) {
    matD result = {};
    for_(k, D * D) {
        result.data[k] = scalar * M.data[k];
    }
    return result;
}
tuDm  operator *  (matD M, real scalar) {
    return scalar * M;
}
tuDm &operator *= (matD &M, real scalar) {
    M = scalar * M;
    return M;
}
tuDm  operator -  (matD M) {
    return -1 * M;
}

tuDm  operator /  (matD M, real scalar) {
    return (1 / scalar) * M;
}
tuDm &operator /= (matD &M, real scalar) {
    M = M / scalar;
    return M;
}

// important vector functions //////////////////////////////////////////////////

tuD real dot(vecD A, vecD B) {
    real result = 0.0f;
    for_(i, D) {
        result += A[i] * B[i];
    }
    return result;
}
tuDm outer(vecD A, vecD B) {
    matD result = {};
    for_(row, D) {
        for_(col, D) {
            result(row, col) = A[row] * B[col];
        }
    }
    return result;
}

real cross(vec2 A, vec2 B) {
    return A.x * B.y - A.y * B.x;
}
vec3 cross(vec3 A, vec3 B) {
    return { A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x };
}

tuD real squaredNorm(vecD A) {
    return dot(A, A);
}
tuD real norm(vecD A) {
    return sqrt(squaredNorm(A));
}
tuD real sum(vecD A) {
    real result = 0.0;
    for_(i, D) result += A[i];
    return result;
}
tuDv normalized(vecD A) {
    real norm_A = norm(A);
    // ASSERT(fabs(norm_v) > 1e-7);
    return (1 / norm_A) * A;
}
tuD real squaredDistance(vecD A, vecD B) {
    return squaredNorm(A - B);
}
tuD real distance(vecD A, vecD B) {
    return norm(A - B);
}

real ATAN2(vec2); // FORNOW: forward declaration
real angle_from_0_TAU(vec2 A, vec2 B) {
    real result = ATAN2(B - A);
    if (result < 0.0f) result += TAU;
    return result;
}

real angle_between(vec2 A, vec2 B) {
    real angle = acos(dot(A, B) / (norm(A) * norm(B)));
    if (cross(A, B) < 0) {
        angle = TAU - angle;
    }
    return angle;
}

// ALIASES
// tuD real length(vecD v) { return norm(v); }
// tuD real squared_length(vecD v) { return squaredNorm(v); }

// important matrix functions //////////////////////////////////////////////////

tuDm transpose(matD M) {
    matD result = {};
    for_(row, D) {
        for_(col, D) {
            result(row, col) = M(col, row);
        }
    }
    return result;
}

real determinant(mat2 M) {
    return M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
}
real determinant(mat3 M) {
    return M(0, 0) * (M(1, 1) * M(2, 2) - M(2, 1) * M(1, 2))
        - M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0))
        + M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
}
real determinant(mat4 M) {
    real A2323 = M(2, 2) * M(3, 3) - M(2, 3) * M(3, 2);
    real A1323 = M(2, 1) * M(3, 3) - M(2, 3) * M(3, 1);
    real A1223 = M(2, 1) * M(3, 2) - M(2, 2) * M(3, 1);
    real A0323 = M(2, 0) * M(3, 3) - M(2, 3) * M(3, 0);
    real A0223 = M(2, 0) * M(3, 2) - M(2, 2) * M(3, 0);
    real A0123 = M(2, 0) * M(3, 1) - M(2, 1) * M(3, 0);
    return M(0, 0) * ( M(1, 1) * A2323 - M(1, 2) * A1323 + M(1, 3) * A1223 ) 
        - M(0, 1) * ( M(1, 0) * A2323 - M(1, 2) * A0323 + M(1, 3) * A0223 ) 
        + M(0, 2) * ( M(1, 0) * A1323 - M(1, 1) * A0323 + M(1, 3) * A0123 ) 
        - M(0, 3) * ( M(1, 0) * A1223 - M(1, 1) * A0223 + M(1, 2) * A0123 ) ;
}

mat2 inverse(mat2 M) {
    real invdet = 1 / determinant(M);
    return { invdet * M(1, 1), 
        invdet * -M(0, 1), 
        invdet * -M(1, 0), 
        invdet * M(0, 0) };
}
mat3 inverse(mat3 M) {
    real invdet = 1 / determinant(M);
    return { invdet * (M(1, 1) * M(2, 2) - M(2, 1) * M(1, 2)),
        invdet * (M(0, 2) * M(2, 1) - M(0, 1) * M(2, 2)),
        invdet * (M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)),
        invdet * (M(1, 2) * M(2, 0) - M(1, 0) * M(2, 2)),
        invdet * (M(0, 0) * M(2, 2) - M(0, 2) * M(2, 0)),
        invdet * (M(1, 0) * M(0, 2) - M(0, 0) * M(1, 2)),
        invdet * (M(1, 0) * M(2, 1) - M(2, 0) * M(1, 1)),
        invdet * (M(2, 0) * M(0, 1) - M(0, 0) * M(2, 1)),
        invdet * (M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1)) };
}
mat4 inverse(mat4 M) {
    real invdet = 1 / determinant(M);
    real A2323 = M(2, 2) * M(3, 3) - M(2, 3) * M(3, 2) ;
    real A1323 = M(2, 1) * M(3, 3) - M(2, 3) * M(3, 1) ;
    real A1223 = M(2, 1) * M(3, 2) - M(2, 2) * M(3, 1) ;
    real A0323 = M(2, 0) * M(3, 3) - M(2, 3) * M(3, 0) ;
    real A0223 = M(2, 0) * M(3, 2) - M(2, 2) * M(3, 0) ;
    real A0123 = M(2, 0) * M(3, 1) - M(2, 1) * M(3, 0) ;
    real A2313 = M(1, 2) * M(3, 3) - M(1, 3) * M(3, 2) ;
    real A1313 = M(1, 1) * M(3, 3) - M(1, 3) * M(3, 1) ;
    real A1213 = M(1, 1) * M(3, 2) - M(1, 2) * M(3, 1) ;
    real A2312 = M(1, 2) * M(2, 3) - M(1, 3) * M(2, 2) ;
    real A1312 = M(1, 1) * M(2, 3) - M(1, 3) * M(2, 1) ;
    real A1212 = M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1) ;
    real A0313 = M(1, 0) * M(3, 3) - M(1, 3) * M(3, 0) ;
    real A0213 = M(1, 0) * M(3, 2) - M(1, 2) * M(3, 0) ;
    real A0312 = M(1, 0) * M(2, 3) - M(1, 3) * M(2, 0) ;
    real A0212 = M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0) ;
    real A0113 = M(1, 0) * M(3, 1) - M(1, 1) * M(3, 0) ;
    real A0112 = M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0) ;
    return { invdet * ( M(1, 1) * A2323 - M(1, 2) * A1323 + M(1, 3) * A1223 ),
        invdet * - ( M(0, 1) * A2323 - M(0, 2) * A1323 + M(0, 3) * A1223 ),
        invdet *   ( M(0, 1) * A2313 - M(0, 2) * A1313 + M(0, 3) * A1213 ),
        invdet * - ( M(0, 1) * A2312 - M(0, 2) * A1312 + M(0, 3) * A1212 ),
        invdet * - ( M(1, 0) * A2323 - M(1, 2) * A0323 + M(1, 3) * A0223 ),
        invdet *   ( M(0, 0) * A2323 - M(0, 2) * A0323 + M(0, 3) * A0223 ),
        invdet * - ( M(0, 0) * A2313 - M(0, 2) * A0313 + M(0, 3) * A0213 ),
        invdet *   ( M(0, 0) * A2312 - M(0, 2) * A0312 + M(0, 3) * A0212 ),
        invdet *   ( M(1, 0) * A1323 - M(1, 1) * A0323 + M(1, 3) * A0123 ),
        invdet * - ( M(0, 0) * A1323 - M(0, 1) * A0323 + M(0, 3) * A0123 ),
        invdet *   ( M(0, 0) * A1313 - M(0, 1) * A0313 + M(0, 3) * A0113 ),
        invdet * - ( M(0, 0) * A1312 - M(0, 1) * A0312 + M(0, 3) * A0112 ),
        invdet * - ( M(1, 0) * A1223 - M(1, 1) * A0223 + M(1, 2) * A0123 ),
        invdet *   ( M(0, 0) * A1223 - M(0, 1) * A0223 + M(0, 2) * A0123 ),
        invdet * - ( M(0, 0) * A1213 - M(0, 1) * A0213 + M(0, 2) * A0113 ),
        invdet *   ( M(0, 0) * A1212 - M(0, 1) * A0212 + M(0, 2) * A0112 ) };
}

// using 4x4 transforms ////////////////////////////////////////////////////////

tuDv transformPoint(const mat4 &M, vecD p) {
    vec4 p_hom = {};
    memcpy(&p_hom, &p, D * sizeof(real));
    p_hom.w = 1;
    vec4 ret_hom = M * p_hom;
    ret_hom /= ret_hom.w;
    vecD result = {};
    memcpy(&result, &ret_hom, D * sizeof(real));
    return result;
}
tuDv transformVector(const mat4 &M, vecD v) {
    vec3 v_3D = {};
    memcpy(&v_3D, &v, D * sizeof(real));
    vec3 ret_hom = _M3(M) * v_3D;
    vecD result;
    memcpy(&result, &ret_hom, D * sizeof(real));
    return result;
}
tuDv transformNormal(const mat4 &M, vecD n) {
    vec3 ret_hom = inverse(transpose(_M3(M))) * n;
    vecD result;
    memcpy(&result, &ret_hom, D * sizeof(real));
    return result;
}

// 4x4 transform cookbook //////////////////////////////////////////////////////

tuDm identityMatrix() {
    matD result = {};
    for_(i, D) {
        result(i, i) = 1;
    }
    return result;
}
const mat4 _Identity4x4 = identityMatrix<4>();

mat4 M4_Identity() {
    return _Identity4x4; // FORNOW
}

mat4 M4_Translation(real x, real y, real z = 0) {
    mat4 result = _Identity4x4;
    result(0, 3) = x;
    result(1, 3) = y;
    result(2, 3) = z;
    return result;
}
mat4 M4_Translation(vec2 xy) {
    return M4_Translation(xy.x, xy.y);
}
mat4 M4_Translation(vec3 xyz) {
    return M4_Translation(xyz.x, xyz.y, xyz.z);
}
mat4 M4_Scaling(real x, real y, real z = 1) {
    mat4 result = {};
    result(0, 0) = x;
    result(1, 1) = y;
    result(2, 2) = z;
    result(3, 3) = 1;
    return result;
}
mat4 M4_Scaling(real s) {
    return M4_Scaling(s, s, s);
}
mat4 M4_Scaling(vec2 xy) {
    return M4_Scaling(xy.x, xy.y);
}
mat4 M4_Scaling(vec3 xyz) {
    return M4_Scaling(xyz.x, xyz.y, xyz.z);
}
mat4 M4_RotationAboutXAxis(real t) {
    mat4 result = _Identity4x4;
    result(1, 1) = COS(t); result(1, 2) = -SIN(t);
    result(2, 1) = SIN(t); result(2, 2) =  COS(t);
    return result;
}
mat4 M4_RotationAboutYAxis(real t) {
    mat4 result = _Identity4x4;
    result(0, 0) =  COS(t); result(0, 2) = SIN(t);
    result(2, 0) = -SIN(t); result(2, 2) = COS(t);
    return result;
}
mat4 M4_RotationAboutZAxis(real t) {
    mat4 result = _Identity4x4;
    result(0, 0) = COS(t); result(0, 1) = -SIN(t);
    result(1, 0) = SIN(t); result(1, 1) =  COS(t);
    return result;
}

mat4 M4_RotationAbout(vec3 axis, real angle) {
    real x = axis.x;
    real y = axis.y;
    real z = axis.z;
    real x2 = x * x;
    real y2 = y * y;
    real z2 = z * z;
    real xy = x * y;
    real xz = x * z;
    real yz = y * z;
    real col = COS(angle);
    real s = SIN(angle);
    real d = 1-col;
    return { col+x2*d, xy*d-z*s, xz*d+y*s, 0,
        xy*d+z*s, col+y2*d, yz*d-x*s, 0,
        xz*d-y*s, yz*d+x*s, col+z2*d, 0,
        0, 0, 0, 1
    };
}

mat4 M4_RotationFrom(vec3 a, vec3 b) {
    // NOTE: twist dof is whatever
    // https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    // FORNOW
    a = normalized(a);
    b = normalized(b);

    vec3 v = cross(a, b);
    real col = dot(a, b);
    if (ABS(col + 1.0f) < 1e-5f) return M4_Identity();
    mat3 v_x = { 0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0 };
    mat3 R = identityMatrix<3>() + v_x + v_x * v_x / (1 + col);
    return {
        R.data[0], R.data[1], R.data[2], 0.0,
            R.data[3], R.data[4], R.data[5], 0.0,
            R.data[6], R.data[7], R.data[8], 0.0,
            0.0,       0.0,       0.0, 1.0 };
}


// optimization stuff //////////////////////////////////////////////////////////

tuDm firstDerivativeofUnitVector(vecD v) {
    vecD tmp = normalized(v);
    return (1 / norm(v)) * (identityMatrix<D>() - outer(tmp, tmp));
}
#define firstDerivativeOfNorm normalized
#define secondDerivativeOfNorm firstDerivativeofUnitVector

tuD real squaredNorm(matD M) {
    real result = 0;
    for(uint i = 0; i < D * D; ++i) {
        result += M[i] * M[i];
    }
    return result;
}

// misc functions //////////////////////////////////////////////////////////////

tuD real minComponent(vecD A) {
    real result = HUGE_VAL;
    for(uint i = 0; i < D; ++i) result = MIN(result, A[i]);
    return result;
}

tuD real maxComponent(vecD A) {
    real result = -HUGE_VAL;
    for(uint i = 0; i < D; ++i) result = MAX(result, A[i]);
    return result;
}

tuDv cwiseAbs(vecD A) {
    for(uint i = 0; i < D; ++i) A[i] = abs(A[i]);
    return A;
}
tuDv cwiseMin(vecD A, vecD B) {
    vecD result = {};
    for(uint i = 0; i < D; ++i) result[i] = (A[i] < B[i]) ? A[i] : B[i];
    return result;
}
tuDv cwiseMax(vecD A, vecD B) {
    vecD result = {};
    for(uint i = 0; i < D; ++i) result[i] = (A[i] > B[i]) ? A[i] : B[i];
    return result;
}
tuDv cwiseProduct(vecD a, vecD b) {
    vecD result = {};
    for(uint i = 0; i < D; ++i) result[i] = a[i] * b[i];
    return result;
}
vec2 e_theta(real theta) {
    return { COS(theta), SIN(theta) };
}
real ATAN2(vec2 a) {
    return ATAN2(a.y, a.x);
}
vec2 rotated(vec2 a, real theta) {
    return { COS(theta) * a.x - SIN(theta) * a.y, SIN(theta) * a.x + COS(theta) * a.y };
}
vec2 rotated_about(vec2 a, vec2 o, real theta) {
    return rotated(a - o, theta) + o;
}
vec2 scaled_about(vec2 a, vec2 o, real scale) {
    return scale * (a - o) + o;
}
mat2 R_theta_2x2(real theta) {
    return { COS(theta), -SIN(theta), SIN(theta), COS(theta) };
}
vec2 perpendicularTo(vec2 v) {
    return { v.y, -v.x };
}

mat4 xyzo2mat4(vec3 x, vec3 y, vec3 z, vec3 o) {
    return {
        x[0], y[0], z[0], o[0],
        x[1], y[1], z[1], o[1],
        x[2], y[2], z[2], o[2],
        0, 0, 0, 1
    };
}
#define M4_xyzo xyzo2mat4
tuDv magClamped(vecD a, real col) {
    real norm_a = norm(a);
    if (ABS(norm_a) < col) { return a; }
    return a / norm_a * MAG_CLAMP(norm_a, col);
}

// utility /////////////////////////////////////////////////////////////////////

tuD void pprint(vecD A) {
    printf("V%d(", D);
    for_(i, D) {
        printf("%lf", A[i]);
        if (i != D - 1) printf(", ");
    }
    printf(")\n");
}
tuD void pprint(SnailTupleOfUnsignedInts<D> A) {
    printf("U%d(", D);
    for_(i, D) {
        printf("%d", A[i]);
        if (i != D - 1) printf(", ");
    }
    printf(")\n");
}
tuD void pprint(matD M) {
    for_(row, D) {
        printf("| ");
        for_(col, D) {
            printf("%lf", M(row, col));
            if (col != D - 1) printf(", ");
        }
        printf(" |\n");
    }
}

// math math ///////////////////////////////////////////////////////////////////

struct RayTriangleIntersectionResult {
    bool hit;
    real distance;
    vec3 pos;
};
RayTriangleIntersectionResult ray_triangle_intersection(vec3 o, vec3 dir, vec3 a, vec3 b, vec3 c) {
    RayTriangleIntersectionResult result = {};
    vec4 w_t = inverse(M4(
                a[0], b[0], c[0], -dir[0],
                a[1], b[1], c[1], -dir[1],
                a[2], b[2], c[2], -dir[2],
                1.0f, 1.0f, 1.0f,     0.0))
        * V4(o, 1.0f);
    result.hit = ((w_t.x > -TINY_VAL) && (w_t.y > -TINY_VAL) && (w_t.z > -TINY_VAL) && (w_t.w > -TINY_VAL));
    result.distance = w_t.w;
    result.pos = o + dir * result.distance;
    return result;
}


// <!> End linalg.cpp <!>
// <!> Begin string.cpp <!> 
struct String {
    char *data;
    uint length;
};

bool string_pointer_is_valid(String string, char *pointer) {
    return (string.data <= pointer) && (pointer < string.data + string.length);
}

#define STRING(cstring_literal) { (char *)(cstring_literal), uint(strlen(cstring_literal)) }

#define _STRING_CALLOC(name, length) String name = { (char *) calloc(1, length) }

String _string_from_cstring(char *cstring) {
    return { (char *)(cstring), uint(strlen(cstring)) };
}

bool string_matches_prefix(String string, String prefix) {
    if (string.length < prefix.length) return false;
    return (memcmp(string.data, prefix.data, prefix.length) == 0);
}

bool string_matches_prefix(String string, char *prefix) {
    return string_matches_prefix(string, STRING(prefix));
}

bool string_matches_suffix(String string, String suffix) {
    if (string.length < suffix.length) return false;
    return (memcmp(&string.data[string.length - suffix.length], &suffix.data[suffix.length - suffix.length], suffix.length) == 0);
}

bool string_matches_suffix(String string, char *prefix) {
    return string_matches_suffix(string, STRING(prefix));
}

bool string_equal(String string1, String string2) {
    if (string1.length != string2.length) return false;

    for_(i, string1.length) {
        if (string1.data[i] != string2.data[i]) return false;
    }
    return true;
}

real strtof(String string) { // FORNOW
    static char cstring[4096];
    memset(cstring, 0, sizeof(cstring));
    ASSERT(string.length < sizeof(cstring));
    memcpy(cstring, string.data, string.length);
    return strtof(cstring, NULL);
}

bool string_read_line_from_file(String *string, uint max_line_length, FILE *file) {
    bool result = fgets(string->data, max_line_length, file);
    if (result) string->length = uint(strlen(string->data));
    return result;
}

FILE *FILE_OPEN(String filename, char *code, bool skip_assert = false) { // FORNOW
    static char cstring[4096];
    memset(cstring, 0, sizeof(cstring));
    ASSERT(filename.length < sizeof(cstring));
    memcpy(cstring, filename.data, filename.length);
    FILE *result = fopen(cstring, code);
    if (!skip_assert) ASSERT(result);
    return result;
}

bool FILE_EXISTS(String filename) {
    FILE *file = FILE_OPEN(filename, "r", true);
    if (!file) {
        return false;
    }
    fclose(file);
    return true;
}

bool FGETS(String *line, uint LINE_MAX_LENGTH, FILE *file) {
    ASSERT(line);
    ASSERT(file);
    bool result = fgets(line->data, LINE_MAX_LENGTH, file);
    line->length = uint(strlen(line->data));
    return result;
}
// <!> End string.cpp <!>

// <!> Begin window.cpp <!> 
// TODO: an fps camera shouldn't be allowed to be ortho
// TODO: camera should have clip planes as member variables

#ifdef OPERATING_SYSTEM_APPLE
#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_GL_COREARB
#include <OpenGL/gl3.h>
#include "glfw3.h"
#elif defined(OPERATING_SYSTEM_WINDOWS)
#include "glad.c"
#include "glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#include "glfw3native.h"
#endif

GLFWwindow *glfw_window;
real _window_macbook_retina_fixer_VERY_MYSTERIOUS;

// #define DEBUG_OPENGL

#ifdef OPERATING_SYSTEM_WINDOWS
GLenum glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        char *error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        void MESSAGE_FAILURE(char *format, ...);
        MESSAGE_FAILURE("GL ERROR %s %s %d\n", error, file, line);
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__) 

#ifdef DEBUG_OPENGL
void APIENTRY glDebugOutput(GLenum source, 
                            GLenum type, 
                            unsigned int id, 
                            GLenum severity, 
                            GLsizei length, 
                            const char *message, 
                            const void *userParam)
{
    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 

    char *msg_source = NULL;
    char *msg_type = NULL;
    char *msg_severity = NULL;

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             msg_source = "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   msg_source = "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: msg_source = "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     msg_source = "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     msg_source = "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           msg_source = "Source: Other"; break;
    }

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               msg_type = "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: msg_type = "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  msg_type = "Type: Undefined Behaviour"; break; 
        case GL_DEBUG_TYPE_PORTABILITY:         msg_type = "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         msg_type = "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              msg_type = "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          msg_type = "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           msg_type = "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               msg_type = "Type: Other"; break;
    }
    
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         msg_severity = "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       msg_severity = "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          msg_severity = "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: msg_severity = "Severity: notification"; break;
    } 
    
    if (msg_source) MESSAGE_ERROR("%s", msg_source);
    if (msg_type) MESSAGE_ERROR("%s", msg_type);
    if (msg_severity) MESSAGE_ERROR("%s", msg_severity);
}
*/
#endif
#else
#define glCheckError() 
#endif


run_before_main {
    ASSERT(glfwInit());

    #ifndef DEBUG_OPENGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // 3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1); // 3
    #else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    #endif
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 1);
    glfwWindowHint(GLFW_STENCIL_BITS, 8); // ?

    #ifdef DEBUG_OPENGL
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
    #endif


    // this is to make it full screen and stuff
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    glfw_window = glfwCreateWindow(mode->width - 16, mode->height - 88, __DATE__ " " __TIME__, NULL, NULL);
    glfwSetWindowPos(glfw_window, 0, 30);
    // glfw_window = glfwCreateWindow(960, 720,  __DATE__ " " __TIME__, NULL, NULL);
    if (!glfw_window) {
        printf("Something's gone wonky; if you weren't just messing with init(...) or something, please try restarting your computer and trying again.\n");
        ASSERT(0);
    }
    glfwMakeContextCurrent(glfw_window);

    #ifdef DEBUG_OPENGL
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); 
        glDebugMessageCallback(glDebugOutput, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    }
    #endif


    // not need for full screen but keeping in case we want window
    // glfwSetWindowPos(glfw_window, 0, 100);
    // glfwSetWindowAttrib(glfw_window, GLFW_FLOATING, false);
    // glfwSetWindowAttrib(glfw_window, GLFW_DECORATED, true);


    #ifdef OPERATING_SYSTEM_WINDOWS
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    #endif

    // glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // ?
    glEnable(GL_CULL_FACE);                    // ?
    glCullFace(GL_BACK);                       // ?
    glfwSwapInterval(1);

    { // _macbook_retina_scale
        int num, den, _;
        glfwGetFramebufferSize(glfw_window, &num, &_);
        glfwGetWindowSize(glfw_window, &den, &_);
        _window_macbook_retina_fixer_VERY_MYSTERIOUS = real(num / den);
    }


};


vec2 window_get_size_Pixel() {
    ASSERT(glfw_window);
    int _width, _height;
    glfwGetFramebufferSize(glfw_window, &_width, &_height);
    real width = real(_width) / _window_macbook_retina_fixer_VERY_MYSTERIOUS;
    real height = real(_height) / _window_macbook_retina_fixer_VERY_MYSTERIOUS;
    return { width, height };
}
uint window_get_width_Pixel() { return window_get_size_Pixel().x; }
uint window_get_height_Pixel() { return window_get_size_Pixel().y; }


void gl_scissor_Pixel(double x, double y, double dx, double dy) {
    // y_Pixel_upper_left -> y_Scissor_upper_left -> y_Scissor_lower_left
    y = window_get_height_Pixel() - y - dy;
    real factor = _window_macbook_retina_fixer_VERY_MYSTERIOUS;
    glScissor(uint(factor * x), uint(factor * y), uint(factor * dx), uint(factor * dy));
}
#ifdef glScissor
#undef glScissor
#endif
#define glScissor RETINA_BREAKS_THIS_FUNCTION_USE_gl_scissor_WRAPPER



real window_get_aspect() {
    vec2 size = window_get_size_Pixel();
    return size.x / size.y;
}

mat4 window_get_NDC_from_Pixel() {
    // NDC                            Pixel 
    // [x'] = [1/r_x      0   0 -1] [x] = [x/r_x - 1]
    // [y'] = [    0 -1/r_y   0  1] [y] = [1 - y/r_y]
    // [z'] = [    0      0   1  0] [z] = [        z] // so invertible (otherwise, would just have row [ 0 0 0 0 ]
    // [1 ] = [    0      0   0  1] [1] = [        1]
    vec2 r = window_get_size_Pixel() / 2;
    mat4 result = {};
    result(0, 0) = 1.0f / r.x;
    result(1, 1) = -1.0f / r.y;
    result(2, 2) = 1.0f;
    result(0, 3) = -1.0f;
    result(1, 3) = 1.0f;
    result(3, 3) = 1.0f;
    return result;
}

mat4 transform_get_P_persp(real angle_of_view, vec2 post_nudge_OpenGL = {}, real near_z_Camera = 0, real far_z_Camera = 0, real aspect = 0) {
    if (IS_ZERO(near_z_Camera)) { near_z_Camera = -0.100f; }
    if (IS_ZERO(far_z_Camera)) { far_z_Camera = -1000.0f; }
    if (IS_ZERO(aspect)) { aspect = window_get_aspect(); }
    // ASSERT(near_z_Camera < 0.0f);
    // ASSERT(far_z_Camera < 0.0f);

    // consider a point with coordinates (x, y, -z) in the camera's coordinate system
    //                                                                   where z < 0*
    //                              *recall that the camera's z-axis points backwards

    // 1) imagine projecting the point onto some film plane with height r_y and distance D

    //                r_y                               
    //               -|                                 
    //              - |                                 
    //  angle_y    -  |           y <~ vertex           
    //         \  -   |       -   |                     
    //          |-    |   -       |                     
    //          v     +           |                     
    //         -) -   |           |                     
    //        0-------+-----------+----->               
    //                D          -z                     

    // 2) scale film plane by 1 / r_y to yield OpenGL film plane (with height 1) and distance Q_y
    // y' is the projected position of vertex y in OpenGL; i.e., if we can get y', we're done :) 

    //                1 <~ edge of OpenGL film plane
    //               -|                          
    //              - |                          
    //  angle_y    -  |           y              
    //         \  -   |       -   |              
    //          |-    |   -       |              
    //          v     y'          |              
    //         -) -   |           |              
    //        0-------+-----------+----->        
    //              D / r_y      -z              
    //                ^                          
    //                |                          
    //                cot(angle_y) := Q_y        

    // similar triangles has y' / Q_y = y / -z                     
    //                          => y' = -Q_y * (y / z) (Equation 1)

    // we can repeat this procedure in x      
    // the only difference is Q_x vs. Q_y     
    // -------------------------------------- 
    // cot(angle_x) = D / r_x                 
    // cot(angle_y) = D / r_y                 
    // => r_x cot(angle_x) = r_y cot(angle_y) 
    // recall: aspect := r_x / r_y            
    //  => aspect cot(angle_x) = cot(angle_y) 
    //                  => Q_x = Q_y / aspect.

    // encode Equation 1 (and the variant for x) into a homogeneous matrix equation
    // the third row is a   typical clip plane mapping                             

    //  OpenGL                    Camera
    //  [x'] = [Q_x   0  0  0] [x] = [ Q_x * x] ~> [-Q_x * (x / z)]
    //  [y'] = [  0 Q_y  0  0] [y] = [ Q_y * y] ~> [-Q_y * (y / z)]
    //  [z'] = [  0   0  a  b] [z] = [  az + b] ~> [      -a - b/z]
    //  [ 1] = [  0   0 -1  0] [1] = [      -z] ~> [             1]

    real angle_y = 0.5f * angle_of_view;
    real Q_y = 1 / TAN(angle_y);
    real Q_x = Q_y / aspect;

    mat4 result = {};
    result(0, 0) = Q_x;
    result(1, 1) = Q_y;
    result(3, 2) = -1;

    // z'(z) = [-a - b/z]              
    // we want to map [n, f] -> [-1, 1]
    // z'(n) = -a - b/n := -1          
    // z'(f) = -a - b/f :=  1          
    //                                 
    // => a + b/n =  1                 
    //    a + b/f = -1                 
    // => b/n - b/f = 2                
    //                                 
    // => b * (f - n) / (n * f) = 2    
    // => b = (2 * n * f) / (f - n)    
    //                                 
    // => a + (2 * f) / (f - n) = 1    
    // => a = -(n + f) / (f - n)       
    //       = (n + f) / (n - f)       
    result(2, 2) = (near_z_Camera + far_z_Camera) / (near_z_Camera - far_z_Camera);
    result(2, 3) = (2 * near_z_Camera * far_z_Camera) / (far_z_Camera - near_z_Camera);

    // [1 0 0  t_x_OpenGL] [Q_x   0  0  0]
    // [0 1 0  t_y_OpenGL] [  0 Q_y  0  0]
    // [0 0 1           0] [  0   0  a  b]
    // [0 0 0           1] [  0   0 -1  0]
    result(0, 2) = -post_nudge_OpenGL.x;
    result(1, 2) = -post_nudge_OpenGL.y;

    return result;
}

mat4 transform_get_P_ortho(real height_World, vec2 post_nudge_OpenGL = {}, real near_z_Camera = 0, real far_z_Camera = 0, real aspect = 0) {
    // ASSERT(!IS_ZERO(height_World));
    if (ARE_EQUAL(near_z_Camera, far_z_Camera)) {
        near_z_Camera = 1000000.0f;
        far_z_Camera = -near_z_Camera;
    }
    if (IS_ZERO(aspect)) { aspect = window_get_aspect(); }

    // consider a point with coordinates (x, y, z) in the camera's coordinate system

    // 1) imagine projecting the point onto some film plane with height r_y

    // r_y                                  
    // |                                    
    // |                                    
    // +-----------y                        
    // |           |                        
    // |           |                        
    // +-----------------> minus_z direction

    // 2) scale everything by 1 / r_y to yield OpenGL film plane (with height 1)

    // 1                                     
    // |                                     
    // |                                     
    // y'----------y / r_y                   
    // |           |                         
    // |           |                         
    // +-----------------> minus_z  direction

    // => y' = y / r_y

    // OpenGL                        Camera
    // [x'] = [1/r_x      0   0  0] [x] = [ x/r_x]
    // [y'] = [    0  1/r_y   0  0] [y] = [ y/r_y]
    // [z'] = [    0      0   a  b] [z] = [az + b]
    // [1 ] = [    0      0   0  1] [1] = [     1]

    // z'(z) = [az + b]                
    // we want to map [n, f] -> [-1, 1]
    // z'(n) = an + b := -1            
    // z'(f) = af + b :=  1            
    //                                 
    // => a * (f - n) = 2              
    //    a = 2 / (f - n)              
    //                                 
    // (2 * f) / (f - n) + b = 1       
    // => b = (n + f) / (n - f)        

    real r_y = height_World / 2;
    real r_x = window_get_aspect() * r_y;
    real a = 2.0f / (far_z_Camera - near_z_Camera);
    real b = (near_z_Camera + far_z_Camera) / (near_z_Camera - far_z_Camera);

    mat4 result = {};
    result(0, 0) = 1.0f / r_x;
    result(1, 1) = 1.0f / r_y;
    result(2, 2) = a;
    result(2, 3) = b;
    result(3, 3) = 1.0f;

    // [1 0 0  t_x] [1/r_x      0   0  0]
    // [0 1 0  t_y] [    0  1/r_y   0  0]
    // [0 0 1    0] [    0      0   a  b]
    // [0 0 0    1] [    0      0   0  1]

    result(0, 3) = post_nudge_OpenGL.x;
    result(1, 3) = post_nudge_OpenGL.y;

    return result;
}

enum class CameraType {
    None,
    Camera2D,
    OrbitCamera3D,
    FirstPersonCamera3D,
};

struct Camera {
    CameraType type;
    real angle_of_view; // NOTE: 0.0f <=> ortho camera
    vec3 euler_angles;
    union {
        struct {
            vec2 pre_nudge_World;
            union {
                real persp_distance_to_origin_World;
                real ortho_screen_height_World;
            };
        };
        vec3 first_person_position_World;
    };
    vec2 post_nudge_OpenGL;

    real near_plane;
    real far_plane;

    mat4 get_P();
    mat4 get_V();
    mat4 get_PV();

    real camera_get_screen_height_World();
};

Camera make_Camera2D(real screen_height_World, vec2 center_World = {}, vec2 post_nudge_OpenGL = {}) {
    Camera result = {};
    result.type = CameraType::Camera2D;
    result.ortho_screen_height_World = screen_height_World;
    result.pre_nudge_World = center_World;
    result.post_nudge_OpenGL = post_nudge_OpenGL;
    return result;
}

Camera make_OrbitCamera3D(real distance_to_origin_World, real angle_of_view = RAD(60.0f), vec3 euler_angles = {}, vec2 pre_nudge_World = {}, vec2 post_nudge_OpenGL = {}) {
    Camera result = {};
    result.type = CameraType::OrbitCamera3D;
    result.angle_of_view = angle_of_view;
    result.persp_distance_to_origin_World = distance_to_origin_World;
    result.euler_angles = euler_angles;
    result.pre_nudge_World = pre_nudge_World;
    result.post_nudge_OpenGL = post_nudge_OpenGL;
    return result;
}

Camera make_FirstPersonCamera3D(vec3 first_person_position_World, real angle_of_view = RAD(60.0f), vec3 euler_angles = {}, vec2 post_nudge_OpenGL = {}) {
    Camera result = {};
    result.type = CameraType::FirstPersonCamera3D;
    result.angle_of_view = angle_of_view;
    result.euler_angles = euler_angles;
    result.first_person_position_World = first_person_position_World;
    result.post_nudge_OpenGL = post_nudge_OpenGL;
    return result;
}

Camera make_EquivalentCamera2D(Camera *orbit_camera_3D) {
    ASSERT(orbit_camera_3D->type == CameraType::OrbitCamera3D);
    bool is_perspective_camera = (!IS_ZERO(orbit_camera_3D->angle_of_view));
    Camera result; {
        result = *orbit_camera_3D;
        result.type = CameraType::Camera2D;
        result.angle_of_view = 0.0f;
        result.euler_angles = {};
        if (is_perspective_camera) result.ortho_screen_height_World = 2.0f * (orbit_camera_3D->persp_distance_to_origin_World * TAN(0.5f * orbit_camera_3D->angle_of_view));
    }
    return result;
}


mat4 Camera::get_P() {
    if (IS_ZERO(this->angle_of_view)) {
        return transform_get_P_ortho(this->ortho_screen_height_World, this->post_nudge_OpenGL, this->near_plane, this->far_plane);
    } else {
        return transform_get_P_persp(this->angle_of_view, this->post_nudge_OpenGL, this->near_plane, this->far_plane);
    }
}

mat4 Camera::get_V() {
    mat4 C; {
        mat4 T = M4_Translation(this->first_person_position_World);
        mat4 R_x = M4_RotationAboutXAxis(this->euler_angles.x);
        mat4 R_y = M4_RotationAboutYAxis(this->euler_angles.y);
        mat4 R_z = M4_RotationAboutZAxis(this->euler_angles.z);
        mat4 R = (R_y * R_x * R_z);
        C = (this->type != CameraType::FirstPersonCamera3D) ? (R * T) : (T * R);
    }
    return inverse(C);
}

mat4 Camera::get_PV() { return get_P() * get_V(); }

real Camera::camera_get_screen_height_World() { return IS_ZERO(this->angle_of_view) ? this->ortho_screen_height_World : 2.0f * (this->persp_distance_to_origin_World * TAN(0.5f * this->angle_of_view)); }

// <!> End window.cpp <!>

// <!> Begin shader.cpp <!> 
uint shader_compile(char *source, GLenum type) {
    uint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    {
        int success = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            { // log
                char infoLog[512];
                glGetShaderInfoLog(shader, 512, NULL, infoLog);
                printf("%s\n", source);
                printf("%s", infoLog);
            }
            ASSERT(0);
        }
    }
    ASSERT(shader);
    return shader;
};

uint shader_build_program(uint vertex_shader, uint geometry_shader, uint fragment_shader) {
    uint shader_program_ID = glCreateProgram();
    ASSERT(shader_program_ID);
    ASSERT(vertex_shader); glAttachShader(shader_program_ID, vertex_shader);
    if (geometry_shader) glAttachShader(shader_program_ID, geometry_shader);
    ASSERT(fragment_shader); glAttachShader(shader_program_ID, fragment_shader);
    glLinkProgram(shader_program_ID);
    {
        int success = 0;
        glGetProgramiv(shader_program_ID, GL_LINK_STATUS, &success);
        if (!success) {
            { // log
                char infoLog[512];
                glGetProgramInfoLog(shader_program_ID, 512, NULL, infoLog);
                printf("%s", infoLog);
            }
            ASSERT(0);
        }
    }
    return shader_program_ID;
};

uint shader_compile_and_build_program(char *vertex_shader_source, char *fragment_shader_source, char *geometry_shader_source = NULL) {
    uint vert = shader_compile(vertex_shader_source, GL_VERTEX_SHADER);
    uint frag = shader_compile(fragment_shader_source, GL_FRAGMENT_SHADER);
    uint geom = geometry_shader_source ? shader_compile(geometry_shader_source, GL_GEOMETRY_SHADER) : 0;
    return shader_build_program(vert, frag, geom);
}

// <!> End shader.cpp <!>
// <!> Begin color.cpp <!> 
struct {
    vec3 red     = { 1.0f, 0.0f, 0.0f };
    vec3 green   = { 0.0f, 1.0f, 0.0f };
    vec3 blue    = { 0.0f, 0.0f, 1.0f };
    vec3 yellow  = { 1.0f, 1.0f, 0.0f };
    vec3 cyan    = { 0.0f, 1.0f, 1.0f };
    vec3 magenta = { 1.0f, 0.0f, 1.0f };
    vec3 white         = V3(1.0f);
    vec3 lightest_gray = V3(0.9375f);
    vec3 lighter_gray  = V3(0.875f);
    vec3 light_gray    = V3(0.75f);
    vec3 gray          = V3(0.5f);
    vec3 dark_gray     = V3(0.25f);
    vec3 darker_gray   = V3(0.125f);
    vec3 darkest_gray  = V3(0.0625f);
    vec3 black         = V3(0.0f);
} basic;

vec3 RGB255(uint r, uint g, uint b) { return V3(real(r), real(g), real(b)) / 255.0f; }

struct {
    // NOTE: not actual monokai yellow cause i don't like it
    vec3 red        = RGB255(249,  38, 114);
    vec3 orange     = RGB255(253, 151,  31);
    vec3 yellow     = RGB255(255, 255,  50);
    vec3 green      = RGB255(166, 226,  46);
    vec3 blue       = RGB255(102, 217, 239);
    vec3 indigo     = RGB255(138, 173, 247);
    vec3 violet     = RGB255(174, 129, 255);
    vec3 white      = RGB255(255, 255, 255); // *shrug*
    vec3 light_gray = RGB255(192, 192, 192); // *shrug*
    vec3 gray       = RGB255(127, 127, 127); // *shrug*
    vec3 dark_gray  = RGB255( 64,  64,  64); // *shrug*
    vec3 black      = RGB255(  0,   0,   0); // *shrug*
    vec3 brown      = RGB255(123,  63,   0); // no actual brown
} monokai;

vec3 get_kelly_color(int i) {
    static vec3 _kelly_colors[]={{255.f/255,179.f/255,0.f/255},{128.f/255,62.f/255,117.f/255},{255.f/255,104.f/255,0.f/255},{166.f/255,189.f/255,215.f/255},{193.f/255,0.f/255,32.f/255},{206.f/255,162.f/255,98.f/255},{129.f/255,112.f/255,102.f/255},{0.f/255,125.f/255,52.f/255},{246.f/255,118.f/255,142.f/255},{0.f/255,83.f/255,138.f/255},{255.f/255,122.f/255,92.f/255},{83.f/255,55.f/255,122.f/255},{255.f/255,142.f/255,0.f/255},{179.f/255,40.f/255,81.f/255},{244.f/255,200.f/255,0.f/255},{127.f/255,24.f/255,13.f/255},{147.f/255,170.f/255,0.f/255},{89.f/255,51.f/255,21.f/255},{241.f/255,58.f/255,19.f/255},{35.f/255,44.f/255,22.f/255}};
    return _kelly_colors[MODULO(i, ARRAY_LENGTH(_kelly_colors))];
}

vec3 color_rainbow_swirl(real t) {
    return {
        (0.5f + 0.5f * COS(TAU * ( 0.000f - t))),
        (0.5f + 0.5f * COS(TAU * ( 0.333f - t))),
        (0.5f + 0.5f * COS(TAU * (-0.333f - t)))
    };
}

// <!> End color.cpp <!>
// <!> Begin soup.cpp <!> 
// XXXX: remove option to pass null in soup_draw
// XXXX: remove rounded edges
// NOTE: (soup draw should essentially never be called by the user)
// TODO: per-vertex size
// TODO: per-vertex stipple
// TODO: properly draw meshes ala that one nvidia white paper

////////////////////////////////////////////////////////////////////////////////
// soup ////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

real global_time; // fornow

#define SOUP_POINTS         GL_POINTS
#define SOUP_LINES          GL_LINES
#define SOUP_LINE_STRIP     GL_LINE_STRIP
#define SOUP_LINE_LOOP      GL_LINE_LOOP
#define SOUP_TRIANGLES      GL_TRIANGLES
#define SOUP_TRIANGLE_FAN   GL_TRIANGLE_FAN
#define SOUP_TRIANGLE_STRIP GL_TRIANGLE_STRIP
#define SOUP_QUADS          255
#define SOUP_TRI_MESH       254 // TODO

struct {
    char *vert = R""(#version 330 core
        layout (location = 0) in vec3 vertex;
        layout (location = 1) in vec4 color;
        layout (location = 2) in float size;

        out BLOCK {
            vec4 color;
            float size;
        } vs_out;

        uniform mat4 transform;
        uniform mat4 M;
        uniform bool force_draw_on_top;

// https://prideout.net/clip-planes
// One word of caution: I’ve heard that some drivers ignore the enable/disable state of GL_CLIP_DISTANCE0.

        uniform bool clip;
        uniform vec4 clip_plane;

        void main() {
            if (clip) {
                vec4 vertex_World = M * vec4(vertex, 1);
                gl_ClipDistance[0] = dot(vertex_World, clip_plane); // FORNOW 
            }

            gl_Position = transform * vec4(vertex, 1);
            if (force_draw_on_top) {
                // FORNOW (Feb 9 2025) switching from -.99* to -.995*
                // NOTE: -1.0 will lead to flickering bad bad
                // NOTE:: Michael thinks this fixes the flickering
                gl_Position.x /= gl_Position.w; 
                gl_Position.y /= gl_Position.w; 
                gl_Position.z = -1.0; // ?
                gl_Position.w = 1.0; // ?
            }
            vs_out.color = color;
            vs_out.size = size;
        }
    )"";

    char *geom_POINTS = R""(#version 330 core
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;
        uniform vec2 OpenGL_from_Pixel_scale;

        in BLOCK {
            vec4 color;
            float size;
        } gs_in[];

        out GS_OUT {
            vec4 color;
            vec2 xy;
        } gs_out;

        void emit(vec4 p, float x, float y) {
            vec2 radius = (gs_in[0].size / 2) * OpenGL_from_Pixel_scale;
            gs_out.color = gs_in[0].color;                                     
            gs_out.xy = vec2(x, y);
            gl_Position = (p + vec4(radius * gs_out.xy, 0, 0)) * gl_in[0].gl_Position.w;
            gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
            EmitVertex();                                               
        }

        void main() {    
            vec4 p = gl_in[0].gl_Position / gl_in[0].gl_Position.w;
            emit(p, -1, -1);
            emit(p, 1, -1);
            emit(p, -1, 1);
            emit(p, 1, 1);
            EndPrimitive();
        }  
    )"";

    char *frag_POINTS = R""(#version 330 core
        in GS_OUT {
            vec4 color;
            vec2 xy;
        } fs_in;

        out vec4 frag_color;

        void main() {
            frag_color = fs_in.color;
            if (length(fs_in.xy) > 1) { discard; }
        }
    )"";

    char *geom_LINES = R""(#version 330 core
        layout (lines) in;
        layout (triangle_strip, max_vertices = 4) out;
        uniform vec2 OpenGL_from_Pixel_scale;

        in BLOCK {
            vec4 color;
            float size;
        } gs_in[];

        out BLOCK {
            vec4 color;
            float size;
            vec2 position_Pixel; // NOTE: y flipped sorry
            float angle;
            vec2 starting_point_Pixel;
        } gs_out;

        void main() {    
            vec4 s = gl_in[0].gl_Position / gl_in[0].gl_Position.w;
            vec4 t = gl_in[1].gl_Position / gl_in[1].gl_Position.w;
            vec4 color_s = gs_in[0].color;
            vec4 color_t = gs_in[1].color;

            float angle = atan(OpenGL_from_Pixel_scale.x * (t.y - s.y), OpenGL_from_Pixel_scale.y * (t.x - s.x));

            vec2 perp = OpenGL_from_Pixel_scale * normalize(OpenGL_from_Pixel_scale * vec2(s.y - t.y, t.x - s.x));
            vec4 perp_s = vec4((gs_in[0].size / 2) * perp, 0, 0);
            vec4 perp_t = vec4((gs_in[1].size / 2) * perp, 0, 0);

            gl_Position = (s - perp_s) * gl_in[0].gl_Position.w;
            gs_out.position_Pixel = (vec2(1.0f) + gl_Position.xy) / OpenGL_from_Pixel_scale;
            gs_out.color = color_s;
            gs_out.angle = angle;
            gs_out.starting_point_Pixel = (vec2(1.0f) + s.xy * gl_in[0].gl_Position.w) / OpenGL_from_Pixel_scale;
            gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
            EmitVertex();

            gl_Position = (t - perp_t) * gl_in[1].gl_Position.w;
            gs_out.position_Pixel = (vec2(1.0f) + gl_Position.xy) / OpenGL_from_Pixel_scale;
            gs_out.color = color_t;
            gs_out.angle = angle;
            gs_out.starting_point_Pixel = (vec2(1.0f) + s.xy * gl_in[0].gl_Position.w) / OpenGL_from_Pixel_scale;
            gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
            EmitVertex();

            gl_Position = (s + perp_s) * gl_in[0].gl_Position.w;
            gs_out.position_Pixel = (vec2(1.0f) + gl_Position.xy) / OpenGL_from_Pixel_scale;
            gs_out.color = color_s;
            gs_out.angle = angle;
            gs_out.starting_point_Pixel = (vec2(1.0f) + s.xy * gl_in[0].gl_Position.w) / OpenGL_from_Pixel_scale;
            gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
            EmitVertex();

            gl_Position = (t + perp_t) * gl_in[1].gl_Position.w;
            gs_out.position_Pixel = (vec2(1.0f) + gl_Position.xy) / OpenGL_from_Pixel_scale;
            gs_out.color = color_t;
            gs_out.angle = angle;
            gs_out.starting_point_Pixel = (vec2(1.0f) + s.xy * gl_in[0].gl_Position.w) / OpenGL_from_Pixel_scale;
            gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
            EmitVertex();

            EndPrimitive();
        }  
    )"";

    char *frag_LINES = R""(#version 330 core
        uniform bool stipple;
        uniform float t_stipple;

        in BLOCK {
            vec4 color;
            float size;
            vec2 position_Pixel;
            float angle;
            vec2 starting_point_Pixel;
        } fs_in;

        out vec4 frag_color;

        void main() {
            frag_color = fs_in.color;
            if (stipple) {
                vec2 xy = fs_in.position_Pixel;
                // rotate by -angle
                float s = sin(fs_in.angle);
                float c = cos(fs_in.angle);
                mat2 Rinv = mat2(c, -s, s, c);
                vec2 uv = Rinv * (xy - fs_in.starting_point_Pixel); // FORNOW

                if (int(uv.x - 17 * t_stipple + 99999) % 10 > 5) discard; // FORNOW
            }
        }
    )"";

    char *frag_TRIANGLES = R""(#version 330 core
        in BLOCK {
            vec4 color;
            float size;
        } fs_in;

        out vec4 frag_color;

        void main() {
            frag_color = fs_in.color;
        }
    )"";

    char *geom_TRI_MESH = R""(#version 330 core
        layout (triangles) in;
        layout (triangle_strip, max_vertices = 3) out;

        uniform vec2 OpenGL_from_Pixel_scale;

        in BLOCK {
            vec4 color;
            float size;
        } gs_in[];

        out GS_OUT {
            vec4 color;
            noperspective vec3 heights;
            noperspective vec3 sizes;
        } gs_out;

        float point_line_dist(vec2 p, vec2 a, vec2 b) {
           vec2 line = b - a;
           vec2 n = vec2(line.y, -line.x);
           vec2 v = p - a;
           return abs(dot(v, n)) / length(n);
        }

        void main() {    
            vec3 sizes = vec3(gs_in[0].size, gs_in[1].size, gs_in[2].size);

            for (int d = 0; d < 3; ++d) {
                gl_Position = gl_in[d].gl_Position / gl_in[d].gl_Position.w;
                gs_out.color = gs_in[d].color;

                int e = (d + 1) % 3;
                int f = (d + 2) % 3;
                vec2 p = gl_in[d].gl_Position.xy / gl_in[d].gl_Position.w / OpenGL_from_Pixel_scale;
                vec2 q = gl_in[e].gl_Position.xy / gl_in[e].gl_Position.w / OpenGL_from_Pixel_scale;
                vec2 r = gl_in[f].gl_Position.xy / gl_in[f].gl_Position.w / OpenGL_from_Pixel_scale;
                vec3 heights = vec3(0);
                heights[d] = point_line_dist(p, q, r);
                vec3 bary = vec3(0);
                bary[d] = 1;

                gs_out.heights = heights;
                gs_out.sizes = sizes;
                gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
                EmitVertex();                                               
            }

            EndPrimitive();
        }  
    )"";

    char *frag_TRI_MESH = R""(#version 330 core
        in GS_OUT {
            vec4 color;
            noperspective vec3 heights;
            noperspective vec3 sizes;
        } fs_in;

        out vec4 frag_color;

        // TODO: eso_size should go in here, and you can remove the z-fight-y pass ...
        // NOTE: passing 0 for size should be NO edges
        void main() {
            int i = 0;
            if (fs_in.heights[1] < fs_in.heights[i]) i = 1;
            if (fs_in.heights[2] < fs_in.heights[i]) i = 2;

            if (fs_in.sizes[i] < 0.01) frag_color = fs_in.color;
            else {
                vec3 h = fs_in.heights / fs_in.sizes;
                float height = min(min(h.x, h.y), h.z);
                frag_color = mix(mix(vec4(0,0,0,1), fs_in.color, 0.5), fs_in.color, smoothstep(0.5, 1.0, height));
            }
        }
    )"";


} soup_source;

struct {
    uint shader_program_POINTS;
    uint shader_program_LINES;
    uint shader_program_TRIANGLES;
    uint shader_program_TRI_MESH;
    uint VAO[1];
    uint VBO[16];
    uint EBO[1];
} soup;

run_before_main {
    uint vert = shader_compile(soup_source.vert, GL_VERTEX_SHADER);
    uint geom_POINTS = shader_compile(soup_source.geom_POINTS, GL_GEOMETRY_SHADER);
    uint geom_LINES = shader_compile(soup_source.geom_LINES, GL_GEOMETRY_SHADER);
    uint geom_TRI_MESH = shader_compile(soup_source.geom_TRI_MESH, GL_GEOMETRY_SHADER);
    uint frag_POINTS = shader_compile(soup_source.frag_POINTS, GL_FRAGMENT_SHADER);
    uint frag_LINES = shader_compile(soup_source.frag_LINES, GL_FRAGMENT_SHADER);
    uint frag_TRIANGLES = shader_compile(soup_source.frag_TRIANGLES, GL_FRAGMENT_SHADER);
    uint frag_TRI_MESH = shader_compile(soup_source.frag_TRI_MESH, GL_FRAGMENT_SHADER);
    soup.shader_program_POINTS = shader_build_program(vert, geom_POINTS, frag_POINTS);
    soup.shader_program_LINES = shader_build_program(vert, geom_LINES, frag_LINES);
    soup.shader_program_TRIANGLES = shader_build_program(vert, 0, frag_TRIANGLES);
    soup.shader_program_TRI_MESH = shader_build_program(vert, geom_TRI_MESH, frag_TRI_MESH);
    glGenVertexArrays(ARRAY_LENGTH(soup.VAO), soup.VAO);
    glGenBuffers(ARRAY_LENGTH(soup.VBO), soup.VBO);
    glGenBuffers(ARRAY_LENGTH(soup.EBO), soup.EBO);
};

void soup_draw(
        mat4 PV,
        mat4 M,
        uint SOUP_primitive,
        uint num_vertices,
        vec3 *vertex_positions,
        vec4 *vertex_colors,
        real *vertex_sizes,
        bool force_draw_on_top,
        bool stipple,
        bool clip,
        vec4 clip_plane) {
    if (num_vertices == 0) { return; } // NOTE: num_vertices zero is valid input

    mat4 transform = PV * M; // FORNOW

    glBindVertexArray(soup.VAO[0]);
    uint attrib_index = 0;
    auto upload_vertex_attribute = [&](void *array, uint count, uint dim) {
        ASSERT(array);
        ASSERT(attrib_index <= ARRAY_LENGTH(soup.VBO));
        glDisableVertexAttribArray(attrib_index); {
            uint buffer_size = count * dim * sizeof(real);
            glBindBuffer(GL_ARRAY_BUFFER, soup.VBO[attrib_index]);
            glBufferData(GL_ARRAY_BUFFER, buffer_size, array, GL_DYNAMIC_DRAW);
            glVertexAttribPointer(attrib_index, dim, GL_FLOAT, GL_FALSE, 0, NULL);
        } glEnableVertexAttribArray(attrib_index);
        ++attrib_index;
    };
    upload_vertex_attribute(vertex_positions, num_vertices, 3);
    upload_vertex_attribute(vertex_colors, num_vertices, 4);
    upload_vertex_attribute(vertex_sizes, num_vertices, 1);

    uint GL_primitive;
    uint shader_program_ID;
    {
        if (SOUP_primitive == SOUP_POINTS) {
            GL_primitive = GL_POINTS;
            shader_program_ID = soup.shader_program_POINTS;
        } else if (SOUP_primitive == SOUP_LINES) {
            GL_primitive = GL_LINES;
            shader_program_ID = soup.shader_program_LINES;
        } else if (SOUP_primitive == SOUP_LINE_STRIP) {
            GL_primitive = GL_LINE_STRIP;
            shader_program_ID = soup.shader_program_LINES;
        } else if (SOUP_primitive == SOUP_LINE_LOOP) {
            GL_primitive = GL_LINE_LOOP;
            shader_program_ID = soup.shader_program_LINES;
        } else if (SOUP_primitive == SOUP_TRIANGLES) {
            GL_primitive = GL_TRIANGLES;
            shader_program_ID = soup.shader_program_TRIANGLES;
        } else if (SOUP_primitive == SOUP_TRIANGLE_FAN) {
            GL_primitive = GL_TRIANGLE_FAN;
            shader_program_ID = soup.shader_program_TRIANGLES;
        } else if (SOUP_primitive == SOUP_TRIANGLE_STRIP) {
            GL_primitive = GL_TRIANGLE_STRIP;
            shader_program_ID = soup.shader_program_TRIANGLES;
        } else if (SOUP_primitive == SOUP_QUADS) {
            GL_primitive = GL_TRIANGLES;
            shader_program_ID = soup.shader_program_TRIANGLES;
        } else { ASSERT(SOUP_primitive == SOUP_TRI_MESH);
            GL_primitive = SOUP_TRIANGLES;
            shader_program_ID = soup.shader_program_TRI_MESH;
        }
    }
    ASSERT(shader_program_ID);
    glUseProgram(shader_program_ID);

    auto LOC = [&](char *name) { return glGetUniformLocation(shader_program_ID, name); };
    vec2 OpenGL_from_Pixel_scale = (2.0f / window_get_size_Pixel());

    glUniform1ui(LOC("stipple"), stipple);
    glUniform1f(LOC("t_stipple"), global_time);
    glUniform1ui(LOC("force_draw_on_top"), force_draw_on_top);
    glUniform2f(LOC("OpenGL_from_Pixel_scale"), OpenGL_from_Pixel_scale.x, OpenGL_from_Pixel_scale.y);
    glUniformMatrix4fv(LOC("transform"), 1, GL_TRUE, transform.data);
    glUniformMatrix4fv(LOC("M"), 1, GL_TRUE, M.data); // FORNOW

    if (clip) { glEnable(GL_CLIP_DISTANCE0); } // FORNOW

    glUniform1ui(LOC("clip"), clip);
    glUniform4fv(LOC("clip_plane"), 1, &clip_plane[0]);


    if (SOUP_primitive != SOUP_QUADS) {
        glDrawArrays(GL_primitive, 0, num_vertices);
    } else { ASSERT(SOUP_primitive == SOUP_QUADS);
        const int MAX_VERTICES = 1000000;
        ASSERT(num_vertices <= MAX_VERTICES);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, soup.EBO[0]);
        {
            GL_primitive = GL_TRIANGLES;
            num_vertices = (num_vertices / 4) * 6;
            static GLuint *indices;
            if (!indices) {
                indices = (GLuint *) malloc(MAX_VERTICES / 4 * 6 * sizeof(GLuint));
                int k = 0;
                for (int i = 0; i < MAX_VERTICES / 4; ++i) {
                    indices[k++] = 4 * i + 2;
                    indices[k++] = 4 * i + 1;
                    indices[k++] = 4 * i + 0;
                    indices[k++] = 4 * i + 3;
                    indices[k++] = 4 * i + 2;
                    indices[k++] = 4 * i + 0;
                }
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, MAX_VERTICES / 4 * 6 * sizeof(GLuint), indices, GL_STATIC_DRAW);
            }
        }
        glDrawElements(GL_primitive, num_vertices, GL_UNSIGNED_INT, NULL);
    }

    glDisable(GL_CLIP_DISTANCE0); // FORNOW
}


// <!> End soup.cpp <!>
// <!> Begin eso.cpp <!> 
////////////////////////////////////////////////////////////////////////////////
// eso /////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define ESO_MAX_VERTICES 9999999

struct {
    bool _called_eso_begin_before_calling_eso_vertex_or_eso_end;

    vec4 current_color;
    real current_size;

    bool overlay;
    bool stipple;

    bool clip;
    vec4 clip_plane; // FORNOW: the caller is responsible for transformting this (will use M when we get back to chowder? maybe? -- TODOLATER; wait for more use cases

    mat4 PV;
    mat4 M;
    uint primitive;

    uint num_vertices;

    vec3 vertex_positions[ESO_MAX_VERTICES];
    vec4 vertex_colors[ESO_MAX_VERTICES];
    real vertex_sizes[ESO_MAX_VERTICES];
} eso;

void eso_begin(mat4 PV, uint primitive) {
    ASSERT(!eso._called_eso_begin_before_calling_eso_vertex_or_eso_end);
    eso._called_eso_begin_before_calling_eso_vertex_or_eso_end = true;

    eso.current_color = V4(basic.magenta, 1.0f);
    eso.current_size = 1.5f;

    eso.overlay = false;
    eso.stipple = false;

    eso.clip = false;
    eso.clip_plane = {};

    eso.PV = PV;
    eso.M = M4_Identity();
    eso.primitive = primitive;

    eso.num_vertices = 0;
}

void eso_begin(mat4 PV, mat4 M, uint primitive) {
    eso_begin(PV, primitive);
    eso.M = M;
}

void eso_end() {
    ASSERT(eso._called_eso_begin_before_calling_eso_vertex_or_eso_end);
    eso._called_eso_begin_before_calling_eso_vertex_or_eso_end = false;
    soup_draw(
            eso.PV,
            eso.M,
            eso.primitive,
            eso.num_vertices,
            eso.vertex_positions,
            eso.vertex_colors,
            eso.vertex_sizes,
            eso.overlay,
            eso.stipple,
            eso.clip,
            eso.clip_plane);
}

void eso_clip(bool clip) {
    eso.clip = clip;
}

void eso_clip_plane(real a, real b, real c, real d) {
    eso.clip_plane = { a, b, c, d };
}

void eso_overlay(bool overlay) {
    eso.overlay = overlay;
}

void eso_stipple(bool stipple) {
    eso.stipple = stipple;
}

void eso_size(real size) {
    eso.current_size = size;
}

void eso_color(real red, real green, real blue, real alpha) {
    eso.current_color[0] = red;
    eso.current_color[1] = green;
    eso.current_color[2] = blue;
    eso.current_color[3] = alpha;
}

void eso_color(real red, real green, real blue) {
    eso_color(red, green, blue, 1.0f);
}

void eso_color(vec3 rgb) {
    eso_color(rgb[0], rgb[1], rgb[2], 1.0f);
}

void eso_color(vec3 rgb, real alpha) {
    eso_color(rgb[0], rgb[1], rgb[2], alpha);
}

void eso_color(vec4 rgba) {
    eso_color(rgba[0], rgba[1], rgba[2], rgba[3]);
}

void eso_vertex(real x, real y, real z) {
    ASSERT(eso._called_eso_begin_before_calling_eso_vertex_or_eso_end);
    ASSERT(eso.num_vertices < ESO_MAX_VERTICES);
    eso.vertex_positions[eso.num_vertices] = { x, y, z };
    eso.vertex_colors[eso.num_vertices] = eso.current_color;
    eso.vertex_sizes[eso.num_vertices] = eso.current_size;
    ++eso.num_vertices;
}

void eso_vertex(real x, real y) {
    eso_vertex(x, y, 0.0f);
}


void eso_vertex(vec2 xy) {
    eso_vertex(xy[0], xy[1]);
}

void eso_vertex(vec3 xyz) {
    eso_vertex(xyz[0], xyz[1], xyz[2]);
}


// <!> End eso.cpp <!>
// <!> Begin text.cpp <!> 
// NOTE: this is a (slightly) modified version of stb_easy_font with a wrapper

static struct stb_easy_font_info_struct {
    unsigned char advance;
    unsigned char h_seg;
    unsigned char v_seg;
} stb_easy_font_charinfo[96] = {
    {  6,  0,  0 },  {  3,  0,  0 },  {  5,  1,  1 },  {  7,  1,  4 },
    {  7,  3,  7 },  {  7,  6, 12 },  {  7,  8, 19 },  {  4, 16, 21 },
    {  4, 17, 22 },  {  4, 19, 23 },  { 23, 21, 24 },  { 23, 22, 31 },
    { 20, 23, 34 },  { 22, 23, 36 },  { 19, 24, 36 },  { 21, 25, 36 },
    {  6, 25, 39 },  {  6, 27, 43 },  {  6, 28, 45 },  {  6, 30, 49 },
    {  6, 33, 53 },  {  6, 34, 57 },  {  6, 40, 58 },  {  6, 46, 59 },
    {  6, 47, 62 },  {  6, 55, 64 },  { 19, 57, 68 },  { 20, 59, 68 },
    { 21, 61, 69 },  { 22, 66, 69 },  { 21, 68, 69 },  {  7, 73, 69 },
    {  9, 75, 74 },  {  6, 78, 81 },  {  6, 80, 85 },  {  6, 83, 90 },
    {  6, 85, 91 },  {  6, 87, 95 },  {  6, 90, 96 },  {  7, 92, 97 },
    {  6, 96,102 },  {  5, 97,106 },  {  6, 99,107 },  {  6,100,110 },
    {  6,100,115 },  {  7,101,116 },  {  6,101,121 },  {  6,101,125 },
    {  6,102,129 },  {  7,103,133 },  {  6,104,140 },  {  6,105,145 },
    {  7,107,149 },  {  6,108,151 },  {  7,109,155 },  {  7,109,160 },
    {  7,109,165 },  {  7,118,167 },  {  6,118,172 },  {  4,120,176 },
    {  6,122,177 },  {  4,122,181 },  { 23,124,182 },  { 22,129,182 },
    {  4,130,182 },  { 22,131,183 },  {  6,133,187 },  { 22,135,191 },
    {  6,137,192 },  { 22,139,196 },  {  6,144,197 },  { 22,147,198 },
    {  6,150,202 },  { 19,151,206 },  { 21,152,207 },  {  6,155,209 },
    {  3,160,210 },  { 23,160,211 },  { 22,164,216 },  { 22,165,220 },
    { 22,167,224 },  { 22,169,228 },  { 21,171,232 },  { 21,173,233 },
    {  5,178,233 },  { 22,179,234 },  { 23,180,238 },  { 23,180,243 },
    { 23,180,248 },  { 22,189,248 },  { 22,191,252 },  {  5,196,252 },
    {  3,203,252 },  {  5,203,253 },  { 22,210,253 },  {  0,214,253 },
};

static unsigned char stb_easy_font_hseg[214] = {
    97,37,69,84,28,51,2,18,10,49,98,41,65,25,81,105,33,9,97,1,97,37,37,36,81,10,98,107,3,100,3,99,58,51,4,99,58,8,73,81,10,50,98,8,73,81,4,10,50,98,8,25,33,65,81,10,50,17,65,97,25,33,25,49,9,65,20,68,1,65,25,49,41,11,105,13,101,76,10,50,10,50,98,11,99,10,98,11,50,99,11,50,11,99,8,57,58,3,99,99,107,10,10,11,10,99,11,5,100,41,65,57,41,65,9,17,81,97,3,107,9,97,1,97,33,25,9,25,41,100,41,26,82,42,98,27,83,42,98,26,51,82,8,41, 35,8,10,26,82,114,42,1,114,8,9,73,57,81,41,97,18,8,8,25,26,26,82,26,82,26,82,41,25,33,82,26,49,73,35,90,17,81,41,65,57,41,65,25,81,90,114,20,84,73,57,41,49,25,33,65,81,9,97,1,97,25,33,65,81,57,33,25,41,25,
};

static unsigned char stb_easy_font_vseg[253] = {
    4,2,8,10,15,8,15,33,8,15,8,73,82,73,57,41,82,10,82,18,66,10,21,29,1,65, 27,8,27,9,65,8,10,50,97,74,66,42,10,21,57,41,29,25,14,81,73,57,26,8,8, 26,66,3,8,8,15,19,21,90,58,26,18,66,18,105,89,28,74,17,8,73,57,26,21, 8,42,41,42,8,28,22,8,8,30,7,8,8,26,66,21,7,8,8,29,7,7,21,8,8,8,59,7,8, 8,15,29,8,8,14,7,57,43,10,82,7,7,25,42,25,15,7,25,41,15,21,105,105,29, 7,57,57,26,21,105,73,97,89,28,97,7,57,58,26,82,18,57,57,74,8,30,6,8,8, 14,3,58,90,58,11,7,74,43,74,15,2,82,2,42,75,42,10,67,57,41,10,7,2,42, 74,106,15,2,35,8,8,29,7,8,8,59,35,51,8,8,15,35,30,35,8,8,30,7,8,8,60, 36,8,45,7,7,36,8,43,8,44,21,8,8,44,35,8,8,43,23,8,8,43,35,8,8,31,21,15, 20,8,8,28,18,58,89,58,26,21,89,73,89,29,20,8,8,30,7,
};

typedef struct {
    unsigned char c[4];
} stb_easy_font_color;

static int stb_easy_font_draw_segs(float x, float y, unsigned char *segs, int num_segs, int vertical, stb_easy_font_color c, char *vbuf, int vbuf_size, int offset) {
    int i,j;
    for (i=0; i < num_segs; ++i) {
        int len = segs[i] & 7;
        x += (float) ((segs[i] >> 3) & 1);
        if (len && offset+64 <= vbuf_size) {
            float y0 = y + (float) (segs[i]>>4);
            for (j=0; j < 4; ++j) {
                * (float *) (vbuf+offset+0) = x  + (j==1 || j==2 ? (vertical ? 1 : len) : 0);
                * (float *) (vbuf+offset+4) = y0 + (    j >= 2   ? (vertical ? len : 1) : 0);
                * (float *) (vbuf+offset+8) = 0.f;
                * (stb_easy_font_color *) (vbuf+offset+12) = c;
                offset += 16;
            }
        }
    }
    return offset;
}

static float stb_easy_font_spacing_val = 0;

// static void stb_easy_font_spacing(float spacing) {
//    stb_easy_font_spacing_val = spacing;
// }

static int stb_easy_font_print(float x, float y, String string, unsigned char color[4], void *vertex_buffer, int vbuf_size) {
    char *vbuf = (char *) vertex_buffer;
    float start_x = x;
    int offset = 0;

    stb_easy_font_color c = { 255,255,255,255 }; // use structure copying to avoid needing depending on memcpy()
    if (color) { c.c[0] = color[0]; c.c[1] = color[1]; c.c[2] = color[2]; c.c[3] = color[3]; }

    char *text = string.data;
    while (string_pointer_is_valid(string, text) && (offset < vbuf_size)) {
        if (*text == '\n') {
            y += 12;
            x = start_x;
        } else {
            unsigned char advance = stb_easy_font_charinfo[*text-32].advance;
            float y_ch = advance & 16 ? y+1 : y;
            int h_seg, v_seg, num_h, num_v;
            h_seg = stb_easy_font_charinfo[*text-32  ].h_seg;
            v_seg = stb_easy_font_charinfo[*text-32  ].v_seg;
            num_h = stb_easy_font_charinfo[*text-32+1].h_seg - h_seg;
            num_v = stb_easy_font_charinfo[*text-32+1].v_seg - v_seg;
            offset = stb_easy_font_draw_segs(x, y_ch, &stb_easy_font_hseg[h_seg], num_h, 0, c, vbuf, vbuf_size, offset);
            offset = stb_easy_font_draw_segs(x, y_ch, &stb_easy_font_vseg[v_seg], num_v, 1, c, vbuf, vbuf_size, offset);
            x += advance & 15;
            x += stb_easy_font_spacing_val;
        }
        ++text;
    }
    return (unsigned) offset/64;
}

static int stb_easy_font_travel_x(String string) {
    float len = 0;
    float max_len = 0;
    char *text = string.data;
    while (string_pointer_is_valid(string, text)) {
        if (*text == '\n') {
            if (len > max_len) max_len = len;
            len = 0;
        } else {
            len += stb_easy_font_charinfo[*text-32].advance & 15;
            len += stb_easy_font_spacing_val;
        }
        ++text;
    }
    if (len > max_len) max_len = len;
    return (int) ceil(max_len);
}

static int stb_easy_font_travel_y(String string) {
    int count = 0;
    char *text = string.data;
    while (string_pointer_is_valid(string, text)) {
        if (*text == '\n') ++count;
        ++text;
    }
    return count * 12;
}

static vec2 stb_easy_font_travel(String string) {
    return V2(real(stb_easy_font_travel_x(string)), real(stb_easy_font_travel_y(string)));
}

////////////////////////////////////////
// text_draw ///////////////////////////
////////////////////////////////////////

vec2 text_travel(String string, real font_height_Pixel) {
    return (font_height_Pixel / 12.0f) * stb_easy_font_travel(string);
}

// TODO: consider text_drawf
template <uint D_position, uint D_color> vec2 text_draw(
        mat4 PV,
        String string,
        Vector<D_position> position_World,
        Vector<D_color> color,
        real font_height_Pixel = 12.0f,
        vec2 nudge_Pixel = {},
        bool overlay = true
        ) {
    STATIC_ASSERT((D_position == 2) || (D_position == 3));
    STATIC_ASSERT((D_color == 3) || (D_color == 4));

    vec2 *vertex_positions;
    uint num_vertices;
    {
        uint size = 99999 * sizeof(float);
        static void *_vertex_positions = malloc(size);
        vertex_positions = (vec2 *) _vertex_positions;

        num_vertices = 4 * stb_easy_font_print(0, 0, string, NULL, _vertex_positions, size);
        { // NOTE: stb stores like this [x:float y:float z:float color:uint8[4]]
            for_(i, num_vertices) {
                ((vec2 *) vertex_positions)[i] = {
                    ((float *) vertex_positions)[4 * i + 0],
                    ((float *) vertex_positions)[4 * i + 1],
                };
            }
        }
    }

    vec3 position_World3; {
        position_World3.z = 0;
        for_(d, D_position) position_World3[d] = position_World[d];
    }
    vec3 position_OpenGL = transformPoint(PV, position_World3);

    if (!IS_BETWEEN_LOOSE(position_OpenGL.z, -1.0f, 1.0f)) return {};

    vec2 position_Pixel = transformPoint(inverse(window_get_NDC_from_Pixel()), _V2(position_OpenGL));

    mat4 transform = window_get_NDC_from_Pixel()
        * M4_Translation(position_Pixel + nudge_Pixel)
        * M4_Scaling(font_height_Pixel / 12.0f);
    eso_begin(transform, SOUP_QUADS);
    eso_overlay(overlay);
    eso_color(color);
    for_(i, num_vertices) eso_vertex(vertex_positions[i]);
    eso_end();

    return text_travel(string, font_height_Pixel);
}

////////////////////////////////////////
// easy_text_draw ///////////////////////////
////////////////////////////////////////

struct EasyTextPen {
    vec2 origin;
    real font_height_Pixel;
    vec3 color;
    bool automatically_append_newline;

    real one_minus_alpha;
    vec2 offset_Pixel;

    bool ghost_write; // does all the math and updates, just doesn't draw

    vec2 get_position_Pixel() {
        return this->origin + this->offset_Pixel;
    }
    real get_x_Pixel() { return this->origin.x + this->offset_Pixel.x; }
    real get_y_Pixel() { return this->origin.y + this->offset_Pixel.y; }
};

void easy_text_draw(EasyTextPen *pen, String string) {
    vec2 travel;
    if (!pen->ghost_write) {
        travel = text_draw(window_get_NDC_from_Pixel(), string, pen->get_position_Pixel(), V4(pen->color, 1.0f - pen->one_minus_alpha), pen->font_height_Pixel);
    } else {
        travel = text_travel(string, pen->font_height_Pixel);
    }

    if (IS_ZERO(travel.y) && (!pen->automatically_append_newline)) {
        pen->offset_Pixel.x += travel.x;
    } else {
        pen->offset_Pixel.x = 0.0f;
        pen->offset_Pixel.y += travel.y;
        if (pen->automatically_append_newline) pen->offset_Pixel.y += pen->font_height_Pixel;
    }
}

void easy_text_drawf(EasyTextPen *pen, const char *format, ...) {
    #define EASY_TEXT_MAX_LENGTH 4096
    static _STRING_CALLOC(string, EASY_TEXT_MAX_LENGTH); {
        va_list arg;
        va_start(arg, format);
        string.length = vsnprintf(string.data, EASY_TEXT_MAX_LENGTH, format, arg);
        va_end(arg);
    }
    easy_text_draw(pen, string);
}

real _easy_text_dx(EasyTextPen *pen, String string) {
    return text_travel(string, pen->font_height_Pixel).x;
}

real _easy_text_dx(EasyTextPen *pen, char *cstring) {
    return _easy_text_dx(pen, STRING(cstring));
}
// <!> End text.cpp <!>
// #include "gui.cpp"

#define sin use_SIN_instead_of_sin
#define cos use_COS_instead_of_cos
#define tan use_TAN_instead_of_tan
#define atan2 use_ATAN2_instead_of_atan2
#define pow use_POW_instead_of_pow
#define sqrt use_SQRT_instead_of_sqrt
#define atan use_atan2_instead_of_atan

// <!> End playground.cpp <!>

mat4 NDC_from_Pixel;

bool mouse_left_pressed;
bool mouse_left_held;
bool mouse_left_released;

bool mouse_right_pressed;
bool mouse_right_held;
bool mouse_right_released;

vec2 mouse_position_Pixel;
vec2 mouse_position_NDC;

real _accumulator_mouse_wheel_offset;
vec2 _accumulator_mouse_change_in_position_Pixel;
vec2 _accumulator_mouse_change_in_position_NDC;

vec2 get_mouse_position_World(mat4 PV) {
    return transformPoint(inverse(PV), mouse_position_NDC);
}

vec2 _get_mouse_change_in_position_World(mat4 PV) {
    return transformVector(inverse(PV), _accumulator_mouse_change_in_position_NDC);
}

#define MAX_NUM_KEYS 512
bool key_pressed[MAX_NUM_KEYS];
bool key_held[MAX_NUM_KEYS];
bool key_released[MAX_NUM_KEYS];
bool key_toggled[MAX_NUM_KEYS];


void easy_callback_key(GLFWwindow *, int key, int, int action, int mods) {
    FORNOW_UNUSED(mods);
    if (key < 0) return;
    if (key >= 512) return;
    if (action == GLFW_PRESS) {
        key_pressed[key] = true;
        key_held[key] = true;
        key_toggled[key] = !key_toggled[key];
    } else if (action == GLFW_RELEASE) {
        key_released[key] = true;
        key_held[key] = false;
    }
}

void easy_callback_cursor_position(GLFWwindow *, double _xpos, double _ypos) {
    vec2 prev_mouse_position_Pixel = mouse_position_Pixel;
    vec2 prev_mouse_position_NDC = mouse_position_NDC;

    mouse_position_Pixel = { real(_xpos), real(_ypos) };
    mouse_position_NDC = transformPoint(NDC_from_Pixel, mouse_position_Pixel);

    _accumulator_mouse_change_in_position_Pixel += (mouse_position_Pixel - prev_mouse_position_Pixel);
    _accumulator_mouse_change_in_position_NDC += (mouse_position_NDC - prev_mouse_position_NDC);
}

void easy_callback_mouse_button(GLFWwindow *, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) { 
            mouse_left_pressed = true;
            mouse_left_held = true;
        } else if (action == GLFW_RELEASE) { 
            mouse_left_released = true;
            mouse_left_held = false;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) { 
            mouse_right_pressed = true;
            mouse_right_held = true;
        } else if (action == GLFW_RELEASE) { 
            mouse_right_released = true;
            mouse_right_held = false;
        }
    }
}

void easy_callback_scroll(GLFWwindow *, double, double yoffset) {
    _accumulator_mouse_wheel_offset += real(yoffset);
}

void easy_callback_framebuffer_size(GLFWwindow *, int width, int height) {
    glViewport(0, 0, width, height);
}
run_before_main {
    glfwSetFramebufferSizeCallback(glfw_window, easy_callback_framebuffer_size);
    glfwSetKeyCallback(glfw_window, easy_callback_key);
    glfwSetCursorPosCallback(glfw_window, easy_callback_cursor_position);
    glfwSetMouseButtonCallback(glfw_window, easy_callback_mouse_button);
    glfwSetScrollCallback(glfw_window, easy_callback_scroll);
};

void _callback_scroll_helper(Camera *camera_2D, double yoffset) {
    // IDEA: preserve mouse position
    ASSERT(IS_ZERO(camera_2D->angle_of_view));
    ASSERT(IS_ZERO(camera_2D->euler_angles));
    vec2 mouse_position_World_before  = transformPoint(inverse(camera_2D->get_PV()), mouse_position_NDC);
    camera_2D->ortho_screen_height_World *= (1.0f - 0.1f * real(yoffset));
    vec2 mouse_position_World_after = transformPoint(inverse(camera_2D->get_PV()), mouse_position_NDC);
    camera_2D->pre_nudge_World -= (mouse_position_World_after - mouse_position_World_before);
}

// TODO: include WASD for FirstPersonCamera3D
void _camera_easy_move(Camera *camera) {
    if (camera->type == CameraType::Camera2D) {
        if (mouse_right_held) {
            camera->pre_nudge_World -= _get_mouse_change_in_position_World(camera->get_PV());
        }
        if (!IS_ZERO(_accumulator_mouse_wheel_offset)) {
            vec2 mouse_position_World_before  = transformPoint(inverse(camera->get_PV()), mouse_position_NDC);
            camera->ortho_screen_height_World *= (1.0f - 0.1f * real(_accumulator_mouse_wheel_offset));
            vec2 mouse_position_World_after = transformPoint(inverse(camera->get_PV()), mouse_position_NDC);
            camera->pre_nudge_World -= (mouse_position_World_after - mouse_position_World_before);
        }
    } else if (camera->type == CameraType::OrbitCamera3D) {
        if (mouse_left_held) {
            real fac = 2.0f;
            camera->euler_angles.y -= fac * _accumulator_mouse_change_in_position_NDC.x;
            camera->euler_angles.x += fac * _accumulator_mouse_change_in_position_NDC.y;
            camera->euler_angles.x = CLAMP(camera->euler_angles.x, -RAD(90), RAD(90));
        }
        if (mouse_right_held) {
            Camera tmp_2D = make_EquivalentCamera2D(camera);
            tmp_2D.pre_nudge_World -= transformVector(inverse(tmp_2D.get_PV()), _accumulator_mouse_change_in_position_NDC);
            camera->pre_nudge_World = tmp_2D.pre_nudge_World;
        }
        if (!IS_ZERO(_accumulator_mouse_wheel_offset)) {
            bool is_perspective_camera = (!IS_ZERO(camera->angle_of_view));
            Camera tmp_2D = make_EquivalentCamera2D(camera);
            _callback_scroll_helper(&tmp_2D, _accumulator_mouse_wheel_offset);
            if (is_perspective_camera) {
                camera->persp_distance_to_origin_World = ((0.5f * tmp_2D.ortho_screen_height_World) / TAN(0.5f * camera->angle_of_view));
            } else {
                camera->ortho_screen_height_World = tmp_2D.ortho_screen_height_World;
            }
            camera->pre_nudge_World = tmp_2D.pre_nudge_World;
        }
    } else {  ASSERT(camera->type == CameraType::FirstPersonCamera3D);
        { // WASD
          // FORNOW
            real fac = 1.0f;
            vec2 _forward = rotated({ 0.0f, -1.0f}, -camera->euler_angles.y);
            vec2 _perp = { -_forward.y, _forward.x };
            vec3 forward = { _forward.x, 0.0f, _forward.y };
            vec3 perp = { _perp.x, 0.0f, _perp.y };
            if (key_held['W']) camera->first_person_position_World += fac * forward;
            if (key_held['A']) camera->first_person_position_World -= fac * perp;
            if (key_held['S']) camera->first_person_position_World -= fac * forward;
            if (key_held['D']) camera->first_person_position_World += fac * perp;
        }
        {
            real fac = 2.0f;
            camera->euler_angles.y -= fac * _accumulator_mouse_change_in_position_NDC.x;
            camera->euler_angles.x += fac * _accumulator_mouse_change_in_position_NDC.y;
            camera->euler_angles.x = CLAMP(camera->euler_angles.x, -RAD(90), RAD(90));
        }
    }
}

void _spoof_easy_callback_cursor_position() {
    {
        double xpos, ypos;
        glfwGetCursorPos(glfw_window, &xpos, &ypos);
        easy_callback_cursor_position(glfw_window, xpos, ypos);
    }
}

// NOTE: patch first frame mouse position issue
run_before_main {
    NDC_from_Pixel = window_get_NDC_from_Pixel();
    _spoof_easy_callback_cursor_position();
};

void pointer_lock() {
    glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void pointer_unlock() {
    glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    _spoof_easy_callback_cursor_position();
}

bool _initialized;
bool begin_frame(Camera *camera) {
    { // clear input before polling
        {
            _accumulator_mouse_wheel_offset = 0.0f;
            _accumulator_mouse_change_in_position_Pixel = {};
            _accumulator_mouse_change_in_position_NDC = {};
        }
        {
            memset(key_pressed, 0, sizeof(key_pressed));
            memset(key_released, 0, sizeof(key_released));
        }
    }

    { // gl
        glfwPollEvents();
        glfwSwapBuffers(glfw_window);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    }

    NDC_from_Pixel = window_get_NDC_from_Pixel();

    if (camera) _camera_easy_move(camera);
    // _gui_begin_frame();

    return (!glfwWindowShouldClose(glfw_window));
}

