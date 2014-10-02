/* Copyright (c) 2009-2013,  NVIDIA CORPORATION

   All rights reserved.
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the distribution.
   Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used to endorse
   or promote products derived from this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/   
 
#include <math.h>
#ifndef _CUSPARSE_OPS_HXX_
#define _CUSPARSE_OPS_HXX_
#define pi 3.1415926535897931

#if !defined(__cplusplus)
#error "This file can only by include in C++ file because it overload function names"
#endif

/* Multiplication */
static __inline__ __device__ __host__   float cuMul( float x , float y )
{
    return(x * y);
}

static __inline__ __device__ __host__  double cuMul( double x , double y )
{
    return(x * y);
}

static __inline__ __device__ __host__  cuComplex cuMul( cuComplex x , cuComplex y )
{
    return( cuCmulf(x , y) );
}

static __inline__ __device__ __host__  cuDoubleComplex cuMul( cuDoubleComplex x , cuDoubleComplex y )
{
    return( cuCmul(x , y));
}  

/* Negation */
static __inline__ __device__ __host__  float cuNeg( float x )
{
    return(-x);
}

static __inline__ __device__ __host__  double cuNeg( double x )
{
    return(-x);
}

static __inline__ __device__ __host__  cuComplex cuNeg( cuComplex x )
{
    return( make_cuComplex( -cuCrealf( x ), -cuCimagf( x ) ));
}

static __inline__ __device__ __host__  cuDoubleComplex cuNeg( cuDoubleComplex x )
{
    return( make_cuDoubleComplex( -cuCreal( x ), -cuCimag( x ) ));
}  


/* Addition */
static __inline__ __device__ __host__  float cuAdd( float x , float y )
{
    return(x + y);
}

static __inline__ __device__ __host__  double cuAdd( double x , double y )
{
    return(x + y);
}

static __inline__ __device__ __host__  cuComplex cuAdd( cuComplex x , cuComplex y )
{
    return( cuCaddf(x , y) );
}

static __inline__ __device__ __host__  cuDoubleComplex cuAdd( cuDoubleComplex x , cuDoubleComplex y )
{
    return( cuCadd(x , y));
}  


/* Subtraction */
static __inline__ __device__ __host__  float cuSub( float x , float y )
{
    return(x - y);
}

static __inline__ __device__ __host__  double cuSub( double x , double y )
{
    return(x - y);
}

static __inline__ __device__ __host__  cuComplex cuSub( cuComplex x , cuComplex y )
{
    return( cuCsubf(x , y) );
}

static __inline__ __device__ __host__  cuDoubleComplex cuSub( cuDoubleComplex x , cuDoubleComplex y )
{
    return( cuCsub(x , y));
}  
/* Division */
static __inline__ __device__ __host__  float cuDiv( float x , float y )
{
    return (x / y);
}

static __inline__ __device__ __host__  double cuDiv( double x , double y )
{
    return (x / y);
}

static __inline__ __device__ __host__  cuComplex cuDiv( cuComplex x , cuComplex y )
{
    return( cuCdivf( x, y ) );
}

static __inline__ __device__ __host__  cuDoubleComplex cuDiv( cuDoubleComplex x , cuDoubleComplex y )
{
    return( cuCdiv( x, y ) );
}  

/* Fma */
static __inline__ __device__ __host__  float cuFma( float x , float y, float d )
{
    return ((x * y) + d);
}

static __inline__ __device__ __host__  double cuFma( double x , double y, double d )
{
    return ((x * y) + d);
}

static __inline__ __device__ __host__  cuComplex cuFma( cuComplex x , cuComplex y, cuComplex d )
{
    return(cuCfmaf(x, y, d));
}

static __inline__ __device__ __host__  cuDoubleComplex cuFma( cuDoubleComplex x , cuDoubleComplex y, cuDoubleComplex d )
{
    return(cuCfma(x , y, d));
}  

/* absolute value */
static __inline__ __device__ __host__  float cuAbs( float x )
{
    return (fabsf(x));
}

static __inline__ __device__ __host__  double cuAbs( double x )
{
    return (fabs(x));
}

static __inline__ __device__ __host__  float cuAbs( cuComplex x )
{
    return(cuCabsf(x));
}

static __inline__ __device__ __host__  double cuAbs( cuDoubleComplex x )
{
    return(cuCabs(x));
}  

/*---------------------------------------------------------------------------------*/
/* Conjugate */
static __inline__ __device__ __host__  float cuConj( float x )
{
    return (x);
}

static __inline__ __device__ __host__  double cuConj( double x )
{
    return (x);
}

static __inline__ __device__ __host__  cuComplex cuConj( cuComplex x )
{
    return( cuConjf(x) );
}
/*---------------------------------------------------------------------------------*/
/* Real part , Imaginary part */
static __inline__ __device__ __host__  float cuReal( float x )
{
    return (x);
}

static __inline__ __device__ __host__  double cuReal( double x )
{
    return (x);
}

static __inline__ __device__ __host__  float cuReal( cuComplex x )
{
    return( cuCrealf(x) );
}
static __inline__ __device__ __host__  double cuReal( cuDoubleComplex x )
{
    return( cuCreal(x) );
}

static __inline__ __device__ __host__  float cuImag( float x )
{
    return (0.0f);
}

static __inline__ __device__ __host__  double cuImag( double x )
{
    return (0.0);
}

static __inline__ __device__ __host__  float cuImag( cuComplex x )
{
    return( cuCimagf(x) );
}
static __inline__ __device__ __host__  double cuImag( cuDoubleComplex x )
{
    return( cuCimag(x) );
}

/* Square root */
static __inline__ __device__ __host__  float cuSqrt( float x )
{
    return (sqrtf(x));
}

static __inline__ __device__ __host__  double cuSqrt( double x )
{
    return (sqrt(x));
}



/*---------------------------------------------------------------------------------*/
/* various __inline__ __device__  function to initialize a T_ELEM */
template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (int);
template <> __inline__ __device__ __host__  float cuGet<float >(int x)
{
    return float(x);
}

template <> __inline__ __device__ __host__  double cuGet<double>(int x)
{
    return double(x);
}

template <> __inline__ __device__ __host__   cuComplex cuGet<cuComplex>(int x)
{
    return (make_cuComplex( float(x), 0.0f ));
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(int x)
{
    return (make_cuDoubleComplex( double(x), 0.0 ));
}


template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (int , int );
template <> __inline__ __device__ __host__  float cuGet<float >(int x, int y)
{
    return float(x);
}

template <> __inline__ __device__ __host__  double cuGet<double>(int x, int y)
{
    return double(x);
}

template <> __inline__ __device__ __host__   cuComplex cuGet<cuComplex>(int x, int y)
{
    return make_cuComplex( float(x), float(y) );
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(int x, int y)
{
    return (make_cuDoubleComplex( double(x), double(y) ));
}



template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (float );
template <> __inline__ __device__ __host__  float cuGet<float >(float x)
{
    return float(x);
}

template <> __inline__ __device__ __host__  double cuGet<double>(float x)
{
    return double(x);
}

template <> __inline__ __device__ __host__   cuComplex cuGet<cuComplex>(float x)
{
    return (make_cuComplex( float(x), 0.0f ));
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(float x)
{
    return (make_cuDoubleComplex( double(x), 0.0 ));
}


template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (float, float );
template <> __inline__ __device__ __host__  float cuGet<float >(float x, float y)
{
    return float(x);
}

template <> __inline__ __device__ __host__  double cuGet<double>(float x, float y)
{
    return double(x);
}

template <> __inline__ __device__ __host__   cuComplex cuGet<cuComplex>(float x, float y)
{
    return (make_cuComplex( float(x), float(y) ));
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(float x, float y)
{
    return (make_cuDoubleComplex( double(x), double(y) ));
}


template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (double );
template <> __inline__ __device__ __host__  float cuGet<float >(double x)
{
    return float(x);
}

template <> __inline__ __device__ __host__  double cuGet<double>(double x)
{
    return double(x);
}

template <> __inline__ __device__ __host__   cuComplex cuGet<cuComplex>(double x)
{
    return (make_cuComplex( float(x), 0.0f ));
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(double x)
{
    return (make_cuDoubleComplex( double(x), 0.0 ));
}

template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (double, double );
template <> __inline__ __device__ __host__  float cuGet<float >(double x, double y)
{
    return float(x);
}

template <> __inline__ __device__ __host__  double cuGet<double>(double x, double y)
{
    return double(x);
}

template <> __inline__ __device__ __host__   cuComplex cuGet<cuComplex>(double x, double y)
{
    return (make_cuComplex( float(x), float(y) ));
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(double x, double y)
{
    return (make_cuDoubleComplex( double(x), double(y) ));
}

//included lines 377:387
template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (cuDoubleComplex);
template <> __inline__ __device__ __host__  cuComplex cuGet<cuComplex>(cuDoubleComplex x)
{
    return (make_cuComplex((float)cuCreal(x), (float)cuCimag(x)));
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(cuDoubleComplex x)
{
    return x;
}
/*---------------------------------------------------------------------------------*/
/* Equal */
static __inline__ __device__ __host__  bool cuEqual( float x, float y )
{
    return( x == y );
}

static __inline__ __device__ __host__  bool cuEqual( double x, double y )
{
    return( x == y );
}

static __inline__ __device__ __host__  bool cuEqual( cuComplex x, cuComplex y )
{
    return( (cuCrealf( x ) == cuCrealf( y )) && (cuCimagf( x ) == cuCimagf( y )) );
}

static __inline__ __device__ __host__  bool cuEqual( cuDoubleComplex x, cuDoubleComplex y )
{
    return( (cuCreal( x ) == cuCreal( y )) && (cuCimag( x ) == cuCimag( y )) );
}

/*---------------------------------------------------------------------------------*/
/* Argument */
static __inline__ __device__ __host__  float cuArg( cuComplex x )
{
    if(cuCrealf(x)>0 and cuCimagf(x)>0)
        return atan(cuCimagf(x)/cuCrealf(x));
    else if(cuCrealf(x)<0 and cuCimagf(x)>0)
        return (pi+atan(cuCimagf(x)/cuCrealf(x)));
    else if(cuCrealf(x)>0 and cuCimagf(x)<0)
        return (-pi/2-atan(cuCrealf(x)/cuCimagf(x)));
    else
        return (-pi+atan(cuCimagf(x)/cuCrealf(x)));
}

static __inline__ __device__ __host__  double cuArg( cuDoubleComplex x )
{
    if(cuCreal(x)>0 and cuCimag(x)>0)
        return atan(cuCimag(x)/cuCreal(x));
    else if(cuCreal(x)<0 and cuCimag(x)>0)
        return (pi+atan(cuCimag(x)/cuCreal(x)));
    else if(cuCreal(x)>0 and cuCimag(x)<0)
        return (-pi/2-atan(cuCreal(x)/cuCimag(x)));
    else
        return (-pi+atan(cuCimag(x)/cuCreal(x)));
}
/*---------------------------------------------------------------------------------*/
/* Logarithm */
static __inline__ __device__ __host__  cuComplex cuLog( cuComplex x )
{
    return (cuGet<cuComplex>(log(cuAbs(x)), cuArg(x)));
}

static __inline__ __device__ __host__  cuDoubleComplex cuLog( cuDoubleComplex x )
{
    return (cuGet<cuDoubleComplex>(log(cuAbs(x)), cuArg(x)));
}
/*---------------------------------------------------------------------------------*/
/* Exponential */
static __inline__ __device__ __host__  cuComplex cuExp( cuComplex x )
{
    return (cuGet<cuComplex>(exp(-cuImag(x))*cosf(cuReal(x)), exp(-cuImag(x))*sinf(cuReal(x))));
}

static __inline__ __device__ __host__  cuDoubleComplex cuExp( cuDoubleComplex x )
{
    return (cuGet<cuDoubleComplex>(exp(-cuImag(x))*cos(cuReal(x)), exp(-cuImag(x))*sin(cuReal(x))));
}

/*----------------------------------------------------------------------------------*/
/* Flops */
template <typename T_ELEM> __inline__ __device__ __host__  long long cuFmaFlops ();

template <> __inline__ __device__ __host__  long long cuFmaFlops<float>()
{
    return( 2 );
}

template <> __inline__ __device__ __host__  long long cuFmaFlops<double>()
{
    return( 2 );
}

template <> __inline__ __device__ __host__  long long cuFmaFlops<cuComplex>()
{
    return( 8 );
}

template <> __inline__ __device__ __host__  long long cuFmaFlops<cuDoubleComplex>()
{
    return( 8 );
}
//-----------------------------------------------
template <typename T_ELEM> __inline__ __device__ __host__  long long cuAddFlops ();
template <> __inline__ __device__ __host__  long long cuAddFlops<float>()
{
    return( 1 );
}

template <> __inline__ __device__ __host__  long long cuAddFlops<double>()
{
    return( 1 );
}

template <> __inline__ __device__ __host__  long long cuAddFlops<cuComplex>()
{
    return( 2 );
}

template <> __inline__ __device__ __host__  long long cuAddFlops<cuDoubleComplex>()
{
    return( 2 );
}
//-----------------------------------------------
template <typename T_ELEM> __inline__ __device__ __host__  long long cuMulFlops ();
template <> __inline__ __device__ __host__  long long cuMulFlops<float>()
{
    return( 1 );
}

template <> __inline__ __device__ __host__  long long cuMulFlops<double>()
{
    return( 1 );
}

template <> __inline__ __device__ __host__  long long cuMulFlops<cuComplex>()
{
    return( 6 );
}

template <> __inline__ __device__ __host__  long long cuMulFlops<cuDoubleComplex>()
{
    return( 6 );
}
//-----------------------------------------------
template <typename T_ELEM> __inline__ __device__ __host__  long long cuConjFlops ();
template <> __inline__ __device__ __host__  long long cuConjFlops<float>()
{
    return( 0 );
}

template <> __inline__ __device__ __host__  long long cuConjFlops<double>()
{
    return( 0 );
}

template <> __inline__ __device__ __host__  long long cuConjFlops<cuComplex>()
{
    return( 1 );
}

template <> __inline__ __device__ __host__  long long cuConjFlops<cuDoubleComplex>()
{
    return( 1 );
}

//-----------------------------------------------
template <typename T_ELEM> __inline__ __device__ __host__  long long cuDivFlops ();
template <> __inline__ __device__ __host__  long long cuDivFlops<float>()
{
    return( 2 );
}

template <> __inline__ __device__ __host__  long long cuDivFlops<double>()
{
    return( 2 );
}

template <> __inline__ __device__ __host__  long long cuDivFlops<cuComplex>()
{
    return( 19 );
}

template <> __inline__ __device__ __host__  long long cuDivFlops<cuDoubleComplex>()
{
    return( 19 );
}

//-------------------------------------------------------------------------------
template <typename T_ELEM> __inline__ __device__ __host__  bool cuIsComplexType();
template <> __inline__ __device__ __host__  bool cuIsComplexType<float>()           {return false;};
template <> __inline__ __device__ __host__  bool cuIsComplexType<double>()          {return false;};
template <> __inline__ __device__ __host__  bool cuIsComplexType<cuComplex>()       {return true;};
template <> __inline__ __device__ __host__  bool cuIsComplexType<cuDoubleComplex>() {return true;};

template <typename T_ELEM> __inline__ __device__ __host__  bool cuIsDpType();
template <> __inline__ __device__ __host__  bool cuIsDpType<float>()           {return false;};
template <> __inline__ __device__ __host__  bool cuIsDpType<double>()          {return true;};
template <> __inline__ __device__ __host__  bool cuIsDpType<cuComplex>()       {return false;};
template <> __inline__ __device__ __host__  bool cuIsDpType<cuDoubleComplex>() {return true;};

//-------------------------------------------------------------------------------------------------
// Trick to be able to use dynamic shared mem in templatized kernel 
/* structure used for declaring dynamic shared memory with templates */
template <typename T_ELEM> struct __dynamic_shmem__{
    __device__ T_ELEM * getPtr() { 
        extern __device__ void error(void);
        error();
        return NULL;
    }
}; 
/* specialization of the above structure for the desired types */
template <> struct __dynamic_shmem__<float>{
    __device__ float * getPtr() { 
        extern __shared__ float Sptr[];
        return Sptr;
    }
};
template <> struct __dynamic_shmem__<double>{
    __device__ double * getPtr() { 
        extern __shared__ double Dptr[];
        return Dptr;
    }
};

template <> struct __dynamic_shmem__<cuComplex>{
    __device__ cuComplex * getPtr() { 
        extern __shared__ cuComplex Cptr[];
        return Cptr;
    }
};
template <> struct __dynamic_shmem__<cuDoubleComplex>{
    __device__ cuDoubleComplex * getPtr() { 
        extern __shared__ cuDoubleComplex Zptr[];
        return Zptr;
    }
};

/*-------------------------------------------------------------------------------------------*/

// Trick to be able to use qualifier "volatile" in templatized kernels
/* functions used to handling volatile loads */

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR "l"
#else
#define __LDG_PTR "r"
#endif 


template <typename T_ELEM> __inline__ __device__ T_ELEM loadVolatile(T_ELEM *addr);
/* specialization of the above structure for the desired types */
template <>
__inline__ __device__ int loadVolatile<int>(int *addr){
    int val;
    volatile int *tmp = (volatile int *)addr;

    val = tmp[0];
    return val;
}
template <>
__inline__ __device__ float loadVolatile<float>(float *addr){
    float val;
    volatile float *tmp = (volatile float *)addr;
   
    val = tmp[0];
    return val;
}
template <>
__inline__ __device__ cuComplex loadVolatile<cuComplex>(cuComplex *addr){
    cuComplex val;
    volatile unsigned long long *tmp = (volatile unsigned long long *)addr;

    *((unsigned long long *)&val) = tmp[0];
    return val;
}
template <>
__inline__ __device__ double loadVolatile<double>(double *addr){
    double val;
    volatile double *tmp = (volatile double *)addr;
   
    val = tmp[0];
    return val;
}
template <>
__inline__ __device__ cuDoubleComplex loadVolatile<cuDoubleComplex>(cuDoubleComplex *addr){
    cuDoubleComplex val;
#if  __CUDA_ARCH__ < 200  
    volatile double *tmp = (volatile double *)addr;
   
    val.x = tmp[0];
    val.y = tmp[1];
#else    
    asm volatile ( "ld.volatile.v2.f64 {%0,%1},[%2];" :"=d" (val.x), "=d" (val.y) :__LDG_PTR(addr) );
#endif    
    return val;
}
/* functions used to handling volatile stores */
template <typename T_ELEM> __inline__ __device__ void  storeVolatile(T_ELEM *addr, T_ELEM val);
/* specialization of the above structure for the desired types */
template <>
__inline__ __device__ void storeVolatile<float>(float *addr, float val){
    volatile float *tmp = (volatile float *)addr;   
    tmp[0] = val;
}
template <>
__inline__ __device__ void storeVolatile<cuComplex>(cuComplex *addr, cuComplex val){
    volatile unsigned long long  *tmp = (volatile unsigned long long *)addr;   
    tmp[0] = *((unsigned long long *)(&val));
}
template <>
__inline__ __device__ void storeVolatile<double>(double *addr, double val){
    volatile double *tmp = (volatile double *)addr;   
    tmp[0] = val;
}
template <>
__inline__ __device__ void storeVolatile<cuDoubleComplex>(cuDoubleComplex *addr, cuDoubleComplex val){
#if  __CUDA_ARCH__ < 200  
    volatile double *tmp = (volatile double *)addr;   
    tmp[0] = val.x;
    tmp[1] = val.y;
#else    
    asm volatile ( "st.volatile.v2.f64 	[%0],{%1,%2};" :: __LDG_PTR(addr),"d" (val.x), "d" (val.y) );
#endif
}
#undef __LDG_PTR
#endif
