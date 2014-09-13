/*******************************************************************************************************
                              University of Illinois/NCSA Open Source License
                                 Copyright (c) 2012 University of Illinois
                                          All rights reserved.

                                        Developed by: IMPACT Group
                                          University of Illinois
                                      http://impact.crhc.illinois.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
  Neither the names of IMPACT Group, University of Illinois, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

*******************************************************************************************************/


#include <stdio.h>
#include <complex.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cusparse_ops.hxx"
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <helper_string.h>    // helper for string parsing
#include <helper_cuda.h>      // helper for cuda error checking functions
#include "datablock.h"
#include <assert.h>

#define DEBUG 0
#define PI 3.141592653589793
static double get_second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

// findBestGrid
template <typename T, typename T_REAL>
int find_marshaled_index(Datablock<T, T_REAL> *data, int m)
{
    int bx;
    int by;
    int bdx = data->blockDim.x;
    int tx;
    int ty;
    int blockIndex;
    int h_stride = data->h_stride;
    int l_stride = data->b_dim;
    int blockOffset;
    int subBlockIndex;
    int subBlockOffset;
    int mNewIndex;

    blockIndex = m/(h_stride*l_stride);
    blockOffset = m%(h_stride*l_stride);
    subBlockIndex = blockOffset/(h_stride*bdx);
    subBlockOffset = blockOffset%(h_stride*bdx);
    bx = subBlockIndex;
    by = blockIndex;
    tx = ((subBlockOffset%h_stride)%bdx);
    ty = subBlockOffset/h_stride;
    mNewIndex = (by*l_stride*h_stride) + (tx*l_stride) + (bx*bdx) + ty + bdx*l_stride*((subBlockOffset%h_stride)/bdx);
    return mNewIndex;
}

template <typename T> 
void findBestGrid(int m, int tile_marshal, int *p_m_pad, int *p_b_dim, int *p_s, int *p_stride)
{
    int b_dim, m_pad, s, stride;
    int B_DIM_MAX, S_MAX;
    
    // due to shared memory being limited??
    if ( sizeof(T) == 4) 
    {
        B_DIM_MAX = 256;
        S_MAX     = 512;
    }
    else if (sizeof(T) == 8)
    { /* double and complex */
        B_DIM_MAX = 128;
        S_MAX     = 256;     
    }
    else 
    { /* doubleComplex */
        B_DIM_MAX = 64;
        S_MAX     = 128;    
    }
    
    /* b_dim must be multiple of 32 */
    // since warp size is 32?
    if ( m < B_DIM_MAX * tile_marshal ) 
    {
        b_dim = max(32, (m/(32*tile_marshal))*32);
        s = 1;
        m_pad = ((m + b_dim * tile_marshal - 1)/(b_dim * tile_marshal)) * (b_dim * tile_marshal);
        // m_pad is m increased to the closest multiple of (b_dim * tile_marshal)  
        stride = m_pad/(s*b_dim);    
    }
    else 
    {
        b_dim = B_DIM_MAX;
        
        s = 1;
        do {       
            int s_tmp = s * 2;
            int m_pad_tmp = ((m + s_tmp*b_dim*tile_marshal - 1)/(s_tmp*b_dim*tile_marshal)) * (s_tmp*b_dim*tile_marshal);           
            float diff = (float)(m_pad_tmp - m)/float(m);
            /* We do not want to have more than 20% oversize ... WHY?*/
            if ( diff < .2 ){
                s = s_tmp;      
            }
            else {
                break;
            }
        } while (s < S_MAX);

        m_pad = ((m + s*b_dim*tile_marshal -1)/(s*b_dim*tile_marshal)) * (s*b_dim*tile_marshal);        
        stride = m_pad/(s*b_dim);
        // m_pad = h_stride * l_stride * gridDim.y
    }
      
    *p_stride = stride;
    *p_m_pad  = m_pad;
    *p_s      = s;
    *p_b_dim  = b_dim;        
}

template <typename T, typename T_REAL> 
void tridiagonalSolver(Datablock<T, T_REAL> *data, const T* dl, T* d, const T* du, T* b, T* bNew, T *rhsUpdateArray, const int m);

template <typename T, typename T_REAL> 
void tridiagonalSolverHost(Datablock<T, T_REAL> *data, const T* dl, T* d, const T* du, T* b, T *bNew, T *rhsUpdateArray, T* x, const int m);


//template<typename T>
void setConstants(cuDoubleComplex *dx_2InvComplex);

//utility
#define EPS 1e-20

// mv_test fnxn takes tridiagonal matrix A (with diagonals a, b, c) and multiplies it with x (d) to give B (x)  
template <typename T> 
void mv_test
(
    T* x,               // result B
    const T* a,         // lower diagonal
    const T* b,         // diagonal
    const T* c,         // upper diagonal
    const T* d,         // variable vector
    const int len       // length of the matrix
)
{
    printf("Multiplying A with result x to get B ...\n");
    int m=len;
    x[0] =  cuAdd(  cuMul(b[0],d[0]), 
                        cuMul(c[0],d[1]));
    // does the multiplication of the first row
    
    // multiplication of rows 1 to m-1
    for(int i=1; i<m-1; i++)
    {   
        //x[i]=  a[i]*d[i-1]+b[i]*d[i]+c[i]*d[i+1];
        x[i]=  cuMul(a[i], d[i-1]);
        x[i]=  cuFma(b[i], d[i], x[i]);
        // cuFma first multiplies 1st 2 params and then adds 3rd one  
        x[i]=  cuFma(c[i], d[i+1], x[i]);
    }
        
    // multiplication of last row m
    x[m-1]= cuAdd(cuMul(a[m-1],d[m-2]) , cuMul(b[m-1],d[m-1]) );
    printf("Multiplication done.\n\n");
}

// mv_test fnxn takes tridiagonal matrix A (with diagonals a, b, c) and multiplies it with x (d) to give B (x)  
template <typename T> 
void mv_test_update
(
    T* x,               // result B
    T dx_2InvComplex,         // lower diagonal
    const T* b,         // diagonal
    const T* d,         // variable vector
    const int len       // length of the matrix
)
{
    printf("Multiplying updated A with result x to get new B ...\n");
    int m = len;
    x[0] =  cuAdd(cuMul(b[0], d[0]), 
                        cuMul(dx_2InvComplex, d[1]));
    // does the multiplication of the first row
    
    // multiplication of rows 1 to m-1
    for(int i=1; i<m-1; i++)
    {   
        //x[i]=  a[i]*d[i-1]+b[i]*d[i]+c[i]*d[i+1];
        x[i]=  cuMul(dx_2InvComplex, d[i-1]);
        x[i]=  cuFma(b[i], d[i], x[i]);
        // cuFma first multiplies 1st 2 params and then adds 3rd one  
        x[i]=  cuFma(dx_2InvComplex, d[i+1], x[i]);
    }
        
    // multiplication of last row m
    x[m-1]= cuAdd(cuMul(dx_2InvComplex, d[m-2]), cuMul(b[m-1], d[m-1]) );
    printf("Multiplication done.\n\n");
}


// compare_result<T, T_REAL>(h_b, h_b_back, 1, m, 1, 1e-10, 1e-10, 50, 3, b_dim);
template <typename T, typename T_REAL> 
void compare_result
(
    const T *x,             // B vector in Ax = B, given to us 
    const T *y,             // B vector in Ax = B, calc from GPU results 
    const int len,          // length of matrix 
    const T_REAL abs_err,   // for abs error checking
    const T_REAL re_err,    // for rel error checking
    const int p_bound,      // bound on error counting
    const int tx
)
{
    printf("Comparing computed B with given B.\n");
    T_REAL err = 0.0;
    T_REAL sum_err = 0.0;
    T_REAL total_sum = 0.0;
    T_REAL r_err = 1.0;
    T_REAL x_2 = 0.0;
    int p = 0; //error counter
    int t = 0; //check counter
    
    for(int i=0;i<len;i++)
    {
        T diff = cuSub(x[i], y[i]);
        err = cuReal(cuMul(diff, cuConj(diff) ));
        sum_err +=err;
        x_2 = cuReal(cuMul(x[i], cuConj(x[i])));
        total_sum += x_2;
        
        //avoid overflow in error check
        r_err = x_2 > EPS ? err/x_2:0.0;
        if(err > abs_err || r_err > re_err)
        {
            if(p < p_bound)
            {
                printf("Error occurred at element %2d, cpu = %E and gpu = %E at %d\n", i, cuReal(x[i]), cuReal(y[i]), i%tx);
                printf("Its absolute error is %le and relative error is %le.\n", err, r_err);
            }
            p++;
        }
        
        if(t < 16)
        {
            printf("Checked element %2d, cpu = %E and gpu = %E\n", i, cuReal(x[i]), cuReal(y[i]));
            t++;
        }
    }
    if(p == 0)
        printf("All correct.\n\n");
    else
        printf("There are %d errors.\n\n", p);

    printf("Total absolute error is %le\n",sqrt(sum_err));
    printf("Total relative error is %le\n",sqrt(sum_err)/sqrt(total_sum));
    printf("Comparing done.\n\n");
}

// This is a testing gtsv function
template <typename T, typename T_REAL> 
void gtsv_randomMatrix(int m, int steps)
{
    // each array is a set of elements in a diagonal stored in contiguous mem locations.
    T *h_dl;            // set of lower diagonal elements of mat A (n-1 elements)
    T *h_d;             // diagonal elements of mat A (n elements)
    T *h_du;            // set of upper diagonal elements of mat A (n-1 elements)
    T *h_b;             // RHS array has n elements
    T *h_field;         // field array to store x in cpu computation
    T *h_rhsUpdateArray;// array to be multiplied for RHS update
    T *h_bNew;          // bNew after n steps on CPU
    
    T *h_x_gpu;     // results from GPU
    T *h_bNew_gpu;  // copies updated RHS from GPU
    T *h_b_back;    // stores b computed from GPU results
    T *h_bNew_back; // stores updated RHS computed from GPU results
    T       *h_Ex;  // initial Gaussian wave
    T_REAL  *h_x;   // distance from origin
    T_REAL  *h_n;   // refractive index profile

    // vectors on the device
    T       *dl;    // lower diagonal in B
    T       *d;     // main diagonal in B
    T       *du;    // upper diagonal in B
    T       *b;     // B in Ax = B
    T       *bNew;  // to store new RHS array on device
    T       *rhsUpdateArray; // to store array which is to be multipled to get B new

    // constants
    // printf("-------------------------------------------\n");
    // printf("steps = %d\n", steps);    
    // printf("-------------------------------------------\n");
    T_REAL halfWidth= 2;
    T_REAL simDomain= 40;
    T_REAL dx       = simDomain/(m+2);
    T_REAL dx_2Inv  = 1/(dx*dx);
    T_REAL dz       = 0.55;
    T_REAL dzInv    = 1/dz;
    T_REAL nCore    = 1.5;
    T_REAL nClad    = 1.48;
    T_REAL nRef     = 1.48;
    T_REAL lambda   = 1.55;
    T_REAL k0       = 2*PI/lambda;
    T_REAL k0_2     = k0*k0;
    T_REAL beta     = k0*nRef;
    T dx_2InvComplex= cuGet<T>(-dx_2Inv, (T_REAL)0.0);
    cuDoubleComplex dx_2InvComplex_1= cuGet<cuDoubleComplex>(dx_2Inv, (T_REAL)0.0);

    // parameter declaration
    int s;                  // gridDim.x (or gridDim.y?)
    int stride;             // number of elements given to a thread
    int b_dim, m_pad;       // b_dim is used, for what? m_pad is the new size of the diagonal arrays after data transformation
    int tile_marshal = 16;  // blockDim in each direction for data marshaling
    int T_size = sizeof(T); // size of T

    // finds appropriate gridSize for data marshaling (will be referred to as DM from now on)
    findBestGrid<T>(m, tile_marshal, &m_pad, &b_dim, &s, &stride);
    printf("m = %d, m_pad = %d, s = %d, b_dim (l_stride) = %d, stride (h_stride) = %d\n", m, m_pad, s, b_dim, stride);    

    // int local_reduction_share_size   = 2*b_dim*3*T_size;
    // int global_share_size            = 2*s*3*T_size;
    // int local_solving_share_size     = (2*b_dim*2+2*b_dim+2)*T_size;
    // int marshaling_share_size        = tile_marshal*(tile_marshal+1)*T_size;
    
    Datablock<T, T_REAL> data(m, m_pad, s, steps, dx_2InvComplex, b_dim);
    dim3 gridDim(b_dim/tile_marshal, s);        // g_data
    dim3 blockDim(tile_marshal, tile_marshal);  // b_data
    data.setLaunchParameters(gridDim, blockDim, s, b_dim, tile_marshal, stride);

    // allocation of host vectors
    checkCudaErrors(cudaMallocHost((void **) &h_d, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_b, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_n, sizeof(T_REAL) * (m+2)));
    checkCudaErrors(cudaMallocHost((void **) &h_x, sizeof(T_REAL) * (m+2)));
    checkCudaErrors(cudaMallocHost((void **) &h_dl, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_du, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_field, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_Ex, T_size * (m+2)));
    checkCudaErrors(cudaMallocHost((void **) &h_x_gpu, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_b_back, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_bNew, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_bNew_gpu, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_bNew_back, T_size * m));
    checkCudaErrors(cudaMallocHost((void **) &h_rhsUpdateArray, T_size * m));
    // file is meant to store result at every step
    FILE *fp1   = fopen("output", "w");

    // setting refractive index profile, distance and initial source conditions
    int i;
    for(i=0; i<m+2; i++)
    {
        h_x[i]  = -20 + i*dx;
        if(h_x[i] > -halfWidth && h_x[i] < halfWidth)
            h_n[i] = nCore;
        else
            h_n[i] = nClad;
        h_Ex[i] = cuGet<T>(exp(-h_x[i]*h_x[i]/16), (T_REAL)0.0);
    }
    
    // allocation of device vectors
    checkCudaErrors(cudaMalloc((void **)&dl,    T_size*m)); 
    checkCudaErrors(cudaMalloc((void **)&du,    T_size*m)); 
    checkCudaErrors(cudaMalloc((void **)&d,     T_size*m)); 
    checkCudaErrors(cudaMalloc((void **)&b,     T_size*m));
    checkCudaErrors(cudaMalloc((void **)&bNew,  T_size*m));
    checkCudaErrors(cudaMalloc((void **)&rhsUpdateArray,  T_size*m));

    // the device vectors corresponding to entries of tridiagonal matrix are all set to zero
    checkCudaErrors(cudaMemset(d,  0, m * T_size));
    checkCudaErrors(cudaMemset(dl, 0, m * T_size));
    checkCudaErrors(cudaMemset(du, 0, m * T_size));
    checkCudaErrors(cudaMemset(b,  0, m * T_size));
    
    // T gammaLeft     = cuDiv(h_Ex[1], h_Ex[2]);
    // T gammaRight    = cuDiv(h_Ex[m], h_Ex[m-1]);
    T gammaLeft     = cuGet<T>((T_REAL)0.0, (T_REAL)0.0);
    T gammaRight    = cuGet<T>((T_REAL)0.0, (T_REAL)0.0);
    T constLhsTop   = cuGet<T>(2*dx_2Inv - k0_2*(pow(h_n[1], 2) - pow(nRef, 2)), 4*beta*dzInv);
    T constLhsBot   = cuGet<T>(2*dx_2Inv - k0_2*(pow(h_n[m], 2) - pow(nRef, 2)), 4*beta*dzInv);
    T constRhsTop   = cuGet<T>(-2*dx_2Inv + k0_2*(pow(h_n[1], 2) - pow(nRef, 2)), 4*beta*dzInv);
    T constRhsBot   = cuGet<T>(-2*dx_2Inv + k0_2*(pow(h_n[m], 2) - pow(nRef, 2)), 4*beta*dzInv);
    // checkCudaErrors(cudaMemcpy(data.constLhsBot, &constLhsBot, T_size, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(data.constLhsTop, &constLhsTop, T_size, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(data.constRhsBot, &constRhsBot, T_size, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(data.constRhsTop, &constRhsTop, T_size, cudaMemcpyHostToDevice));
    *(data.constLhsTop) = constLhsTop;
    *(data.constLhsBot) = constLhsBot;
    *(data.constRhsTop) = constRhsTop;
    *(data.constRhsBot) = constRhsBot;

    // setting first elements
    // first element in sub-diagonal is equal to 0 
    h_dl[0]   = cuGet<T>((T_REAL)0.0, (T_REAL)0.0); 
    h_d[0]    = cuFma(dx_2InvComplex, gammaLeft, constLhsTop);
    h_rhsUpdateArray[0] = constRhsTop;
    h_du[0]   = dx_2InvComplex;

    // setting last elements
    h_dl[m-1] = dx_2InvComplex;
    h_d[m-1]  = cuFma(dx_2InvComplex, gammaRight, constLhsBot);
    h_rhsUpdateArray[m-1] = constRhsBot;
    h_du[m-1] = cuGet<T>((T_REAL)0.0, (T_REAL)0.0);
    // last element in super diagonal is equal to 0
    
    // By following this convention, we can access elements of dl, du, d present in the same row by the row's index.

    h_b[0] = cuMul(cuFma(gammaLeft, dx_2InvComplex, constRhsTop), h_Ex[1]);
    h_b[0] = cuFma(dx_2InvComplex, h_Ex[2], h_b[0]);
    h_b[m-1] = cuMul(cuFma(gammaRight, dx_2InvComplex, constRhsTop), h_Ex[m-1]);
    h_b[m-1] = cuFma(dx_2InvComplex, h_Ex[m-2], h_b[m-1]);

    // setting interior elements
    for(int k=1; k<m-1; k++)
    {
        h_dl[k] = dx_2InvComplex;
        h_du[k] = dx_2InvComplex;
        h_d[k]  = cuGet<T>(2*dx_2Inv - k0_2*(pow(h_n[k+1], 2) - pow(nRef, 2)), 4*beta*dzInv);
        h_rhsUpdateArray[k]  = cuGet<T>(-2*dx_2Inv + k0_2*(pow(h_n[k+1], 2) - pow(nRef, 2)), 4*beta*dzInv);
        // h_b[k]  = cuGet<T>((-2*dx_2Inv + k0_2*(pow(h_n[k+1], 2) - pow(nRef, 2))) * cuReal(h_Ex[k+1]) - 4*beta*dzInv*cuImag(h_Ex[k+1]) + dx_2Inv * (cuReal(h_Ex[k]) + cuReal(h_Ex[k+2])), (-2*dx_2Inv + k0_2*(pow(h_n[k+1], 2) - pow(nRef, 2))) * cuImag(h_Ex[k+1]) + 4*beta*dzInv * cuReal(h_Ex[k+1]) + dx_2Inv * (cuImag(h_Ex[k]) + cuImag(h_Ex[k+2])));
        h_b[k]  = cuMul(cuGet<T>(dx_2InvComplex_1), h_Ex[k]);
        h_b[k]  = cuFma(cuGet<T>(-2*dx_2Inv + k0_2*(pow(h_n[k+1], 2) - pow(nRef, 2)), 4*beta*dzInv), h_Ex[k+1], h_b[k]);
        h_b[k]  = cuFma(cuGet<T>(dx_2InvComplex_1), h_Ex[k+2], h_b[k]);
    }
    
    // copying arrays from host to device
    checkCudaErrors(cudaMemcpy(dl,  h_dl,   m*T_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d,   h_d,    m*T_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(du,  h_du,   m*T_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(b,   h_b,    m*T_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rhsUpdateArray,   h_rhsUpdateArray,    m*T_size, cudaMemcpyHostToDevice));

    // setting device constant with value equal to dx_2InvComplex_1
    setConstants(&dx_2InvComplex_1);

    // finding 'marshaled' index of 1st, m-2 th, m-1 th element
    // 0 th elem remains in the same position
    int marshaledIndex_1;
    int marshaledIndex_m_2;
    int marshaledIndex_m_1;
    marshaledIndex_1   = find_marshaled_index<T, T_REAL>(&data, 1);
    marshaledIndex_m_2 = find_marshaled_index<T, T_REAL>(&data, m-2);
    marshaledIndex_m_1 = find_marshaled_index<T, T_REAL>(&data, m-1);
    data.setMarshaledIndex(marshaledIndex_1, marshaledIndex_m_2, marshaledIndex_m_1);

    // solving the matrix
    double start, stop;
    start = get_second();
    for(int i=0; i<steps; i++)
    {
        data.step = i;
        tridiagonalSolver<T, T_REAL>(&data, dl, d, du, b, bNew, rhsUpdateArray, m);
        cudaDeviceSynchronize();
        cudaGetLastError();
    }
    stop = get_second();
    printf("time on gpu = %.6f s\n", stop-start);

    // copy back the results to CPU
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_x_gpu, b, m*T_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_bNew_gpu, bNew, m*T_size, cudaMemcpyDeviceToHost));
    
    start = get_second();
    for(int i=0; i<steps; i++)
    {
        data.step = i;
        tridiagonalSolverHost<T, T_REAL>(&data, h_dl, h_d, h_du, h_b, h_bNew, h_rhsUpdateArray, h_field, m);
    }
    stop = get_second();
    printf("time on cpu = %.6f s\n", stop-start);

    compare_result<T, T_REAL>(h_field, h_x_gpu, m, 1e-10, 1e-10, 50, stride);
    compare_result<T, T_REAL>(h_bNew, h_bNew_gpu, m, 1e-10, 1e-10, 50, stride);
    
    // Uncomment the next 12 lines only when program is being run without CPU computation
    // gammaLeft = cuDiv(h_x_gpu[0], h_x_gpu[1]);
    // gammaRight = cuDiv(h_x_gpu[m-1], h_x_gpu[m-2]);
    // h_rhsUpdateArray[0] = cuFma(gammaLeft, cuGet<T>(dx_2InvComplex_1), h_rhsUpdateArray[0]);    
    // h_rhsUpdateArray[m-1] = cuFma(gammaRight, cuGet<T>(dx_2InvComplex_1), h_rhsUpdateArray[m-1]);    

    // mv_test computes B (h_b_back) in Ax = B where x is the result from the gpu
    // mv_test<T>(h_b_back, h_dl, h_d, h_du, h_x_gpu, m);
    // mv_test_update<T>(h_bNew_back, cuGet<T>(dx_2InvComplex_1), h_rhsUpdateArray, h_x_gpu, m);

    // // compares the result from the gpu and the host
    // compare_result<T, T_REAL>(h_b, h_b_back, m, 1e-10, 1e-10, 50, stride);
    // compare_result<T, T_REAL>(h_bNew_gpu, h_bNew_back, m, 1e-10, 1e-10, 50, stride);

    for(int i=0; i < m; i++)
        fprintf(fp1, "%E\n", cuAbs(h_x_gpu[i]));
    

    checkCudaErrors(cudaFreeHost(h_d));
    checkCudaErrors(cudaFreeHost(h_b));
    checkCudaErrors(cudaFreeHost(h_n));
    checkCudaErrors(cudaFreeHost(h_x));
    checkCudaErrors(cudaFreeHost(h_dl));
    checkCudaErrors(cudaFreeHost(h_du));
    checkCudaErrors(cudaFreeHost(h_Ex));
    checkCudaErrors(cudaFreeHost(h_x_gpu));
    // checkCudaErrors(cudaFreeHost(h_b_back));
    checkCudaErrors(cudaFreeHost(h_bNew_gpu));
    // checkCudaErrors(cudaFreeHost(h_bNew_back));
    checkCudaErrors(cudaFreeHost(h_rhsUpdateArray));
    // TODO: don't forget to free memory
    // no need to find best grid every time, create buffers, just replace them and free them in this function.
    // use cudaMallocHost for everything --> pinned mem
    // check whether running kernel for dl, d, du is better or copying them again is better.
}

void
showHelp()
{
    printf("\nTridiagonal Solver : Command line options\n");
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
    printf("> The default matrix size can be overridden with these parameters\n");
    printf("\t-size=row_dim_size (matrix row    dimensions)\n");
}

int 
main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        showHelp();
        return 0;
    }

    // printf("\n-------------------------------------------\n");
    int m, steps, type, devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;
    // Uncomment line 1011 in helper_cuda

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    // checkCudaErrors(cudaSetDevice(0));
    // checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

    // printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
    // printf("> SM Capability %d.%d detected.\n", deviceProp.major, deviceProp.minor);
    
    if (checkCmdLineFlag(argc, (const char **)argv, "size"))
    {
        m = getCmdLineArgumentInt(argc, (const char **)argv, "size=");

        if (m < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            if (m < 10)
            {
                printf("Enter m value which is greater than 10. Exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
        m = 1024;

    if (checkCmdLineFlag(argc, (const char **)argv, "steps"))
    {
        steps = getCmdLineArgumentInt(argc, (const char **)argv, "steps=");

        if (steps < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
    }
    else
        steps = 1;

    if (checkCmdLineFlag(argc, (const char **)argv, "type"))
    {
        type = getCmdLineArgumentInt(argc, (const char **)argv, "type=");

        if (type < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
    }
    else
        type = 1;

    // printf("-------------------------------------------\n");
    // printf("Matrix height = %d\n", m);
    // printf("-------------------------------------------\n");
    if(type == 1){
        // printf("GTSV solving using cuComplex ...\n");
        gtsv_randomMatrix<cuComplex, float>(m, steps);
        // printf("Finished GTSV solving using cuComplex\n");
    }
    if(type == 2){
        // printf("GTSV solving using cuDoubleComplex ...\n");
        gtsv_randomMatrix<cuDoubleComplex, double>(m, steps);
        // printf("Finished GTSV solving using cuDoubleComplex\n");
    }
    // printf("-------------------------------------------\n");

    return 0;
}
