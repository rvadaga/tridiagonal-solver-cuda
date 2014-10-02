/*******************************************************************************************************
                              University of Illinois/NCSA Open Source License
                                 Copyright (c) 2012 University of Illinois
                                          All rights reserved.

                                        Developed by: IMPACT Group
                                          University of Illinois
                                      http://impact.crhc.illinois.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
  Neither the names of IMPACT Group, University of Illinois, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

*******************************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "datablock.h"
#include "spike_kernel.hxx"
#include "cusparse_ops.hxx"
#include <complex.h>

//template <typename T>
void setConstants(cuDoubleComplex *dx_2InvNeg)
{
     checkCudaErrors(cudaMemcpyToSymbol(constant1, dx_2InvNeg,
                    sizeof(cuDoubleComplex)));
     checkCudaErrors(cudaGetLastError());
}

template <typename T, typename T_REAL> 
void tridiagonalSolver(Datablock<T, T_REAL> *data, const T* dl, T* d, const T* du, T* b, T* bNew, T* rhsUpdateArray, const int m)
{
    // prefer larger L1 cache and smaller shared memory
    cudaFuncSetCacheConfig(tiled_diag_pivot_x1<T,T_REAL>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(spike_GPU_back_sub_x1<T, T_REAL>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(multiply_kernel<T>, cudaFuncCachePreferL1);
    
    T* dl_buffer    = data->dl_buffer;    // lower digonal after DM
    T* d_buffer     = data->d_buffer;     // diagonal after DM
    T* du_buffer    = data->du_buffer;    // upper diagonal after DM
    T* b_buffer     = data->b_buffer;     // B array after DM (here, B is in Ax = B)
    T* w_buffer     = data->w_buffer;     // W in A_i * W_i = vector w/ partition's lower diagonal element
    T* v_buffer     = data->v_buffer;     // V in A_i * V_i = vector w/ partition's upper diagonal element
    T* c2_buffer    = data->c2_buffer;    // stores modified diagonal elements in diagonal pivoting method
    T* bNew_buffer  = data->bNew_buffer;  // new DM B array after multiplying with updated A (here, B is in Ax = B) 
    T* rhsUpdateArrayBuffer  = data->rhsUpdateArrayBuffer;  // DM RHS update array
    T* bottomElemBuffer      = data->bottomElemBuffer;      // elements needed for finding new rhs' bottom elems
    T* topElemBuffer         = data->topElemBuffer;         // elements needed for finding new rhs' top elems
    
    T* x_level_2 = data->x_level_2;
    T* w_level_2 = data->w_level_2;
    T* v_level_2 = data->v_level_2;
    
    int step = data->step;
    T_REAL* field = data->field;
    size_t pitch = data->pitch;

    int local_reduction_share_size  = data->local_reduction_share_size;
    int global_share_size           = data->global_share_size;
    int local_solving_share_size    = data->local_solving_share_size;
    int marshaling_share_size       = data->marshaling_share_size;

    dim3 gridDim  = data->gridDim;
    dim3 blockDim = data->blockDim;

    int s       = data->s;
    int b_dim   = data->b_dim;
    int stride  = data->h_stride;
    int tile    = 128;

    int marshaledIndex_1    = data->marshaledIndex_1;
    int marshaledIndex_m_2  = data->marshaledIndex_m_2;
    int marshaledIndex_m_1  = data->marshaledIndex_m_1;

    T_REAL dx = *(data->dx);
    T *h_x_0    = data->h_x_0;
    T *h_x_1    = data->h_x_1;
    T *h_x_m_2  = data->h_x_m_2;
    T *h_x_m_1  = data->h_x_m_1;
    T *h_diagonal_m_1   = data->h_diagonal_m_1;
    T *h_diagonal_0     = data->h_diagonal_0;

    T* h_gammaLeft      = data->h_gammaLeft;
    T* h_kxbLeft        = data->h_kxbLeft;
    T* h_gammaRight     = data->h_gammaRight;
    T* h_kxbRight       = data->h_kxbRight;
    T* dx_2InvNeg       = data->dx_2InvNeg;   // equals -1/(dx*dx)
    T* dx_2InvPos       = data->dx_2InvPos;   // equals +1/(dx*dx)
    checkCudaErrors(cudaMemset(bNew_buffer, 0, sizeof(T)*s*b_dim*stride));

    // data layout transformation
    if(data->step == 0){
        forward_marshaling_bxb<T><<<gridDim, blockDim, marshaling_share_size>>>(dl_buffer, dl, stride, b_dim, m, cuGet<T>(0));
        forward_marshaling_bxb<T><<<gridDim, blockDim, marshaling_share_size>>>(du_buffer, du, stride, b_dim, m, cuGet<T>(0));
        forward_marshaling_bxb<T><<<gridDim, blockDim, marshaling_share_size>>>(rhsUpdateArrayBuffer,  rhsUpdateArray,  stride, b_dim, m, cuGet<T>(1));
        forward_marshaling_bxb<T><<<gridDim, blockDim, marshaling_share_size>>>(d_buffer,  d,  stride, b_dim, m, cuGet<T>(1));
        forward_marshaling_bxb<T><<<gridDim, blockDim, marshaling_share_size>>>(b_buffer,  b,  stride, b_dim, m, cuGet<T>(0));
    }

    // partitioned solver
    tiled_diag_pivot_x1<T,T_REAL><<<s, b_dim>>>(b_buffer, w_buffer, v_buffer, c2_buffer, dl_buffer, d_buffer, du_buffer, stride, tile);
    
    // SPIKE solver
    spike_local_reduction_x1<T><<<s, b_dim, local_reduction_share_size>>>(b_buffer, w_buffer, v_buffer, x_level_2, w_level_2, v_level_2, stride);
    spike_GPU_global_solving_x1<<<1, 32, global_share_size>>>(x_level_2, w_level_2, v_level_2, s);
    spike_GPU_local_solving_x1<T><<<s, b_dim, local_solving_share_size>>>(b_buffer, w_buffer, v_buffer, x_level_2, stride);
    spike_GPU_back_sub_x1<T, T_REAL><<<s, b_dim>>>(b_buffer, w_buffer, v_buffer, x_level_2, stride, field + step*pitch/sizeof(T_REAL));
    // Solution to Ax = B is in b_buffer. It is data marshaled here.
    
    checkCudaErrors(cudaMemcpy(h_x_0,   b_buffer,                    sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_x_1,   b_buffer+marshaledIndex_1,   sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_x_m_2, b_buffer+marshaledIndex_m_2, sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_x_m_1, b_buffer+marshaledIndex_m_1, sizeof(T), cudaMemcpyDeviceToHost));

    *h_gammaLeft     = cuDiv(*h_x_0, *h_x_1);
    *h_gammaRight    = cuDiv(*h_x_m_1, *h_x_m_2);
    
    *h_kxbLeft = cuDiv(cuMul(cuLog(*h_gammaLeft), cuGet<T>((T_REAL)0.0, (T_REAL)1.0)), cuGet<T>(dx, (T_REAL)0.0));
    if(cuReal(*h_kxbLeft) < 0){
        *h_kxbLeft = cuGet<T>((T_REAL)0.0, cuImag(*h_kxbLeft));
        *h_gammaLeft = cuExp(cuMul(cuGet<T>((T_REAL)0.0, -dx), *h_kxbLeft));
    }
    
    *h_kxbRight = cuDiv(cuMul(cuLog(*h_gammaRight), cuGet<T>((T_REAL)0.0, (T_REAL)1.0)), cuGet<T>(dx, (T_REAL)0.0));
    if(cuReal(*h_kxbRight) < 0){
        *h_kxbRight = cuGet<T>((T_REAL)0.0, cuImag(*h_kxbRight));
        *h_gammaRight = cuExp(cuMul(cuGet<T>((T_REAL)0.0, -dx), *h_kxbRight));
    }

    *h_diagonal_0    = cuFma(*h_gammaLeft, *dx_2InvPos, *(data->constRhsTop));
    *h_diagonal_m_1  = cuFma(*h_gammaRight, *dx_2InvPos, *(data->constRhsBot));

    checkCudaErrors(cudaMemcpy(rhsUpdateArrayBuffer, h_diagonal_0, sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rhsUpdateArrayBuffer+marshaledIndex_m_1, h_diagonal_m_1, sizeof(T), cudaMemcpyHostToDevice));

    // TODO: time this thing on GPU/CPU? Check this.
    int blockSize = b_dim*stride;
    b_buffer += b_dim*(stride-1);
    topElemBuffer += 1;
    for (int i=0; i<s; i++)
        checkCudaErrors(cudaMemcpy(topElemBuffer + i*b_dim, b_buffer + i*blockSize, sizeof(T)*b_dim, cudaMemcpyDeviceToDevice));

    b_buffer -= b_dim*(stride-1);
    topElemBuffer -= 1;
    checkCudaErrors(cudaMemset(topElemBuffer, 0, sizeof(T)));

    for (int i=0; i<s; i++)
        checkCudaErrors(cudaMemcpy(bottomElemBuffer + i*b_dim, b_buffer + i*blockSize, sizeof(T)*b_dim, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(bottomElemBuffer + s*(b_dim), 0, sizeof(T)));

    // finds new RHS with rhsUpdateArrayBuffer having its 1st and last elements modified
    multiply_kernel<T><<<s, b_dim>>>(rhsUpdateArrayBuffer, topElemBuffer, bottomElemBuffer+1, b_buffer, bNew_buffer, stride, tile);

    // do back data marshaling only in the last step
    if(step == (data->totalSteps)-1){
        back_marshaling_bxb<T><<<gridDim, blockDim, marshaling_share_size>>>(b, b_buffer, stride, b_dim, m);
        back_marshaling_bxb<T><<<gridDim, blockDim, marshaling_share_size>>>(bNew, bNew_buffer, stride, b_dim, m);
        // may not be reqd.! check!!!
    }
    
    // updating A in Ax=B. This will be used again in the next step for solving. 
    else{
        checkCudaErrors(cudaMemcpy(b_buffer, bNew_buffer, sizeof(T)*data->mPad, cudaMemcpyDeviceToDevice));
        
        // modifying main diagonal
        *h_diagonal_0    = cuFma(*h_gammaLeft, *dx_2InvNeg, *(data->constLhsTop));
        *h_diagonal_m_1  = cuFma(*h_gammaRight, *dx_2InvNeg, *(data->constLhsBot));
        
        checkCudaErrors(cudaMemcpy(d_buffer, h_diagonal_0, sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_buffer+marshaledIndex_m_1, h_diagonal_m_1, sizeof(T), cudaMemcpyHostToDevice));
    }
    
    // cudaMemcpy(h_gammaLeft, d_gamma)
    // printf("Solving done.\n\n");
    // free pivotingData both h and dev
    // use checkCudaErrors for all cudaMallocs
    // change all *h_x to h_x
}

template <typename T, typename T_REAL> 
void tridiagonalSolverHost(Datablock<T, T_REAL> *data, const T* dl, T* d, const T* du, T* b, T* bNew, T* rhsUpdateArray, T* x, const int m)
{
    T* h_gammaLeft  = data->h_gammaLeft;
    T* h_gammaRight = data->h_gammaRight;
    T* h_kxbLeft    = data->h_kxbLeft;
    T* h_kxbRight   = data->h_kxbRight;
    T* dx_2InvNeg   = data->dx_2InvNeg;
    T* dx_2InvPos   = data->dx_2InvPos;
    T_REAL dx = *(data->dx);
    T *gamma  = data->gamma;
    T *h_diagonal_m_1   = data->h_diagonal_m_1;
    T *h_diagonal_0     = data->h_diagonal_0;
    T  beta = d[0];
    x[0] = cuDiv(b[0], beta);
    int i;
    for (i=1; i<m; i++){
        gamma[i] = cuDiv(du[i-1], beta);
        beta = cuFma(cuNeg(gamma[i]), dl[i], d[i]);
        x[i] = cuFma(cuNeg(x[i-1]), dl[i], b[i]);
        x[i] = cuDiv(x[i], beta);
    }
    int k;
    for (i=1; i<m; i++){
        k = m-i;
        x[k-1] = cuFma(cuNeg(x[k]), gamma[k], x[k-1]);
    }

    *h_gammaLeft     = cuDiv(x[0], x[1]);
    *h_gammaRight    = cuDiv(x[m-1], x[m-2]);
    
    *h_kxbLeft = cuDiv(cuMul(cuLog(*h_gammaLeft), cuGet<T>((T_REAL)0.0, (T_REAL)1.0)), cuGet<T>(dx, (T_REAL)0.0));
    if(cuReal(*h_kxbLeft) < 0){
        *h_kxbLeft = cuGet<T>((T_REAL)0.0, cuImag(*h_kxbLeft));
        *h_gammaLeft = cuExp(cuMul(cuGet<T>((T_REAL)0.0, -dx), *h_kxbLeft));
    }
    
    *h_kxbRight = cuDiv(cuMul(cuLog(*h_gammaRight), cuGet<T>((T_REAL)0.0, (T_REAL)1.0)), cuGet<T>(dx, (T_REAL)0.0));
    if(cuReal(*h_kxbRight) < 0){
        *h_kxbRight = cuGet<T>((T_REAL)0.0, cuImag(*h_kxbRight));
        *h_gammaRight = cuExp(cuMul(cuGet<T>((T_REAL)0.0, -dx), *h_kxbRight));
    }
    
    *h_diagonal_0    = cuFma(*h_gammaLeft, *dx_2InvPos, *(data->constRhsTop));
    *h_diagonal_m_1  = cuFma(*h_gammaRight, *dx_2InvPos, *(data->constRhsBot));
    rhsUpdateArray[0] = *h_diagonal_0;
    rhsUpdateArray[m-1] = *h_diagonal_m_1;
    
    bNew[0]   = cuAdd(cuMul(rhsUpdateArray[0], x[0]), cuMul(*dx_2InvPos, x[1]));
    bNew[m-1] = cuAdd(cuMul(*dx_2InvPos, x[m-2]), cuMul(rhsUpdateArray[m-1], x[m-1]));
    for (i=1; i<m-1; i++){
        bNew[i] = cuMul(*dx_2InvPos, x[i-1]);
        bNew[i] = cuFma(rhsUpdateArray[i], x[i], bNew[i]);
        bNew[i] = cuFma(*dx_2InvPos, x[i+1], bNew[i]);
    }

    if (data->step != (data->totalSteps-1)){
        d[0] = cuFma(*h_gammaLeft, *dx_2InvNeg, *(data->constLhsTop));
        d[m-1] = cuFma(*h_gammaRight, *dx_2InvNeg, *(data->constLhsBot));
        memcpy(b, bNew, sizeof(T)*m);
    }
}

// template<typename T>
// void set_constants(T *dx_2InvNeg);
// template void set_constants<cuComplex>(cuComplex *);
// template 
// void set_constants<cuDoubleComplex>(cuDoubleComplex *);

template <typename T, typename T_REAL> 
void tridiagonalSolver(Datablock<T, T_REAL> *data, const T* dl, T* d, const T* du, T* b, T *bNew, T *rhsUpdateArray, const int m);
/* explicit instanciation */
template void tridiagonalSolver<cuComplex, float>(Datablock<cuComplex, float> *data, const cuComplex* dl, cuComplex* d, const cuComplex* du, cuComplex* b, cuComplex *bNew, cuComplex *rhsUpdateArray, const int m);
template void tridiagonalSolver<cuDoubleComplex, double>(Datablock<cuDoubleComplex, double> *data, const cuDoubleComplex* dl, cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b, cuDoubleComplex *bNew, cuDoubleComplex *rhsUpdateArray, const int m);


template <typename T, typename T_REAL> 
void tridiagonalSolverHost(Datablock<T, T_REAL> *data, const T* dl, T* d, const T* du, T* b, T *bNew, T *rhsUpdateArray, T* x, const int m);
// explicit instanciation
template void tridiagonalSolverHost<cuComplex, float>(Datablock<cuComplex, float> *data, const cuComplex* dl, cuComplex* d, const cuComplex* du, cuComplex* b, cuComplex *bNew, cuComplex *rhsUpdateArray, cuComplex *x, const int m);
template void tridiagonalSolverHost<cuDoubleComplex, double>(Datablock<cuDoubleComplex, double> *data, const cuDoubleComplex* dl, cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b, cuDoubleComplex *bNew, cuDoubleComplex *rhsUpdateArray, cuDoubleComplex *x, const int m);