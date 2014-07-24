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

#include "spike_kernel.hxx"

// findBestGrid

template <typename T> void findBestGrid( int m, int tile_marshal, int *p_m_pad, int *p_b_dim, int *p_s, int *p_stride)
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
void gtsv_spike_partial_diag_pivot(const T* dl, const T* d, const T* du, T* b, const int m)
{

	printf("\nRunning GTSV SPIKE.\n");
	cudaFuncSetCacheConfig(tiled_diag_pivot_x1<T,T_REAL>,cudaFuncCachePreferL1);
	// prefer larger L1 cache and smaller shared memory
	cudaFuncSetCacheConfig(spike_GPU_back_sub_x1<T>,cudaFuncCachePreferL1);
	
	// parameter declaration
	int s; 					// gridDim.x (or gridDim.y?)
	int stride;				// number of elements given to a thread
	int b_dim, m_pad;		// b_dim is used, for what? m_pad is the new size of the diagonal arrays after data transformation
	int tile = 128;			// 
	int tile_marshal = 16; 	// blockDim in each direction for data marshaling
	int T_size = sizeof(T);	// size of datatype
    int *pivotingData;
    int *h_pivotingData;

    cudaMalloc((void **)&pivotingData, sizeof(int));
    cudaMemset((void *)pivotingData, 0, sizeof(int));
    h_pivotingData = (int *)malloc(sizeof(int));

    // finds appropriate gridSize for data marshaling (will be referred to as DM from now on)
    findBestGrid<T>( m, tile_marshal, &m_pad, &b_dim, &s, &stride);
   
    printf("m = %d, m_pad = %d, s = %d, b_dim (l_stride) = %d, stride (h_stride) = %d\n", m, m_pad, s, b_dim, stride);    
	    	
	int local_reduction_share_size 	= 2*b_dim*3*T_size;
	int global_share_size 			= 2*s*3*T_size;
	int local_solving_share_size 	= (2*b_dim*2+2*b_dim+2)*T_size;
	int marshaling_share_size 		= tile_marshal*(tile_marshal+1)*T_size;
	
	dim3 g_data(b_dim/tile_marshal, s);
	dim3 b_data(tile_marshal, tile_marshal);
	
	// _buffer suffix - matrix with transformed data layout

	bool* flag; 	// tag for pivoting
    T* dl_buffer;   // lower digonal after DM
	T* d_buffer;    // digonal after DM
	T* du_buffer; 	// upper diagonal after DM
	T* b_buffer;	// B array after DM (here, B is in Ax = B)
	T* w_buffer;	// W in A_i * W_i = vector w/ partition's lower diagonal element
	T* v_buffer;	// V in A_i * V_i = vector w/ partition's upper diagonal element
	T* c2_buffer;	// stores modified diagonal elements in diagonal pivoting method 
	
	T* x_level_2;
	T* w_level_2;
	T* v_level_2;
	
	//buffer allocation
	cudaMalloc((void **)&flag, 		sizeof(bool)*m_pad); 
	cudaMalloc((void **)&dl_buffer, T_size*m_pad); 
	cudaMalloc((void **)&d_buffer, 	T_size*m_pad); 
	cudaMalloc((void **)&du_buffer, T_size*m_pad); 
	cudaMalloc((void **)&b_buffer, 	T_size*m_pad); 
	cudaMalloc((void **)&w_buffer, 	T_size*m_pad); 
	cudaMalloc((void **)&v_buffer, 	T_size*m_pad); 
	cudaMalloc((void **)&c2_buffer, T_size*m_pad); 
	
	cudaMalloc((void **)&x_level_2, T_size*s*2); 
	cudaMalloc((void **)&w_level_2, T_size*s*2); 
	cudaMalloc((void **)&v_level_2, T_size*s*2); 
	
	//kernels 
	//data layout transformation
	forward_marshaling_bxb<T><<<g_data, b_data, marshaling_share_size>>>(dl_buffer, dl, stride, b_dim, m, cuGet<T>(0));
	forward_marshaling_bxb<T><<<g_data, b_data, marshaling_share_size>>>(d_buffer,  d,  stride, b_dim, m, cuGet<T>(1));
	forward_marshaling_bxb<T><<<g_data, b_data, marshaling_share_size>>>(du_buffer, du, stride, b_dim, m, cuGet<T>(0));
	forward_marshaling_bxb<T><<<g_data, b_data, marshaling_share_size>>>(b_buffer,  b,  stride, b_dim, m, cuGet<T>(0));
	 
	// partitioned solver
	// tiled_diagonal_pivoting<<<s,b_dim>>>(x, w, v, c2_buffer, flag, dl,d,du,b, stride,tile);
	tiled_diag_pivot_x1<T,T_REAL><<<s, b_dim>>>(b_buffer, w_buffer, v_buffer, c2_buffer, flag, dl_buffer, d_buffer, du_buffer, stride, tile, pivotingData);
	
	
	// SPIKE solver
	spike_local_reduction_x1<T><<<s, b_dim, local_reduction_share_size>>>(b_buffer, w_buffer, v_buffer, x_level_2, w_level_2, v_level_2, stride);
	spike_GPU_global_solving_x1<<<1, 32, global_share_size>>>(x_level_2, w_level_2, v_level_2, s);
	spike_GPU_local_solving_x1<T><<<s, b_dim, local_solving_share_size>>>(b_buffer, w_buffer, v_buffer, x_level_2, stride);
	spike_GPU_back_sub_x1<T><<<s, b_dim>>>(b_buffer, w_buffer, v_buffer, x_level_2, stride);

	back_marshaling_bxb<T><<<g_data, b_data, marshaling_share_size>>>(b, b_buffer, stride, b_dim, m);
	cudaMemcpy(h_pivotingData, pivotingData, sizeof(int), cudaMemcpyDeviceToHost);
	printf("No of 1 by 1 pivotings done = %d.\n", *h_pivotingData);
	printf("Solving done.\n\n");
	
	// free memory
	cudaFree(flag);
	cudaFree(pivotingData);
	cudaFree(dl_buffer);
	cudaFree(d_buffer);
	cudaFree(du_buffer);
	cudaFree(b_buffer);
	cudaFree(w_buffer);
	cudaFree(v_buffer);
	cudaFree(c2_buffer);
	cudaFree(x_level_2);
	cudaFree(w_level_2);
	cudaFree(v_level_2);				
}

template <typename T, typename T_REAL> 
void gtsv_spike_partial_diag_pivot(const T* dl, const T* d, const T* du, T* b, const int m);
/* explicit instanciation */
template void gtsv_spike_partial_diag_pivot<float, float>(const float* dl, const float* d, const float* du, float* b, const int m);
template void gtsv_spike_partial_diag_pivot<double, double>(const double* dl, const double* d, const double* du, double* b, const int m);
template void gtsv_spike_partial_diag_pivot<cuComplex, float>(const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* b,const int m);
template void gtsv_spike_partial_diag_pivot<cuDoubleComplex, double>(const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b, const int m);