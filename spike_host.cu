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
void gtsv_spike_partial_diag_pivot_v1(const T* dl, const T* d, const T* du, T* b,const int m)
{

	printf("Running GTSV SPIKE Version 1, RHS is 1.\n");
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
    
    // finds appropriate gridSize for data marshaling (will be referred to as DM from now on)
    findBestGrid<T>( m, tile_marshal, &m_pad, &b_dim, &s, &stride);
   
    printf("m = %d, m_pad = %d, s = %d, b_dim = %d, stride = %d\n", m, m_pad, s, b_dim, stride);    
	    	
	int local_reduction_share_size = 2*b_dim*3*T_size;
	int global_share_size = 2*s*3*T_size;
	int local_solving_share_size = (2*b_dim*2+2*b_dim+2)*T_size;
	int marshaling_share_size = tile_marshal*(tile_marshal+1)*T_size;
	
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
	T* c2_buffer;	// 
	
	T* x_level_2;
	T* w_level_2;
	T* v_level_2;
	
	
	//buffer allocation
	cudaMalloc((void **)&flag, sizeof(bool)*m_pad); 
	cudaMalloc((void **)&dl_buffer, T_size*m_pad); 
	cudaMalloc((void **)&d_buffer, T_size*m_pad); 
	cudaMalloc((void **)&du_buffer, T_size*m_pad); 
	cudaMalloc((void **)&b_buffer, T_size*m_pad); 
	cudaMalloc((void **)&w_buffer, T_size*m_pad); 
	cudaMalloc((void **)&v_buffer, T_size*m_pad); 
	cudaMalloc((void **)&c2_buffer, T_size*m_pad); 
	
	cudaMalloc((void **)&x_level_2, T_size*s*2); 
	cudaMalloc((void **)&w_level_2, T_size*s*2); 
	cudaMalloc((void **)&v_level_2, T_size*s*2); 
	
	//kernels 
	//data layout transformation
	printf("gridDim(%d, %d), blockDim(%d, %d)\n", g_data.x, g_data.y, b_data.x, b_data.y);
	printf("h_stride = %d, l_stride = %d\n", stride, b_dim);
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(dl_buffer, dl, stride, b_dim, m, cuGet<T>(0));
	cudaDeviceSynchronize();
	exit(1);
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(d_buffer,  d,  stride, b_dim, m, cuGet<T>(1));
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(du_buffer, du, stride, b_dim, m, cuGet<T>(0));
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(b_buffer,  b,  stride, b_dim, m, cuGet<T>(0));
	 
	//partitioned solver
	//tiled_diagonal_pivoting<<<s,b_dim>>>(x, w, v, c2_buffer, flag, dl,d,du,b, stride,tile);
	tiled_diag_pivot_x1<T,T_REAL><<<s,b_dim>>>(b_buffer, w_buffer, v_buffer, c2_buffer, flag, dl_buffer, d_buffer, du_buffer, stride, tile);
	
	
	//SPIKE solver
	spike_local_reduction_x1<T><<<s,b_dim,local_reduction_share_size>>>(b_buffer,w_buffer,v_buffer,x_level_2, w_level_2, v_level_2,stride);
	spike_GPU_global_solving_x1<<<1,32,global_share_size>>>(x_level_2,w_level_2,v_level_2,s);
	spike_GPU_local_solving_x1<T><<<s,b_dim,local_solving_share_size>>>(b_buffer,w_buffer,v_buffer,x_level_2,stride);
	spike_GPU_back_sub_x1<T><<<s,b_dim>>>(b_buffer,w_buffer,v_buffer, x_level_2,stride);

	back_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(b,b_buffer,stride,b_dim,m);
	
	//free
	
	cudaFree(flag);
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

/*
template void gtsv_spike_partial_diag_pivot_v1<float,float>(const float* dl, const float* d, const float* du, float* b,const int m);
template void gtsv_spike_partial_diag_pivot_v1<double,double>(const double* dl, const double* d, const double* du, double* b,const int m);
template void gtsv_spike_partial_diag_pivot_v1<cuComplex,float>(const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* b,const int m);
template void gtsv_spike_partial_diag_pivot_v1<cuDoubleComplex,double>(const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b,const int m);
*/


template <typename T, typename T_REAL> 
void gtsv_spike_partial_diag_pivot_v_few(const T* dl, const T* d, const T* du, T* b,const int m,const int k)
{
	//k means the number of rhs, k must be larger than 1
	//if k==1, use v1 version
	
	printf("Running GTSV SPIKE Version 2, rhs = %d.\n", k);
	// using L1 cache
	cudaFuncSetCacheConfig(tiled_diag_pivot_x1<T,T_REAL>,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(tiled_diag_pivot_x_few<T,T_REAL>,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spike_GPU_back_sub_x_few<T>,cudaFuncCachePreferL1);
	
	//parameter declaration
	int s; //gridDim.x
	int stride;
	int b_dim, m_pad; //blockDim, 
	int tile = 128;
	int tile_marshal = 16;
	int T_size = sizeof(T);
    
    findBestGrid<T>( m, tile_marshal, &m_pad, &b_dim, &s, &stride);
   
    printf("m = %d, m_pad = %d, s = %d, b_dim = %d, stride = %d\n", m, m_pad, s, b_dim, stride);    
	    	
	int local_reduction_share_size = 2*b_dim*3*T_size;
	int global_share_size = 2*s*3*T_size;
	int local_solving_share_size = (2*b_dim*2+2*b_dim+2)*T_size;
	int marshaling_share_size = tile_marshal*(tile_marshal+1)*T_size;
	
	
	dim3 g_data(b_dim/tile_marshal,s);
	dim3 b_data(tile_marshal,tile_marshal);
	
	dim3 g_dp(s,k-1);
	dim3 g_spike(s,k);
	
	
	bool* flag; // tag for pivoting
    T* dl_buffer;   //dl buffer
	T* d_buffer;    //b
	T* du_buffer; 
	T* b_buffer;
	T* w_buffer;
	T* v_buffer;
	T* c2_buffer;
	
	T* w_mirror;  //size is 2x number of total thread
	T* v_mirror;  //size is 2x number of total thread
		
	T* x_mirror2; //size is 2k time number of block and 
	T* w_mirror2;
	T* v_mirror2;
	
	
	//buffer allocation
	cudaMalloc((void **)&flag, sizeof(bool)*m_pad); 
	cudaMalloc((void **)&dl_buffer, T_size*m_pad); 
	cudaMalloc((void **)&d_buffer, T_size*m_pad); 
	cudaMalloc((void **)&du_buffer, T_size*m_pad); 
	cudaMalloc((void **)&b_buffer, T_size*m_pad*k);   //same as x 
	cudaMalloc((void **)&w_buffer, T_size*m_pad); 
	cudaMalloc((void **)&v_buffer, T_size*m_pad); 
	cudaMalloc((void **)&c2_buffer, T_size*m_pad); 
	
	cudaMalloc((void **)&w_mirror, T_size*b_dim*s*2); 
	cudaMalloc((void **)&v_mirror, T_size*b_dim*s*2); 
	
	cudaMalloc((void **)&x_mirror2, T_size*s*2*k); 
	cudaMalloc((void **)&w_mirror2, T_size*s*2); 
	cudaMalloc((void **)&v_mirror2, T_size*s*2); 
	
	//For debug 
	/*
	T* w_buffer2;
	T* v_buffer2;
	T* w_mirror22;
	T* v_mirror22;
	cudaMalloc((void **)&w_buffer2, T_size*m_pad); 
	cudaMalloc((void **)&v_buffer2, T_size*m_pad); 
	cudaMalloc((void **)&w_mirror22, T_size*s*2); 
	cudaMalloc((void **)&v_mirror22, T_size*s*2); 
	*/
	
	//kernels 
	
	//data layout transformation
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(dl_buffer, dl, stride, b_dim,m, cuGet<T>(0));
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(d_buffer,  d,  stride, b_dim,m, cuGet<T>(1));
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(du_buffer, du, stride, b_dim,m, cuGet<T>(0));
	
	//TODO: it will be replaced by a kernel
	for(int i=0;i<k;i++)
	{
	
		foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(b_buffer+i*m_pad,  b+i*m,  stride, b_dim,m, cuGet<T>(0));
	}
	 
	//partitioned solver
	//solve w, v and the fist x
	tiled_diag_pivot_x1<T,T_REAL><<<s,b_dim>>>(b_buffer, w_buffer, v_buffer, c2_buffer, flag, dl_buffer, d_buffer, du_buffer, stride, tile);
	// solve the rest x
	tiled_diag_pivot_x_few<T,T_REAL><<<g_dp,b_dim>>>(b_buffer+m_pad,flag,dl_buffer,c2_buffer,du_buffer,stride,tile,m_pad);  //correct


	//SPIKE solver
	spike_local_reduction_x_few<T><<<g_spike,b_dim,local_reduction_share_size>>>(b_buffer,w_buffer,v_buffer,w_mirror,v_mirror,x_mirror2,w_mirror2,v_mirror2,stride,m_pad); 
	spike_GPU_global_solving_x_few<T><<<k,32,global_share_size>>>(x_mirror2,w_mirror2,v_mirror2,s);    
	spike_GPU_local_solving_x_few<T><<<g_spike,b_dim,local_solving_share_size>>>(b_buffer,w_mirror,v_mirror,x_mirror2,stride,m_pad); 
	spike_GPU_back_sub_x_few<T><<<g_spike,b_dim>>>(b_buffer,w_buffer,v_buffer,x_mirror2,stride,m_pad);     
	
	
	
	
	//for debug
	/*
	cudaMemcpy(w_buffer2, w_buffer, m_pad*sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(v_buffer2, v_buffer, m_pad*sizeof(T), cudaMemcpyDeviceToDevice);
			
	spike_local_reduction_x1<T><<<s,b_dim,local_reduction_share_size>>>(b_buffer,w_buffer,v_buffer,x_mirror2, w_mirror2, v_mirror2,stride);
	spike_local_reduction_x1<T><<<s,b_dim,local_reduction_share_size>>>(b_buffer+m_pad,w_buffer2,v_buffer2,x_mirror2+2*s, w_mirror22, v_mirror22,stride);
	
	
	spike_GPU_global_solving_x1<<<1,32,global_share_size>>>(x_mirror2,w_mirror2,v_mirror2,s);
	spike_GPU_global_solving_x1<<<1,32,global_share_size>>>(x_mirror2+2*s,w_mirror22,v_mirror22,s);

	

	spike_GPU_local_solving_x1<T><<<s,b_dim,local_solving_share_size>>>(b_buffer,w_buffer,v_buffer,x_mirror2,stride);
	spike_GPU_local_solving_x1<T><<<s,b_dim,local_solving_share_size>>>(b_buffer+m_pad,w_buffer2,v_buffer2,x_mirror2+2*s,stride);

	spike_GPU_back_sub_x1<T><<<s,b_dim>>>(b_buffer,w_buffer,v_buffer, x_mirror2,stride);
	spike_GPU_back_sub_x1<T><<<s,b_dim>>>(b_buffer+m_pad,w_buffer2,v_buffer2, x_mirror2+2*s,stride);

	*/

	
	//TODO: it will be replaced by a kernel
	for(int i=0;i<k;i++)
	{
		back_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(b+i*m,b_buffer+i*m_pad,stride,b_dim,m);
	}
	
	
	
	
	//free
	
	cudaFree(flag);
	cudaFree(dl_buffer);
	cudaFree(d_buffer);
	cudaFree(du_buffer);
	cudaFree(b_buffer);
	cudaFree(w_buffer);
	cudaFree(v_buffer);
	cudaFree(c2_buffer);
	cudaFree(w_mirror);
	cudaFree(v_mirror);	
	cudaFree(x_mirror2);
	cudaFree(w_mirror2);
	cudaFree(v_mirror2);				
}

/*
template void gtsv_spike_partial_diag_pivot_v_few<float,float>(const float* dl, const float* d, const float* du, float* b,const int m,const int k);
template void gtsv_spike_partial_diag_pivot_v_few<double,double>(const double* dl, const double* d, const double* du, double* b,const int m,const int k);
template void gtsv_spike_partial_diag_pivot_v_few<cuComplex,float>(const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* b,const int m,const int k);
template void gtsv_spike_partial_diag_pivot_v_few<cuDoubleComplex,double>(const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b,const int m,const int k);
*/


template <typename T, typename T_REAL> 
void gtsv_spike_partial_diag_pivot(const T* dl, const T* d, const T* du, T* b,const int m,const int k)
{
	if(k<=0)
		return;
	if(k==1)
	{
		gtsv_spike_partial_diag_pivot_v1<T,T_REAL>(dl, d, du, b, m);
	}
	else
	{
		gtsv_spike_partial_diag_pivot_v_few<T,T_REAL>(dl, d, du, b,m,k);
	}
}

/* explicit instanciation */
template void gtsv_spike_partial_diag_pivot<float,float>(const float* dl, const float* d, const float* du, float* b,const int m,const int k);
template void gtsv_spike_partial_diag_pivot<double,double>(const double* dl, const double* d, const double* du, double* b,const int m,const int k);
template void gtsv_spike_partial_diag_pivot<cuComplex,float>(const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* b,const int m,const int k);
template void gtsv_spike_partial_diag_pivot<cuDoubleComplex,double>(const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b,const int m,const int k);

template <typename T> 
void dtsvb_spike_v1(const T* dl, const T* d, const T* du, T* b,const int m)
{



	cudaFuncSetCacheConfig(thomas_v1<T>,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spike_GPU_back_sub_x1<T>,cudaFuncCachePreferL1);
	//parameter declaration
	int s; //griddim.x
	int stride;
	int b_dim;
    int m_pad;

	int tile_marshal = 16;
	int T_size = sizeof(T);
	
    findBestGrid<T>( m, tile_marshal, &m_pad, &b_dim, &s, &stride);
   
    printf("m=%d m_pad=%d s=%d b_dim=%d stride=%d\n", m, m_pad, s, b_dim, stride);

	
	int local_reduction_share_size = 2*b_dim*3*T_size;
	int global_share_size = 2*s*3*T_size;
	int local_solving_share_size = (2*b_dim*2+2*b_dim+2)*T_size;
	int marshaling_share_size = tile_marshal*(tile_marshal+1)*T_size;
	
	
	dim3 g_data(b_dim/tile_marshal,s);
	dim3 b_data(tile_marshal,tile_marshal);
	

    T* dl_buffer;   //dl buffer
	T* d_buffer;    //b
	T* du_buffer; 
	T* b_buffer;
	T* w_buffer;
	T* v_buffer;
	T* c2_buffer;
	
	T* x_level_2;
	T* w_level_2;
	T* v_level_2;
	
	
	//buffer allocation
	cudaMalloc((void **)&dl_buffer, T_size*m_pad); 
	cudaMalloc((void **)&d_buffer, T_size*m_pad); 
	cudaMalloc((void **)&du_buffer, T_size*m_pad); 
	cudaMalloc((void **)&b_buffer, T_size*m_pad); 
	cudaMalloc((void **)&w_buffer, T_size*m_pad); 
	cudaMalloc((void **)&v_buffer, T_size*m_pad); 
	cudaMalloc((void **)&c2_buffer, T_size*m_pad); 
	
	cudaMalloc((void **)&x_level_2, T_size*s*2); 
	cudaMalloc((void **)&w_level_2, T_size*s*2); 
	cudaMalloc((void **)&v_level_2, T_size*s*2); 
	
	
	
	//kernels 
	
	//data layout transformation
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(dl_buffer, dl, stride, b_dim, m, cuGet<T>(0));
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(d_buffer,  d,  stride, b_dim, m, cuGet<T>(1));
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(du_buffer, du, stride, b_dim, m, cuGet<T>(0));
	foward_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(b_buffer,  b,  stride, b_dim, m, cuGet<T>(0));
	 
	//partitioned solver
	thomas_v1<T><<<s,b_dim>>>(b_buffer, w_buffer, v_buffer, c2_buffer, dl_buffer, d_buffer, du_buffer, stride);
	
	
	//SPIKE solver
	spike_local_reduction_x1<T><<<s,b_dim,local_reduction_share_size>>>(b_buffer,w_buffer,v_buffer,x_level_2, w_level_2, v_level_2,stride);
	spike_GPU_global_solving_x1<<<1,32,global_share_size>>>(x_level_2,w_level_2,v_level_2,s);
	spike_GPU_local_solving_x1<T><<<s,b_dim,local_solving_share_size>>>(b_buffer,w_buffer,v_buffer,x_level_2,stride);
	spike_GPU_back_sub_x1<T><<<s,b_dim>>>(b_buffer,w_buffer,v_buffer, x_level_2,stride);

	back_marshaling_bxb<T><<<g_data ,b_data, marshaling_share_size >>>(b,b_buffer,stride,b_dim, m);
	
	//free
	
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




/* explicit instanciation */
template void dtsvb_spike_v1<float>(const float* dl, const float* d, const float* du, float* b,const int m);
template void dtsvb_spike_v1<double>(const double* dl, const double* d, const double* du, double* b,const int m);
template void dtsvb_spike_v1<cuComplex>(const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* b,const int m);
template void dtsvb_spike_v1<cuDoubleComplex>(const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b,const int m);
