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

#include "cuComplex.h"
#include "cusparse_ops.hxx"

// Data layout transformation for inputs
// dim3 g_data(b_dim/tile_marshal (8), s (32));
// dim3 b_data(tile_marshal (16), tile_marshal (16));
// forward_marshaling_bxb<T><<<g_data (8, 32), b_data (16, 16), marshaling_share_size (16 * 17 * size(T))>>>(dl_buffer, dl, stride (144), b_dim (128), m (524800), cuGet<T>(0));
template <typename T> 
__global__ void forward_marshaling_bxb ( T* x,					// output array
                                        const T* y,				// input array
                                        const int h_stride,		// width (horizontal stride) of block to be transposed 
                                        const int l_stride,		// height (length stride) of block to be transposed
                                        int m,					// input array size 
                                        T pad 					// number to be padded in shared mem
                                        )
{	
	// This kernel does the following:
	// DM is meant to do a local transpose of a block of elements in which a thread
	// block will work later (for partitioned system solver).

	// Carefully note the difference made below between blocks, sub-blocks, and thread blocks to understand this concept.
	// The same convention has been followed throughout.

	// Consider the input array (which is a set of elements in the tridiagonal matrix) arranged as a matrix of
	// width h_stride. The matrix is padded with 'pad' such that it's size becomes equal to m_pad.
	// Now, the height of the matrix is m_pad/h_stride. This matrix is divided into blocks of size (h_stride x l_stride). 
	// This fnxn does a local transpose of these blocks. Each block is further divided into sub-blocks of size (h_stride x
	// bdx). Therefore, each block is horizontally divided into l_stride/bdx sub-blocks i.e. each block has l_stride/bdx 
	// sub-blocks arranged vertically. Note that l_stride has to be an integral multiple of bdx. Each thread block 
	// works on a single sub-block. There are tile_marshal * tile_marshal threads in a thread block. So, each thread
	// is going to work on h_stride/tile_marshal elements. Also, observe that tile_marshal = bdx.
	// So, when a block is transposed, it's sub-blocks which were in the y dimension, would now be in the x dimension.
	// There are 'b_dim/tile_marshal' thread blocks in x dimension. Note: b_dim = l_stride. 
	// All the thread-blocks in the x dimension work on single block.
	
	// A next set of b_dim/tile_marshal thread-blocks, with blockIdx.y = 1, work on the next block of h_stride x 
	// l_stride elements, which is present directly below the first block of elements. Therefore, when this block is
	// transposed locally, it's elements would be below the tranposed version of the first block of elements.
	
	int bdx;
	int global_in;
	int global_out;
	int shared_in;
	int shared_out;
	int k;	
	bdx = blockDim.x; // tile_marshal = 16

	global_in = (blockIdx.y * l_stride * h_stride) + ((blockIdx.x * bdx + threadIdx.y) * h_stride) + threadIdx.x;
	// On moving from one set of thread-blocks arranged in the x dimension, to the next set of thread-blocks below, an
	// offset of l_stride x h_stride has to be added. This explains the first term.
	global_out = (blockIdx.y * l_stride * h_stride) + (threadIdx.y * l_stride) + (blockIdx.x * bdx) + threadIdx.x;
	shared_in = threadIdx.y*(bdx+1)+threadIdx.x;
	shared_out = threadIdx.x*(bdx+1)+threadIdx.y;
	// 1 added to avoid bank conflicts

    struct __dynamic_shmem__<T> shmem; 
    T *share = shmem.getPtr();

	for(k=0; k < h_stride; k += bdx)
	{	
		share[shared_in]= global_in >= m ? pad : y[global_in];
		global_in += bdx;
		__syncthreads();	// to ensure all loads into shmem are done 
		// A thread with index (tx, ty) loads an element e1 into shared memory.
		// Another thread with index (ty, tx) loads another element e2 into shmem.
		// The first thread copies element e2 from shmem to the appr. location in output matrix.
		// And, the 2nd therad copies element e1 into the output matrix
		x[global_out] = share[shared_out];
		global_out+=bdx*l_stride;
		__syncthreads();	// to ensure all writes are done
	}		
}

// Data layout transformation for results
template <typename T> 
__global__ void  back_marshaling_bxb (
                                      T* x,
                                      const T* y,
                                      const int h_stride,
                                      const int l_stride,
                                      int m
                                      )
{	
	int bdx;
	
	int global_in;
	int global_out;
	int shared_in;
	int shared_out;
	
	bdx = blockDim.x; 	// 16

	global_out = blockIdx.y*l_stride*h_stride + (blockIdx.x*bdx + threadIdx.y)*h_stride + threadIdx.x;
	global_in = blockIdx.y*l_stride*h_stride + threadIdx.y*l_stride + blockIdx.x*bdx + threadIdx.x;
	shared_in = threadIdx.y*(bdx+1) + threadIdx.x;
	shared_out = threadIdx.x*(bdx+1) + threadIdx.y;
	
	int k;

    struct __dynamic_shmem__<T> shmem; 
    T *share = shmem.getPtr();

	for(k=0; k<h_stride; k+=bdx)
	{
	
		share[shared_in]=y[global_in];
		global_in += bdx*l_stride;
		
		__syncthreads(); // all loads into shmem are done
		
        if (global_out < m) {
		    x[global_out] = share[shared_out];
        }
		global_out+=bdx;
		__syncthreads();
	}
}


// Partitioned solver with tiled diagonal pivoting. Solves for V_i, W_i, Y_i (the modified rhs)
// Each thread works on 'h_stride' elements (i.e. a sub-matrix of h_stride x h_stride).
// In the padded tridiagonal matrix, a thread works on 'h_stride' consecutive rows.
// T_ELEM_REAL to hold the type of sigma (it is necesary for complex variants)
// All arrays (d, dl, du, b, v, w, c2) are data marshaled.
// tiled_diag_pivot_x1<T,T_REAL><<<s, b_dim>>>(b_buffer, w_buffer, v_buffer, c2_buffer, flag, dl_buffer, d_buffer, du_buffer, stride, tile);
// The strategy adopted here: dynamic tiling approach - illustrated in the code snippet below 
// k = 0
// for(i=0; i<T; i++) {
// 	n_barrier = (i+1)*n/T
// 	while(k < n_barrier) {
// 		if(condition) {
// 			1-by-1 pivoting
// 			k += 1
// 		} 
// 		else {
// 			2-by-2 pivoting
// 			k += 2
// 		}
// 	}
// 	barrier for a warp
// }

// Tridiagonal matrix A is recursively decomposed using LBM^T factorization. 
// In every step, the following operations are done: (here, f is rhs vector)
// 1) Solve Lz = f, 2) Then solve By = z, and 3) Finally, M^Tx = y
// In every block, a thread computes X, W, and V arrays.

template <typename T_ELEM , typename T_ELEM_REAL> 
__global__ void tiled_diag_pivot_x1(
                                      T_ELEM* x,			// rhs array (b_buffer)
                                      T_ELEM* w,  			// left halo (w_buffer)
                                      T_ELEM* v,  			// right halo (v_buffer)
                                      T_ELEM* b_buffer,  	// modified msin diagonal (c2_buffer)
                                      bool *flag,  			// buffer to tag pivot (flag)
                                      const T_ELEM* a,    	// lower diagonal (dl_buffer)
                                      const T_ELEM* b,    	// main diagonal (d_buffer)
                                      const T_ELEM* c,    	// upper diagonal (du_buffer)
                                      const int stride,		// h_stride (stride)
                                      const int tile, 		// tile
                                      int* pivotingData
                                      )                                    
{
	
	int b_dim;
	int ix;
	int bx;
	
	bx = blockIdx.x;
	b_dim = blockDim.x;
	ix = bx*stride*b_dim + threadIdx.x;
	
	int k = 0; // k denotes row index
	T_ELEM b_k, b_k_1, a_k_1, c_k, c_k_1, a_k_2;
	T_ELEM x_k, x_k_1;
	T_ELEM w_k, w_k_1;
	T_ELEM v_k_1;
	
	T_ELEM_REAL kappa = (sqrt(5.0)-1.0)/2.0;

	// elements in the first row
	b_k = b[ix];
	c_k = c[ix];
	// x_k = d[ix];
    x_k = x[ix];
	w_k = a[ix];
	
	// elements in the second row
	a_k_1 = a[ix+b_dim];
	b_k_1 = b[ix+b_dim];
	c_k_1 = c[ix+b_dim];
	// x_k_1 = d[ix+b_dim];
    x_k_1 = x[ix+b_dim];
	
	// element in the third row
	a_k_2 = a[ix+2*b_dim];
	
	int i;

	// forward
	for(i=1; i<=tile; i++) // dynamic tiling approach 
	{
		while(k < (stride*i)/tile)
		{        
			T_ELEM_REAL sigma;
			
			// math.h has an intrinsics for float, double 
            sigma = max(cuAbs(c_k), cuAbs(a_k_1));
			sigma = max(sigma, 		cuAbs(b_k_1));
			sigma = max(sigma, 		cuAbs(c_k_1));
			sigma = max(sigma, 		cuAbs(a_k_2));			
			
			// if condition is satisfied, 1 by 1 pivoting ==> d = 1
			if(cuMul(cuAbs(b_k), sigma) >= cuMul(kappa, cuMul(cuAbs(c_k), cuAbs(a_k_1))))
			{    
                T_ELEM b_inv = cuDiv(cuGet<T_ELEM>(1), b_k);
				flag[ix] = true; // if 1 by 1 pivoting, set flag = 1
				
				x_k = cuMul(x_k, b_inv);
				w_k = cuMul(w_k, b_inv);
				
				x[ix] = x_k;		// row k
				w[ix] = w_k;		// row k
				b_buffer[ix] = b_k;	// row k

				if(k < stride-1)	// runs as long as we are not in last row
				{
					ix += b_dim;
					// update elements in k+1 row					        
                    x_k = cuFma(cuNeg(a_k_1), x_k, x_k_1); 					// k+1 row
					w_k = cuMul(cuNeg(a_k_1), w_k);        					// k+1 row                					
                    b_k = cuFma(cuNeg(a_k_1), cuMul(c_k, b_inv), b_k_1);	// k+1 row

					if(k < stride-2) // runs on all rows excluding last, last but 1.				
					{
						// update elements in k+1, k+2 row and set elements in k+3 row
						b_k_1 = b[ix+b_dim];  // k+2 row
						a_k_1 = a_k_2;		  // k+2 row
						// x_k_1 = d[ix+b_dim];
                        x_k_1 = x[ix+b_dim];  // k+2 row
						c_k   = c_k_1;		  // k+1 row
						c_k_1 = c[ix+b_dim];  // k+2 row
						
						a_k_2 = k < (stride-3) ? a[ix+2*b_dim] : cuGet<T_ELEM>(0); // k+3 row
					}

					else  // k = stride-2, runs on last but 1 row
					{
						b_k_1 = cuGet<T_ELEM>(0);
						a_k_1 = cuGet<T_ELEM>(0);
						x_k_1 = cuGet<T_ELEM>(0);
						c_k   = cuGet<T_ELEM>(0);
						c_k_1 = cuGet<T_ELEM>(0);
						a_k_2 = cuGet<T_ELEM>(0);
					}
				}

				else  // k = stride-1, runs on last row
				{
					v[ix] = cuMul(c[ix], b_inv); // update v[last row]
					ix   += b_dim;
				}

				k += 1;
				atomicAdd(&pivotingData[0], 1);             							
			}
			
			// do 2 by 2 pivoting ==> d = 2
			else
			{		
				T_ELEM delta;
				flag[ix] = false; // flag = 0 if 2-by-2 pivoting

				// finding determinant of block B_1
                delta = cuFma(b_k, b_k_1, cuNeg(cuMul(c_k, a_k_1)));
				delta = cuDiv(cuGet<T_ELEM>(1), delta);
                x[ix] = cuFma(x_k, b_k_1, cuNeg(cuMul(c_k, x_k_1))); // k row
                x[ix] = cuMul(x[ix], delta);
                
				w[ix] = cuMul(w_k, cuMul(b_k_1, delta)); // k row
				b_buffer[ix] = b_k;				
				
                x_k_1 = cuFma(b_k, x_k_1, cuNeg(cuMul(a_k_1, x_k))); // k+1 row
                x_k_1 = cuMul(x_k_1, delta);
				w_k_1 = cuMul(cuMul(cuNeg(a_k_1), w_k), delta);	  // k+1 row
				
				x[ix+b_dim]        = x_k_1;	  // k+1 row
				w[ix+b_dim]        = w_k_1;	  // k+1 row
				b_buffer[ix+b_dim] = b_k_1;				
				flag[ix+b_dim]	   = false;	
				
				if(k < stride-2) // runs on all rows except last, last but 1.
				{
					ix += 2*b_dim;		
					// update					
                    // x_k = cuFma(cuNeg(a_k_2), x_k_1, d[ix]);		     // k+2 row
                    x_k = cuFma(cuNeg(a_k_2), 		x_k_1, x[ix]);       // k+2 row              
					w_k = cuMul(cuNeg(a_k_2), 		w_k_1);     		 // k+2 row					
                    b_k = cuMul(cuMul(a_k_2, b_k), 	cuMul(c_k_1, delta));// k+2 row
                    b_k = cuSub(b[ix], b_k);
					
					if(k < stride-3) // runs on all rows excluding last, last but 1, last but 2
					{
						// load new data
						c_k   = c[ix]; 		    // k+2 row
						b_k_1 = b[ix+b_dim];	// k+3 row
						a_k_1 = a[ix+b_dim];  	// k+3 row
						c_k_1 = c[ix+b_dim];  	// k+3 row
						// x_k_1 = d[ix+b_dim]; // k+3 row
                        x_k_1 = x[ix+b_dim];  	// k+3 row
						a_k_2 = k < stride-4 ? a[ix+2*b_dim] : cuGet<T_ELEM>(0);
					}
					else 	// k = stride-3, runs on last but 2 row
					{
						b_k_1 = cuGet<T_ELEM>(0);
						a_k_1 = cuGet<T_ELEM>(0);
						c_k   = cuGet<T_ELEM>(0);
						c_k_1 = cuGet<T_ELEM>(0);
						x_k_1 = cuGet<T_ELEM>(0);
						a_k_2 = cuGet<T_ELEM>(0);
					}
				}
				else		// k = stride-2, runs on last but 1 row
				{
					T_ELEM v_temp;
					v_temp = cuMul(c[ix+b_dim], delta);					
                    v[ix]  = cuMul(v_temp, 		cuNeg(c_k));
					v[ix+b_dim] = cuMul(v_temp,	b_k);
					ix += 2*b_dim;
				}				
				k += 2;    			
			}
		}
	}

	// Now, k = stride
	// backward
	// go to last row
	k  -= 1;
	ix -= b_dim;

	if(flag[ix]) // 1-by-1 pivoting
	{
		x_k_1 = x[ix];
		w_k_1 = w[ix];
		v_k_1 = v[ix];
		k    -= 1;
		// x[ix]
	}
	else // 2-by-2 pivoting
	{
		ix   -= b_dim;
		x_k_1 = x[ix];
		w_k_1 = w[ix];
		v_k_1 = v[ix];
		k    -= 2;
	}

	ix -= b_dim;
	
	for(i=tile-1; i>=0; i--)
	{
		while(k>=(i*stride)/tile)
		{
			if(flag[ix]) // 1-by-1 pivoting
			{	
				c_k = c[ix];
				b_k = b_buffer[ix];				
                
                T_ELEM tempDiv = cuDiv(cuNeg(c_k), b_k);
                x_k_1 = cuFma(x_k_1, tempDiv, x[ix]);                				
                w_k_1 = cuFma(w_k_1, tempDiv, w[ix]);                
				v_k_1 = cuMul(v_k_1, tempDiv);
                
				x[ix] = x_k_1;
				w[ix] = w_k_1;
				v[ix] = v_k_1;
				k -= 1;
			}
			else // 
			{
				T_ELEM delta;
				b_k   = b_buffer[ix-b_dim];
				c_k   = c[ix-b_dim];
				a_k_1 = a[ix];
				b_k_1 = b_buffer[ix];
				c_k_1 = c[ix];
                delta = cuFma(b_k, b_k_1, cuNeg(cuMul(c_k, a_k_1)));
				delta = cuDiv(cuGet<T_ELEM>(1), delta);	                
                
                T_ELEM prod = cuMul(c_k_1, cuMul(b_k, delta));
                
				x[ix] = cuFma(cuNeg(x_k_1), prod, x[ix]);
				w[ix] = cuFma(cuNeg(w_k_1), prod, w[ix]);
				v[ix] = cuMul(cuNeg(v_k_1), prod);
				
                ix   -= b_dim;
                prod  = cuMul(c_k_1, cuMul(c_k, delta));
                
                x_k_1 = cuFma(x_k_1, prod, x[ix]);
                w_k_1 = cuFma(w_k_1, prod, w[ix]);
                v_k_1 = cuMul(v_k_1, prod);
                x[ix] = x_k_1;
                w[ix] = w_k_1;                 
                v[ix] = v_k_1; 
                k -= 2;
			}             
			ix -= b_dim;         
		}            
	}    
}

// SPIKE solver within a thread block for 1x rhs
// spike_local_reduction_x1<T><<<s, b_dim, local_reduction_share_size>>>(b_buffer, w_buffer, v_buffer, x_level_2, w_level_2, v_level_2, stride);
template <typename T> 
__global__ void spike_local_reduction_x1
(
T* x,				// modified main diagonal from diagonal pivoting
T* w,				// left halo
T* v,				// right halo
T* x_mirror,		// 
T* w_mirror,		// 
T* v_mirror,		// 
const int stride  	// stride per thread
)
{
	int tx;
	int b_dim;
	int bx;
	
	tx = threadIdx.x;
	b_dim = blockDim.x;
	bx = blockIdx.x;

	// extern __shared__ T shared[];
    struct __dynamic_shmem__<T> shmem; 
    T *shared = shmem.getPtr();
        
	T* sh_w = shared;
	T* sh_v = sh_w + 2*b_dim;
	T* sh_x = sh_v + 2*b_dim;

	// a ~~ w
	// b ~~ I
	// c ~~ v
	// d ~~ x
	
	int base = bx*stride*b_dim;
	
	// load halo to scratchpad
	sh_w[tx] 		= w[base+tx];
	sh_w[tx+b_dim] 	= w[base+tx+(stride-1)*b_dim]; // w is data marshaled, so we have (stride-1)*b_dim
	sh_v[tx] 		= v[base+tx];
	sh_v[tx+b_dim] 	= v[base+tx+(stride-1)*b_dim];
	sh_x[tx] 		= x[base+tx];
	sh_x[tx+b_dim] 	= x[base+tx+(stride-1)*b_dim];
	
	__syncthreads();

	int scaler = 2;

	while(scaler <= b_dim)
	{
		if(tx < b_dim/scaler)
		{
			int index;
			int up_index;
			int down_index;

			index 		= scaler*tx + scaler/2 - 1;
			up_index	= scaler*tx;
			down_index 	= scaler*tx + scaler - 1;

			T det = cuGet<T>(1);
			det = cuFma(cuNeg(sh_v[index+b_dim]), sh_w[index+1], det);
			det = cuDiv(cuGet<T>(1), det);
			
			T d1, d2;
			d1 = sh_x[index+b_dim];
			d2 = sh_x[index+1];
			
            sh_x[index+b_dim] = cuMul(cuFma(sh_v[index+b_dim], cuNeg(d2), d1), det);
            sh_x[index+1]     = cuMul(cuFma(sh_w[index+1], cuNeg(d1), d2), det);			
            sh_w[index+1] 	  = cuMul(sh_w[index+b_dim], cuMul(sh_w[index+1], cuNeg(det)));	            
			sh_w[index+b_dim] = cuMul(sh_w[index+b_dim], det);
									
			sh_v[index+b_dim] = cuMul(sh_v[index+b_dim], cuMul(sh_v[index+1], cuNeg(det)));            
			sh_v[index+1] 	  = cuMul(sh_v[index+1], det);
			
			// boundary
            sh_x[up_index] 			= cuFma(sh_x[index+1], cuNeg(sh_v[up_index]), sh_x[up_index]);            
			sh_x[down_index+b_dim] 	= cuFma(sh_x[index+b_dim], cuNeg(sh_w[down_index+b_dim]), sh_x[down_index+b_dim]);
            
            sh_w[up_index] = cuFma(sh_w[index+1], cuNeg(sh_v[up_index]), sh_w[up_index]);
			sh_v[up_index] = cuMul(cuNeg(sh_v[index+1]), sh_v[up_index]);

            sh_v[down_index+b_dim] = cuFma(sh_v[index+b_dim], cuNeg(sh_w[down_index+b_dim]), sh_v[down_index+b_dim]);
            sh_w[down_index+b_dim] = cuMul(cuNeg(sh_w[index+b_dim]), sh_w[down_index+b_dim]);
		}
		scaler *= 2;
		__syncthreads();
	}
	
	w[base+tx] 					= sh_w[tx];
	w[base+tx+(stride-1)*b_dim] = sh_w[tx+b_dim];
	
	v[base+tx] 					= sh_v[tx];
	v[base+tx+(stride-1)*b_dim] = sh_v[tx+b_dim];
	
	x[base+tx] 					= sh_x[tx];
	x[base+tx+(stride-1)*b_dim] = sh_x[tx+b_dim];
	
	// write mirror
	if(tx < 1)
	{
		int g_dim 			= gridDim.x;
		w_mirror[bx] 		= sh_w[0];
		w_mirror[g_dim+bx] 	= sh_w[2*b_dim-1];
		
		v_mirror[bx] 		= sh_v[0];
		v_mirror[g_dim+bx] 	= sh_v[2*b_dim-1];
		
		x_mirror[bx] 		= sh_x[0];
		x_mirror[g_dim+bx] 	= sh_x[2*b_dim-1];
	}
}


///////////////////////////
/// a global level SPIKE solver for oneGPU
/// One block version
///
////////////////////
template <typename T> 
__global__ void 
spike_GPU_global_solving_x1
(
T* x,
T* w,
T* v,
const int len
)
{
	int ix;
	int b_dim;
	
	b_dim = blockDim.x;

	// extern __shared__ T shared[];
    struct __dynamic_shmem__<T> shmem; 
    T *shared = shmem.getPtr();    
    
	T* sh_w = shared;				
	T* sh_v = sh_w + 2*len;				
	T* sh_x = sh_v + 2*len;	

	// a ~~ w
	// b ~~ I
	// c ~~ v
	// d ~~ x
	
	//read data
	ix = threadIdx.x;
	while(ix < len)
	{
		sh_w[ix] 		= w[ix];
		sh_w[ix+len] 	= w[ix+len];
		
		sh_v[ix] 		= v[ix];
		sh_v[ix+len] 	= v[ix+len];
		
		sh_x[ix] 		= x[ix];
		sh_x[ix+len] 	= x[ix+len];
		
		ix 			   += b_dim;
	}
	__syncthreads();
	
	int scaler = 2;
	while(scaler <= len)
	{
		ix = threadIdx.x;
		while(ix < len/scaler)
		{
			int index;
			int up_index;
			int down_index;
			index 		= scaler*ix + scaler/2 - 1;
			up_index	= scaler*ix;
			down_index 	= scaler*ix + scaler - 1;
			T det 		= cuGet<T>(1);
			det 		= cuFma(cuNeg(sh_v[index+len]), sh_w[index+1], det);
			det 		= cuDiv(cuGet<T>(1), det);			
            
			T d1, d2;
			d1 = sh_x[index+len];
			d2 = sh_x[index+1];
			
            sh_x[index+len] 	= cuMul(cuFma( sh_v[index+len], cuNeg(d2), d1), det);
            sh_x[index+1]  		= cuMul(cuFma(sh_w[index+1], cuNeg(d1), d2), det);
            sh_w[index+1] 		= cuMul(sh_w[index+len], cuMul(sh_w[index+1], cuNeg(det)));		
			sh_w[index+len] 	= cuMul(sh_w[index+len], det);
            sh_v[index+len] 	= cuMul(sh_v[index+len], cuMul(sh_v[index+1], cuNeg(det)));    
			sh_v[index+1] 		= cuMul(sh_v[index+1], det);
			
			//boundary
            sh_x[up_index] 			= cuFma(sh_x[index+1], cuNeg(sh_v[up_index]), sh_x[up_index]); 
            sh_x[down_index+len]	= cuFma(sh_x[index+len], cuNeg(sh_w[down_index+len]), sh_x[down_index+len]);
						
            sh_w[up_index] 			= cuFma(sh_w[index+1], cuNeg(sh_v[up_index]), sh_w[up_index]);
            sh_v[up_index] 			= cuMul(cuNeg(sh_v[index+1]), sh_v[up_index]);	
            
            sh_v[down_index+len] 	= cuFma(sh_v[index+len], cuNeg(sh_w[down_index+len]), sh_v[down_index+len]);
            sh_w[down_index+len] 	= cuMul(cuNeg(sh_w[index+len]), sh_w[down_index+len]);
			ix += b_dim;
		}
		scaler *= 2;
		__syncthreads();
	}
	
	// backward reduction
	
	scaler = len/2;
	while(scaler >= 2)
	{
		ix = threadIdx.x;
		while(ix < len/scaler)
		{
			int index;
			int up_index;
			int down_index;
			index 		= scaler*ix + scaler/2 - 1;
			up_index	= scaler*ix - 1;
			down_index 	= scaler*ix + scaler;
			up_index	= up_index < 0 ? 0 : up_index;
			down_index	= down_index < len ? down_index : len-1;
			
            sh_x[index+len] = cuFma(cuNeg(sh_w[index+len]), sh_x[up_index+len], sh_x[index+len]);
            sh_x[index+len] = cuFma(cuNeg(sh_v[index+len]), sh_x[down_index], sh_x[index+len]);
            
			sh_x[index+1]   = cuFma(cuNeg(sh_w[index+1]), sh_x[up_index+len], sh_x[index+1]);
            sh_x[index+1]   = cuFma(cuNeg(sh_v[index+1]), sh_x[down_index], sh_x[index+1]);	
            
			ix += b_dim;
		}
		scaler /= 2;
		__syncthreads();
	}
	
	// write out
	ix = threadIdx.x;
	while(ix < len)
	{
		x[ix] 		= sh_x[ix];
		x[ix+len] 	= sh_x[ix+len];
		ix 		   += b_dim;
	}
}


// a thread-block level SPIKE solver 
template <typename T> 
__global__ void spike_GPU_local_solving_x1(
                                            T* x,
                                            const T* w,
                                            const T* v,
                                            const T* x_mirror,
                                            const int stride
                                          )
{
	int tx;
	int b_dim;
	int bx;
	
	tx 		= threadIdx.x;
	b_dim 	= blockDim.x;
	bx 		= blockIdx.x;

    struct __dynamic_shmem__<T> shmem; 
    T *shared = shmem.getPtr();
        
	T *sh_w	= shared;				
	T *sh_v	= sh_w + 2*b_dim;				
	T *sh_x	= sh_v + 2*b_dim;		//sh_x is 2*b_dim + 2
	
	// a ~~ w
	// b ~~ I
	// c ~~ v
	// d ~~ x
	
	int base = bx*stride*b_dim;
	
	// load halo to scratchpad
	sh_w[tx] 		= w[base+tx];
	sh_w[tx+b_dim] 	= w[base+tx+(stride-1)*b_dim];
	sh_v[tx] 		= v[base+tx];
	sh_v[tx+b_dim] 	= v[base+tx+(stride-1)*b_dim];
	
	// swap the order of x
	// why
	sh_x[tx+1] 		 = x[base+tx+(stride-1)*b_dim];
	sh_x[tx+b_dim+1] = x[base+tx];
	
	__syncthreads();
	
	if(tx < 1)
	{
		int g_dim = gridDim.x;
		sh_x[0]	  		= bx > 0 ? x_mirror[bx-1+g_dim] : cuGet<T>(0);
		sh_x[2*b_dim+1]	= bx < g_dim-1 ? x_mirror[bx+1] : cuGet<T>(0);
		sh_x[b_dim+1]	= x_mirror[bx];
		sh_x[b_dim]		= x_mirror[bx+g_dim];
	}
	__syncthreads();
	
	int scaler = b_dim;
	while(scaler >= 2)
	{
		if(tx < b_dim/scaler)
		{
			int index;
			int up_index;
			int down_index;
			index 		= scaler*tx + scaler/2 - 1;
			up_index	= scaler*tx;
			down_index 	= scaler*tx + scaler + 1;
	
            sh_x[index+1] = cuFma(sh_w[index+b_dim], cuNeg(sh_x[up_index]), sh_x[index+1]);
            sh_x[index+1] = cuFma(sh_v[index+b_dim], cuNeg(sh_x[down_index+b_dim]), sh_x[index+1]);
            
            sh_x[index+b_dim+2]  = cuFma(sh_w[index+1], cuNeg(sh_x[up_index]), sh_x[index+b_dim+2]);
            sh_x[index+b_dim+2]  = cuFma(sh_v[index+1], cuNeg(sh_x[down_index+b_dim]), sh_x[index+b_dim+2]);
		}
		scaler /= 2;
		__syncthreads();
	}
	
	// write out
	x[base+tx] 					= sh_x[tx+b_dim+1];
	x[base+tx+(stride-1)*b_dim] = sh_x[tx+1];


}

// backward substitution for SPIKE solver
template <typename T> 
__global__ void 
spike_GPU_back_sub_x1
(
T* x,
const T* w,
const T* v,
const T* x_mirror,
const int stride
)
{
	int tx;
	int b_dim;
	int bx;

	
	tx 		= threadIdx.x;
	b_dim 	= blockDim.x;
	bx 		= blockIdx.x;
	
	int base = bx*stride*b_dim;
	T x_up,x_down;
	
	if(tx>0 && tx<b_dim-1)
	{
		x_up 	= x[base+tx-1+(stride-1)*b_dim];
		x_down 	= x[base+tx+1];
	}
	else
	{
		int g_dim=gridDim.x;
		if(tx == 0)
		{
			x_up   = bx > 0 ? x_mirror[bx-1+g_dim] : cuGet<T>(0);
			x_down = x[base+tx+1];
		}
		else
		{
			x_up   = x[base+tx-1+(stride-1)*b_dim];
			x_down = bx < g_dim-1 ? x_mirror[bx+1] : cuGet<T>(0);
		}
	}
	
	int k;
	for(k=1; k<stride-1; k++)
	{        
        x[base+tx+k*b_dim] = cuFma(w[base+tx+k*b_dim], cuNeg(x_up), x[base+tx+k*b_dim]);
        x[base+tx+k*b_dim] = cuFma(v[base+tx+k*b_dim], cuNeg(x_down), x[base+tx+k*b_dim]);
	}
}

// template<typename T>
// __global__ void multiply(const T* a, const T* b, const T* c, const T* x, const T* d)
// {
// 	struct __dynamic_shmem__<T> shmem; 
//     T *share = shmem.getPtr();
	
// 	bx = blockIdx.x;
// 	b_dim = blockDim.x;
// 	ix = bx*stride*b_dim + threadIdx.x;

// 	int k = 0; // k denotes row index
	
// 	T a_k, b_k, c_k, x_k_1, x_k_2, x_k_3;
// 	a_k = a[ix];
// 	b_k = b[ix];
// 	c_k = c[ix];
// 	x_k_1 = x[ix];
// 	x_k_2 = x[ix+b_dim];
// 	x_k_3 = x[ix+b_dim];

// 	int i;

// 	for(i=1; i<=tile; i++) // dynamic tiling approach 
// 	{
// 		while(k < (stride*i)/tile)
// 		{
// 			d_k = cuAdd(d_k, cuMul(a_k, x_k_1));
// 			d_k = cuAdd(d_k, cuMul(b_k, x_k_2));
// 			d_k = cuAdd(d_k, cuMul(c_k, x_k_3));
// 			d[ix] = d_k;
// 			x_k_1 = x_k_2;
// 			x_k_2 = x_k_3;
// 			ix += b_dim;
// 			k += b_dim;
// 			x_k_3 = x[ix+b_dim];

// 		}


// }