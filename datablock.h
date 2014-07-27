#ifndef __DATABLOCK__
#define __DATABLOCK__
#include <cuda_runtime.h>
#include <helper_cuda.h>
//#include "cusparse_ops.hxx"

template <typename T, typename T_REAL>
class Datablock
{
	public:
	int T_size;		// size of datatype
	dim3 gridDim;
	dim3 blockDim;
	int local_reduction_share_size; 	
	int	global_share_size;		
	int	local_solving_share_size; 	
	int	marshaling_share_size;		
	int s, b_dim, h_stride;
	int marshaledIndex_1;
	int marshaledIndex_m_2;
	int marshaledIndex_m_1;

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

	T *h_gammaLeft;
	T *h_gammaRight;
	T *dx_2InvComplex;

    T *h_x_0;   	// 0th element of x in Ax = B 
    T *h_x_1;   	// 1st element of x in Ax = B 
    T *h_x_m_2;     // (m-2) element of x in Ax = B
    T *h_x_m_1;     // (m-1) element of x in Ax = B
    Datablock(int m_pad, int s, T dx_2InvCmplx)
    {
		T_size = sizeof(T);

		// buffer allocation
		checkCudaErrors(cudaMalloc((void **)&flag, 		sizeof(bool)*m_pad)); 
		checkCudaErrors(cudaMalloc((void **)&dl_buffer, T_size*m_pad)); 
		checkCudaErrors(cudaMalloc((void **)&d_buffer, 	T_size*m_pad)); 
		checkCudaErrors(cudaMalloc((void **)&du_buffer, T_size*m_pad)); 
		checkCudaErrors(cudaMalloc((void **)&b_buffer, 	T_size*m_pad)); 
		checkCudaErrors(cudaMalloc((void **)&w_buffer, 	T_size*m_pad)); 
		checkCudaErrors(cudaMalloc((void **)&v_buffer, 	T_size*m_pad)); 
		checkCudaErrors(cudaMalloc((void **)&c2_buffer, T_size*m_pad)); 
		
		checkCudaErrors(cudaMalloc((void **)&x_level_2, T_size*s*2)); 
		checkCudaErrors(cudaMalloc((void **)&w_level_2, T_size*s*2)); 
		checkCudaErrors(cudaMalloc((void **)&v_level_2, T_size*s*2));

		h_x_0 	= (T *)malloc(T_size);
		h_x_1 	= (T *)malloc(T_size);
		h_x_m_2 = (T *)malloc(T_size); 
		h_x_m_1 = (T *)malloc(T_size); 
		
		h_gammaLeft = (T *)malloc(T_size); 
		h_gammaRight = (T *)malloc(T_size); 
		dx_2InvComplex = (T *)malloc(T_size); 
		*dx_2InvComplex = dx_2InvCmplx;
    }

    // Destructor which frees all the array pointers allocated
    ~Datablock()
    {
		// free memory
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
    void setLaunchParameters(dim3 gridData, dim3 blockData, int a, int b, int tile_marshal, int stride)
    {
    	gridDim = gridData;
    	blockDim = blockData;
    	s = a;
    	b_dim = b;
    	h_stride = stride;
    	local_reduction_share_size 	= 2*b*3*T_size;
		global_share_size 			= 2*a*3*T_size;
		local_solving_share_size 	= (2*b*2+2*b+2)*T_size;
		marshaling_share_size 		= tile_marshal*(tile_marshal+1)*T_size;
    }
    void setMarshaledIndex(int index_1, int index_m_2, int index_m_1)
    {
    	marshaledIndex_1 	= index_1;
    	marshaledIndex_m_2 	= index_m_2;
    	marshaledIndex_m_1 	= index_m_1;
    } 
};
#endif