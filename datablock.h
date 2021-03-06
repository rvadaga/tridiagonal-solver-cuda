#ifndef __DATABLOCK__
#define __DATABLOCK__
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <assert.h>
#include <stdlib.h>
//#include "cusparse_ops.hxx"

template <typename T, typename T_REAL>
class Datablock
{
    public:
    int T_size;     // size of datatype
    dim3 gridDim;   // gridDim for marshaling
    dim3 blockDim;  // blockDim for marshaling
    int local_reduction_share_size;
    int global_share_size;
    int local_solving_share_size;
    int marshaling_share_size;   
    int s, b_dim, h_stride;
    int marshaledIndex_1;
    int marshaledIndex_m_2;
    int marshaledIndex_m_1;
    FILE *fp2;

    bool* flag;     // tag for pivoting
    T* dl_buffer;   // lower digonal after DM
    T* d_buffer;    // digonal after DM
    T* du_buffer;   // upper diagonal after DM
    T* b_buffer;    // B array after DM (here, B is in Ax = B)
    T* w_buffer;    // W in A_i * W_i = vector w/ partition's lower diagonal element
    T* v_buffer;    // V in A_i * V_i = vector w/ partition's upper diagonal element
    T* c2_buffer;   // stores modified diagonal elements in diagonal pivoting method 
    T* bNew_buffer; // new DM B array after multiplying with updated A (here, B is in Ax = B)
    T* rhsUpdateArrayBuffer;    // DM main diagonal array used for finding updated RHS
    T* topElemBuffer;           // stores all the elements required by each thread within a
                                // thread block to compute its top element of updated RHS
    T* bottomElemBuffer;        // stores all the elements required by each thread within a
                                // thread block to compute its bottom element of upd. RHS

    size_t pitch;
    T* constLhsTop;
    T* constLhsBot;
    T* constRhsTop;
    T* constRhsBot;
    T* x_level_2;
    T* w_level_2;
    T* v_level_2;
    int step;
    int totalSteps;
    int mPad;
    T *gamma;
    T_REAL *field;
    T *h_gammaLeft;
    T *h_gammaRight;
    T *h_kxbLeft;
    T *h_kxbRight;
    T *dx_2InvNeg;  // -1/(dx*dx)
    T *dx_2InvPos;// 1/(dx*dx)
    T_REAL *dx;

    T *h_x_0;       // 0th element of x in Ax = B 
    T *h_x_1;       // 1st element of x in Ax = B 
    T *h_x_m_2;     // (m-2) element of x in Ax = B
    T *h_x_m_1;     // (m-1) element of x in Ax = B
    T *h_diagonal_0;       // 0 th element of main diagonal in Ax = B
    T *h_diagonal_m_1;     // (m-1) element of main diagonal in Ax = B
    Datablock(int m, int m_pad, int s, int steps, T dx_2InvCmplx, int l_stride, T_REAL deltax)
    {
        T_size = sizeof(T);
        checkCudaErrors(cudaMalloc((void **)&flag, sizeof(bool)*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&dl_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&d_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&du_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&b_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&w_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&v_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&c2_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&bNew_buffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&rhsUpdateArrayBuffer, T_size*m_pad)); 
        checkCudaErrors(cudaMalloc((void **)&bottomElemBuffer, T_size*(l_stride+1)*s)); 
        checkCudaErrors(cudaMalloc((void **)&topElemBuffer, T_size*(l_stride+1)*s)); 
        checkCudaErrors(cudaMallocHost((void **) &constLhsBot, T_size));
        checkCudaErrors(cudaMallocHost((void **) &constLhsTop, T_size));
        checkCudaErrors(cudaMallocHost((void **) &constRhsBot, T_size));
        checkCudaErrors(cudaMallocHost((void **) &constRhsTop, T_size));
        // checkCudaErrors(cudaMalloc((void **)&constLhsBot, T_size));
        // checkCudaErrors(cudaMalloc((void **)&constRhsBot, T_size));
        // checkCudaErrors(cudaMalloc((void **)&constLhsTop, T_size));
        // checkCudaErrors(cudaMalloc((void **)&constRhsTop, T_size));
        checkCudaErrors(cudaMalloc((void **)&x_level_2, T_size*s*2)); 
        checkCudaErrors(cudaMalloc((void **)&w_level_2, T_size*s*2)); 
        checkCudaErrors(cudaMalloc((void **)&v_level_2, T_size*s*2)); 

        h_x_0   = (T *)malloc(T_size);
        h_x_1   = (T *)malloc(T_size);
        h_x_m_2 = (T *)malloc(T_size); 
        h_x_m_1 = (T *)malloc(T_size); 
        h_diagonal_0 = (T *)malloc(T_size); 
        h_diagonal_m_1 = (T *)malloc(T_size); 
        
        dx = (T_REAL *)malloc(sizeof(T_REAL)); 
        h_gammaLeft = (T *)malloc(T_size); 
        h_gammaRight = (T *)malloc(T_size); 
        h_kxbLeft = (T *)malloc(T_size); 
        h_kxbRight = (T *)malloc(T_size); 
        dx_2InvNeg = (T *)malloc(T_size); 
        dx_2InvPos = (T *)malloc(T_size);

        checkCudaErrors(cudaMallocHost((void **)&h_x_0, T_size));
        checkCudaErrors(cudaMallocHost((void **)&h_x_1, T_size));
        checkCudaErrors(cudaMallocHost((void **)&h_x_m_1, T_size));
        checkCudaErrors(cudaMallocHost((void **)&h_x_m_2, T_size));
        checkCudaErrors(cudaMallocHost((void **)&h_diagonal_0, T_size));
        checkCudaErrors(cudaMallocHost((void **)&h_diagonal_m_1, T_size));
        checkCudaErrors(cudaMallocHost((void **)&h_gammaRight, T_size));
        checkCudaErrors(cudaMallocHost((void **)&h_gammaLeft, T_size));
        checkCudaErrors(cudaMallocHost((void **)&dx_2InvNeg, T_size));
        checkCudaErrors(cudaMallocHost((void **)&dx_2InvPos, T_size));
        checkCudaErrors(cudaMallocHost((void **)&gamma, T_size * m));
        checkCudaErrors(cudaMallocPitch( (void **) &field,
                                        &pitch, 
                                        sizeof(T_REAL)*s*l_stride,
                                        steps));
        *dx_2InvNeg = dx_2InvCmplx;
        *dx_2InvPos = cuNeg(dx_2InvCmplx);
        *dx = deltax;
    }

    // Destructor which frees all the array pointers allocated
    ~Datablock()
    {
        // free memory
        checkCudaErrors(cudaFree(flag));
        checkCudaErrors(cudaFree(dl_buffer));
        checkCudaErrors(cudaFree(d_buffer));
        checkCudaErrors(cudaFree(du_buffer));
        checkCudaErrors(cudaFree(b_buffer));
        checkCudaErrors(cudaFree(w_buffer));
        checkCudaErrors(cudaFree(v_buffer));
        checkCudaErrors(cudaFree(c2_buffer));
        checkCudaErrors(cudaFree(bNew_buffer));
        checkCudaErrors(cudaFree(rhsUpdateArrayBuffer));
        checkCudaErrors(cudaFree(bottomElemBuffer));
        checkCudaErrors(cudaFree(topElemBuffer));
        checkCudaErrors(cudaFree(x_level_2));
        checkCudaErrors(cudaFree(w_level_2));
        checkCudaErrors(cudaFree(v_level_2));
        // checkCudaErrors(cudaFree(constLhsTop));
        // checkCudaErrors(cudaFree(constLhsBot));
        // checkCudaErrors(cudaFree(constRhsTop));
        // checkCudaErrors(cudaFree(constRhsBot));
        // TODO: check whether all have been freed or not
        checkCudaErrors(cudaDeviceReset()); 
    }
    void setLaunchParameters(dim3 gridData, dim3 blockData, int a, int b, int tile_marshal, int stride, int steps, int m_pad)
    {
        s = a;
        b_dim = b;
        h_stride = stride;
        mPad = m_pad;
        totalSteps = steps;
        gridDim = gridData;
        blockDim = blockData;
        local_reduction_share_size  = 2*b*3*T_size;
        global_share_size           = 2*a*3*T_size;
        local_solving_share_size    = (2*b*2+2*b+2)*T_size;
        marshaling_share_size       = tile_marshal*(tile_marshal+1)*T_size;
    }
    void setMarshaledIndex(int index_1, int index_m_2, int index_m_1)
    {
        marshaledIndex_1    = index_1;
        marshaledIndex_m_2  = index_m_2;
        marshaledIndex_m_1  = index_m_1;
    } 
};
#endif