#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>

// file to analyze data marshaling pattern.

int main(int argc, char **argv)
{
    int size = atoi(argv[1]);
    printf("size = %d\n", size);
    int m = atoi(argv[2]);
    printf("element = %d\n", m);
    int b_dim, m_pad, s, stride, size_pad;
    int B_DIM_MAX, S_MAX;
    B_DIM_MAX = 128;
    S_MAX     = 256;
    int tile_marshal = 16;  

    if ( size < B_DIM_MAX * tile_marshal ) 
    {
        b_dim = fmax(32, (size/(32*tile_marshal))*32);
        s = 1;
        size_pad = ((size + b_dim * tile_marshal - 1)/(b_dim * tile_marshal)) * (b_dim * tile_marshal);
        // m_pad is m increased to the closest multiple of (b_dim * tile_marshal)  
        stride = size_pad/(s*b_dim);    
    }
    else 
    {
        b_dim = B_DIM_MAX;
        
        s = 1;
        do {       
            int s_tmp = s * 2;
            int size_pad_tmp = ((size + s_tmp*b_dim*tile_marshal - 1)/(s_tmp*b_dim*tile_marshal)) * (s_tmp*b_dim*tile_marshal);           
            float diff = (float)(size_pad_tmp - size)/((float) size);
            /* We do not want to have more than 20% oversize ... WHY?*/
            if ( diff < .2 ){
                s = s_tmp;      
            }
            else {
                break;
            }
        } while (s < S_MAX);

        size_pad = ((size + s*b_dim*tile_marshal -1)/(s*b_dim*tile_marshal)) * (s*b_dim*tile_marshal);        
        stride = size_pad/(s*b_dim);
        // m_pad = h_stride * l_stride * gridDim.y
    }
    int bx;
    int by;
    int bdx = 16;
    int tx;
    int ty;
    int blockIndex;
    int h_stride = stride;
    int l_stride = b_dim;
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
	printf("size = %d, size_pad = %d, s = %d, b_dim (l_stride) = %d, stride (h_stride) = %d\n", size, size_pad, s, b_dim, stride);
	printf("marshaled index of %d is %d.\n", m, mNewIndex);    
	return 0;
}
