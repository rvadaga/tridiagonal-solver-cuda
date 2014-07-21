#include <stdio.h>
#include <stdlib.h>
// file to analyze data marshaling pattern.

int main()
{
	int bx, by, tx, ty, k, bdx=16, l_stride=128, h_stride=144, m=524800, g_in, g_out, shared_in, shared_out;
	
	for (bx = 7; bx<8; ++bx)
		for (by = 1; by<2; ++by){
			printf("bx = %d, by = %d.\n", bx, by);
			for (tx = 0; tx < 16; ++tx)
				for (ty = 0; ty < 16; ++ty)
				{
					g_in = (by * l_stride * h_stride) + ((bx * bdx + ty) * h_stride) + tx;
					g_out = (by * l_stride * h_stride) + (ty * l_stride) + (bx * bdx) + tx;
					shared_in = ty*(bdx+1)+tx;
					shared_out = tx*(bdx+1)+ty;
					printf("--------------------------------------------------------------\n");
					printf("tx = %d, ty = %d\n", tx, ty);
					for(k=0; k<h_stride; k+=bdx)
					{	
						// share[shared_in]= global_in >= m ? pad : y[global_in];
						if(g_in < m)
						printf("x[%4d] --> shared[%4d] 	shared[%4d] --> y[%4d] \n", g_in, shared_in, shared_out, g_out);
						else	
						printf("shared[%4d] = 0, shared[%4d] --> y[%4d]\n", shared_in, shared_out, g_out);
					g_in += bdx;
					g_out+=bdx*l_stride;
					}		
					
				}
		}
	return 0;
}
