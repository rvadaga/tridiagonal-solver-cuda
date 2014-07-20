#include <stdio.h>
#include <stdlib.h>

int main()
{
	int bx=0, by=0, tx, ty, bdx=16, g_in, g_out, shared_in, shared_out;
			for (tx = 0; tx < 16; ++tx)
				for (ty = 0; ty < 16; ++ty)
				{
					g_in = (by * l_stride * 16) + ((bx * bdx + ty) * 16) + tx;
					// g_out = (by * l_stride * 16) + (ty * l_stride) + (bx * bdx) + tx;
					shared_in = ty*(bdx+1)+tx;
					// shared_out = tx*(bdx+1)+ty;
					printf("x[%3d] --> shared[%3d]\n", g_in, shared_in);
				}
	return 0;
}
