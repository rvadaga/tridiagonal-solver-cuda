#include <stdio.h>
#include <stdlib.h>

int main()
{
	int bx=1, by=0, tx, ty, bdx=16, g_in, g_out, shared_in, shared_out;
	int bx=1, by=0, tx, ty, bdx=16, g_in, g_out, shared_in, shared_out;
			for (tx = 0; tx < 16; ++tx)
				for (ty = 0; ty < 16; ++ty)
				{
					// g_in = (by * 32 * 16) + ((bx * bdx + ty) * 16) + tx;
					g_out = (by * 32 * 16) + (ty * 32) + (bx * bdx) + tx;
					// shared_in = ty*(bdx+1)+tx;
					shared_out = tx*(bdx+1)+ty;
					printf("y[%3d] <-- shared[%3d]\n", g_out, shared_out);
				}
	return 0;
}
