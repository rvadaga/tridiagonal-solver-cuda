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


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cusparse_ops.hxx"
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <helper_string.h>    // helper for string parsing
#include <helper_cuda.h>      // helper for cuda error checking functions

#define DEBUG 0

static double get_second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

template <typename T, typename T_REAL> 
void gtsv_spike_partial_diag_pivot(const T* dl, const T* d, const T* du, T* b, const int m);

//utility
#define EPS 1e-20

// mv_test fnxn takes tridiagonal matrix A (with diagonals a, b, c) and multiplies it with x (d) to give B (x)  
template <typename T> 
void mv_test
(
	T* x,				// result B
	const T* a,			// lower diagonal
	const T* b,			// diagonal
	const T* c,			// upper diagonal
	const T* d,			// variable vector
	const int len		// length of the matrix
)
{
	printf("Multiplying A with result x to get B ...\n");
	int m=len;
	x[0] =  cuAdd( 	cuMul(b[0],d[0]), 
						cuMul(c[0],d[1]));
	// does the multiplication of the first row
	
	// multiplication of rows 1 to m-1
	for(int i=1; i<m-1; i++)
	{	
		//x[i]=  a[i]*d[i-1]+b[i]*d[i]+c[i]*d[i+1];
		x[i]=  cuMul(a[i], d[i-1]);
		x[i]=  cuFma(b[i], d[i], x[i]);
		// cuFma first multiplies 1st 2 params and then adds 3rd one  
		x[i]=  cuFma(c[i], d[i+1], x[i]);
	}
		
	// multiplication of last row m
	x[m-1]= cuAdd( cuMul(a[m-1],d[m-2]) , cuMul(b[m-1],d[m-1]) );
	printf("Multiplication done.\n\n");
}


// compare_result<T, T_REAL>(h_b, h_b_back, 1, m, 1, 1e-10, 1e-10, 50, 3, b_dim);
template <typename T, typename T_REAL> 
void compare_result
(
	const T *x,				// B vector in Ax = B, given to us 
	const T *y,				// B vector in Ax = B, calc from GPU results 
	const int len,			// length of matrix 
	const T_REAL abs_err,	// for abs error checking
	const T_REAL re_err,	// for rel error checking
	const int p_bound,		// bound on error counting
	const int tx
)
{
	printf("Comparing computed B with given B.\n");
	T_REAL err = 0.0;
	T_REAL sum_err = 0.0;
	T_REAL total_sum = 0.0;
	T_REAL r_err = 1.0;
	T_REAL x_2 = 0.0;
	int p = 0; //error counter
	int t = 0; //check counter
	
	for(int i=0;i<len;i++)
	{
		T diff = cuSub(x[i], y[i]);
		err = cuReal(cuMul(diff, cuConj(diff) ));
		sum_err +=err;
		x_2 = cuReal(cuMul(x[i], cuConj(x[i])));
		total_sum += x_2;
		
		//avoid overflow in error check
		r_err = x_2 > EPS ? err/x_2:0.0;
		if(err > abs_err || r_err > re_err)
		{
			if(p < p_bound)
			{
				printf("Error occurred at element %2d, cpu = %10.6lf and gpu = %10.6lf at %d\n", i, cuReal(x[i]), cuReal(y[i]), i%tx);
				printf("Its absolute error is %le and relative error is %le.\n", err, r_err);
			}
			p++;
		}
		
		if(t < 16)
		{
			printf("Checked element %2d, cpu = %10.6lf and gpu = %10.6lf\n", i, cuReal(x[i]), cuReal(y[i]));
			t++;
		}
	}
	if(p == 0)
		printf("All correct.\n\n");
	else
		printf("There are %d errors.\n\n", p);

	printf("Total absolute error is %le\n",sqrt(sum_err));
	printf("Total relative error is %le\n",sqrt(sum_err)/sqrt(total_sum));
	printf("Comparing done.\n\n");
}

//This is a testing gtsv function
template <typename T, typename T_REAL> 
void gtsv_randomMatrix(int m)
{
	double start, stop; // timers
	
	// each array is a set of elements in a diagonal stored in contiguous mem locations.
	T *h_dl; 	//	set of lower diagonal elements of mat A (n-1 elements)
	T *h_d; 	//	diagonal elements of mat A (n elements)
	T *h_du; 	//	set of upper diagonal elements of mat A (n-1 elements)
	T *h_b;		// 	RHS array has n elements
	
	T *h_x_gpu;	//	results from GPU
	T *h_b_back;// 

	// vectors on the device
	T *dl; 
	T *d;
	T *du;
	T *b;

	// allocation
	// the vectors on the device are all set to zero
	{
		h_dl=(T *)malloc(sizeof(T)*m);
		h_du=(T *)malloc(sizeof(T)*m);
		h_d=(T *)malloc(sizeof(T)*m);
		h_b=(T *)malloc(sizeof(T)*m);
		
		h_x_gpu=(T *)malloc(sizeof(T)*m);
		h_b_back=(T *)malloc(sizeof(T)*m);
				
		cudaMalloc((void **)&dl, sizeof(T)*m); 
		cudaMalloc((void **)&du, sizeof(T)*m); 
		cudaMalloc((void **)&d, sizeof(T)*m); 
		cudaMalloc((void **)&b, sizeof(T)*m); 

		cudaMemset(d, 0, m * sizeof(T));
		cudaMemset(dl, 0, m * sizeof(T));
		cudaMemset(du, 0, m * sizeof(T));
	}
	
	srand(54321);

	// used for random number generation
	// max value returned by srand is stored in RAND_MAX 
	// generate random data
	h_dl[0]   = cuGet<T>(0); 
	// first elemenyt in sub-diagonal is equal to 0 
	h_d[0]    = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0 );
	h_du[0]   = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	h_dl[m-1] = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	h_d[m-1]  = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	h_du[m-1] = cuGet<T>(0); 
	// last element in super diagonal is equal to 0
	// By following this convention, we can access elements of dl, du, d present in the same row by the row's index.

	h_b[0]    = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0 );
	h_b[m-1]  = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0 );
	
	for(int k = 1; k < m-1; k++)
	{
		h_dl[k] = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
		h_du[k] = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
		h_d[k]  = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
		h_b[k]  = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	}
	
	
	// Memory copy from host to device
	cudaMemcpy(dl, h_dl, m*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d, h_d, m*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(du, h_du, m*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(b, h_b, m*sizeof(T), cudaMemcpyHostToDevice);

	// solving a general matrix
	// noting the time stamps
    start = get_second();
    gtsv_spike_partial_diag_pivot<T,T_REAL>( dl, d, du, b, m);
    // last parameter is used to run v1
    cudaDeviceSynchronize();
	stop = get_second();

  	//copy back the results to CPU
	cudaMemcpy(h_x_gpu, b, m*sizeof(T), cudaMemcpyDeviceToHost);

    // mv_test computes B (h_b_back) in Ax = B where x is the result from the gpu
    mv_test<T>(h_b_back, h_dl, h_d, h_du, h_x_gpu, m);
    
    int b_dim = 128;
	// compares the result from the gpu and the host
	compare_result<T, T_REAL>(h_b, h_b_back, m, 1e-10, 1e-10, 50, b_dim);

    printf("time = %.6f s\n\n", stop-start);    
}

void
showHelp()
{
    printf("\nTridiagonal Solver : Command line options\n");
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
    printf("> The default matrix size can be overridden with these parameters\n");
    printf("\t-size=row_dim_size (matrix row    dimensions)\n");
}

int 
main(int argc, char **argv)
{
	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        showHelp();
        return 0;
    }

    printf("\n-------------------------------------------\n");
    int m, devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
	
	printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
    printf("> SM Capability %d.%d detected.\n", deviceProp.major, deviceProp.minor);
	
    if (checkCmdLineFlag(argc, (const char **)argv, "size"))
    {
        m = getCmdLineArgumentInt(argc, (const char **)argv, "size=");

        if (m < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            if (m < 10)
            {
                printf("Enter m value which is greater than 10. Exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
        m = 512*1024+512;

	printf("-------------------------------------------\n");
	printf("Matrix height = %d\n", m);
	printf("-------------------------------------------\n");
	printf("GTSV solving using double ...\n");
	gtsv_randomMatrix<double, double>(m);
    printf("Finished GTSV solving using double\n");
	printf("-------------------------------------------\n");

	return 0;
}
