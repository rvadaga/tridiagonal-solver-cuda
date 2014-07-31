#*******************************************************************************************************
#cr                              University of Illinois/NCSA Open Source License
#cr                                 Copyright (c) 2012 University of Illinois
#cr                                          All rights reserved.
#cr
#cr                                        Developed by: IMPACT Group
#cr                                          University of Illinois
#cr                                      http://impact.crhc.illinois.edu
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
#  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
#  Neither the names of IMPACT Group, University of Illinois, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
#
#******************************************************************************************************

NVCC=nvcc

# objects
OBJS= main.o  spike_host.o

# flags
CFLAGS=
CPPFLAGS= -O3
NVFLAGS= -arch=sm_30 --ptxas-options=-v -Xptxas -dlcm=ca -O3 -g -G

# final output
BIN=solver

$(BIN): $(OBJS)
	$(NVCC) -o $@ $^

clean:
	rm -rf $(OBJS) $(BIN)

%.o:%.c
	gcc $(CFLAGS) -I. -c -o $@ $<

%.o:%.cpp
	$(NVCC) $(CPPFLAGS) -I. -c -o $@ $<

%.o:%.cu
	$(NVCC) $(NVFLAGS) -I. -c -o $@ $<

