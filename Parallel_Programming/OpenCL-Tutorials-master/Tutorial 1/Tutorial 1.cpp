#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

//DON'T JUST COPY CODE. FLESH IT OUT WISH WORKSHOP GUIDANCE TO GIVE MEANING.

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		//!! 2.6: BASIC PROFILING STARTS HERE
		//!! 2.6.1: this is enabling profiling for the queue
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels_1.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 3 - memory allocation
		//host - input
		std::vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //C++11 allows this type of initialisation
		std::vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
		
		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(int);//size in bytes

		//host - output
		std::vector<int> C(vector_elements);
		//!!2.7.1: Large Datasets
		//if we do not wish to use the below vectors, we can comment them out
		std::vector<int> A(1000);
		std::vector<int> B(1000); // <-- is this in the right order?

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 4 - device operations

		//4.1 Copy arrays A and B to device memory
		//!!2.7.2: Is it needed to create an event 
		//Note: GPU perf is better for longer vectors than CPU. 
		cl::Event A_event; // <-- is this correct?
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		//!! 2.8.2: Creation of another kernel variable set
		cl::Kernel kernel_mult = cl::Kernel(program, "add"); // <-- intialise variables
		kernel_mult.setArg(0, buffer_A);
		kernel_mult.setArg(1, buffer_B);
		kernel_mult.setArg(2, buffer_C);

		//!! 2.6.2: Create an event and attach it to a queue command responsible for kernel launch
		cl::Event prof_event;
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, 
			cl::NDRange(vector_elements), cl::NullRange, NULL, %prof_event); // <-- Issue with operator (read up)
		//Need to understand what the above truly does, not just copy the code.

		//!! 2.8.3: New kernel launches
		queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange,
			cl::NDRange(vector_elements), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange,
			cl::NDRange(vector_elements), cl::NullRange);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl;
		
		//!! 2.6.3: Display the kernel execution time at the end of the program
		std::cout << "kernel execution time [ns]:" <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	//!!2.6.4: Get a detailed profiling account in seconds.
	//std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
	// THE ABOVE prof_event IS UNDEFINED. CHECK LATER.

	return 0;
}

/*
!! Tut2 - 1.1

int id = get_global_id(0);
printf("work item id = %d\n", id);

Where do I need to paste this?

!! Tut2 - 1.2

if (id == 0) { //perform this part only once i.e. for work item 0
printf("work group size %d\n", get_local_size(0));
}
CMP3110M/CMP9057M, Parallel Computing, Tutorial 2
int loc_id = get_local_id(0);
printf("global id = %d, local id = %d\n", id, loc_id); //do it for each work item


*/
