#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h" // Additional - Tutorial 2

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl; // Additional - Tutorial 2
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 1; // device_id 1 is required
	string image_filename = "test.pgm"; // Changed format from 'ppm' to 'pgm'

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	/*
	----------------------------------------------------------------------------------------

	Platform 0: Intel OpenCL, version: OpenCL 2.1, vendor: Intel Corporation

		Device 0: Intel UHD Graphics 620, 
				  version: OpenC: 2.1 NEO,
				  vendor: Intel Corporation,
				  type: GPU, 
				  compute units: 24,
				  clock freq [Mhz]: 1600,
				  max memory size [B]: 3388375040
				  max allocatable memory [B]: 1694187250

		Device 1: Intel Core i5-8250U CPU @ 1.60GHz,
				  version: OpenCL 2.1 (Build 0),
				  vendor: Intel Corporation,
				  type: CPU,
				  compute units: 8,
				  clock freq [Mhz]: 1600,
				  max memory size [B]: 8470945792,
				  max allocatable memory [B]: 2117736448

	----------------------------------------------------------------------------------------

	Platform 1: Nvidia CUDA, version OpenCL 1.2 CUDA 10.2.115, vendor: Nvidia Corporation
	
		Device 0: GeForce GTX 1050, 
				  version: OpenCL 1.2 CUDA, 
				  vendor, NVIDIA Corporation,
				  type: GPU,
				  computer units: 5 
				  clock frequency [Mhz]: 1493,
				  max memory size [B]: 2147483648
				  max memory allocation [B]: 536870912

	----------------------------------------------------------------------------------------

	Platform 2: Intel OpenCL, version: OpenCL 2.1 WINDOWS, vendor: Intel Corporation

		Device 0: Intel Core i5-8250U CPU @ 1.60GHz,
				  version: OpenCL 2.1 (Build 0),
				  vendor: Intel Corporation,
				  type: CPU,
				  compute units: 8,
				  clock freq [Mhz]: 1600,
				  max memory size [B]: 8470945792,
				  max allocatable memory [B]: 2117736448

	----------------------------------------------------------------------------------------
	*/

	cimg::exception_mode(0); // Exception "Image not found"

	//detect any potential exceptions
	try {

		CImg<unsigned char> image_input(image_filename.c_str()); // Digital image is represented as a byte array
		CImgDisplay disp_input(image_input,"input");

		// Convolution Mask is added for Tutorial 2
		//a 3x3 convolution mask implementing an averaging filter -- try to understand this
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id); // same as Tut 1

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // same as Tut 1

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); // it is possible to add CL_QUEUE_PROFILING_ENABLE as the context

		// ------ try to understand the benefits of profiling again ------

		//3.2 Load & build the device code
		cl::Program::Sources sources; 

		AddSources(sources, "kernels/my_kernels.cl"); // This is consistent

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

		/*
		//Section 2.7.1: Large Datasets -- From Tut 1

		// 10^3 / 1,000
		std::vector<int> A(pow(10, 3)); // REMOVE
		std::vector<int> B(pow(10, 3)); // REMOVE
		*/

		// POSSIBLE memory allocation - From Tut 3

		typedef int mytype; // what is this for?

		std::vector<mytype> A(10, 1); // this allocates elements for reduction
		size_t local_size = 1;
		size_t padding_size = A.size() % local_size;
		
		if (padding_size) {
			std::vector<int> A_ext(local_size-padding_size, 0);
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}
		
		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size; // the input vector must be divisible by the work group size

		//host - output
		std::vector<mytype> B(input_elements);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		/*
		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);


		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory


		size_t vector_elements = A.size();//number of elements -- Tut 1
		size_t vector_size = A.size() * sizeof(int);//size in bytes -- Tut 1

		//host - output -- Tut 1
		std::vector<int> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 4 - device operations -- Tut 1

		//4.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		//>Task4U-3: Profiling the code for different vector lengths on a CPU and GPU device (New code)
		cl::Event A_event;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);

		//Tut 1 - POST Task4U-4 --------------------

		//Section 2.8.1: Kernels - Setup and execute the mult kernel
		cl::Kernel kernel_mult = cl::Kernel(program, "mult");
		kernel_mult.setArg(0, buffer_A);
		kernel_mult.setArg(1, buffer_B);
		kernel_mult.setArg(2, buffer_C);

		queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

		//4.2 Setup and execute the add* kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		
		//Section 2.6.2: Basic Profiling - Create an event
		cl::Event prof_event;
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);
		*/

		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size()); // Read only
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]); // Same as buffer_A in Tutorial 1
//		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		// Same as the integer input
		cl::Kernel kernel = cl::Kernel(program, "identity");
		kernel.setArg(0, dev_image_input); // Image in
		kernel.setArg(1, dev_image_output); // Image out
//		kernel.setArg(2, dev_convolution_mask); // Additional convolution mask

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		
		/*
		// Take the vector elements from Tutorial 1
		// Is this a dependancy, (local_size)? Part of include perhaps?
		// ^^ If implemented into Tut 1, would this work?
		// Maybe find NDRange?
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

		// import code from Tutorial 1 to make it function
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; //get device 
		cerr << kernel_add.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; //get info

		kernel_add.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
		*/

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }
		/*
		//4.2 Setup and execute all kernels (i.e. device code) -- From Tut 3
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype))); //local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		*/
	}
	catch (const cl::Error& err) { // no string input?
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) { // no image input
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
