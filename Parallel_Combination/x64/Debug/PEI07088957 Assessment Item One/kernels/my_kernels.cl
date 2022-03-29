// TUTORIAL	1 - KERNEL --------------------------------------------------------------------------------------

//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

//>Task4U-5: Change the kernel code so it does multiplication (New code)
kernel void mult(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}

//a simple smoothing kernel averaging values in a local window (radius 1)
kernel void avg_filter(global const int* A, global int* B) {
	int id = get_global_id(0);
	B[id] = (A[id - 1] + A[id] + A[id + 1])/3;
}

//a simple 2D kernel
kernel void add2D(global const int* A, global const int* B, global int* C) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y*width;

	printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id]= A[id] + B[id];
}


// TUTORIAL	2 - KERNEL --------------------------------------------------------------------------------------


//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	printf("work item id = %d\n", id);
	B[id] = A[id];

	// The output image only contains values for red colour component
	// all other components (GB) should be set to 0.
	// Also, try the filter for other channels too.

	// What is GB?

	// perform this part only once i.e. for work item 0 
	if(id == 1) {printf("work group size %d\n", get_local_size(0));}
		int loc_id = get_local_id(0);
	printf("global id = %d, local id = %d\n", id, loc_id); // do it for each item
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	B[id] = A[id];
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}

// TUTORIAL 3 - KERNEL --------------------------------------------------------------------------------------

//fixed 4 step reduce
kernel void reduce_add_1(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output
	 
	/*
	Task4U 1: What is the maximum number of elements this kernel could reduce?

	1	1	1	1	1	1	1	1	1	1	<---- START
	|	|	|	|	|	|	|	|	|	|
	| +	|	| +	|	| +	|	| +	|	| +	|
	|___|	|___2	|___2	|___2	|___2   <---- STEP 1
		|		|		|		|		|	
		|		|	+	|		|	+	|
		|		|_______4		|_______4	<---- STEP 2
		|				|				|
		|				|		+		|										
		|				|_______________8	<---- STEP 3
		|								|	
		|				+				|
		2_______________________________10	<---- STEP 4
										  

	1	2	1	2	1	4	1	2	1	10*	(B)
	1	1	1	1	1	1	1	1	1	1	(A)

	Max of elements which can be reduced = Five
	
	(*Reverse, for array, as 10 is read first)
	10	1	2	1	4	1	2	1	2	1

	Output of B as printed to console:
	[10, 1, 2, 1, 4, 1, 2, 1, 2, 1]

	TOTAL: 4 steps. Right?
	Clarify this.

	*/

	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	
	// Step 1
	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying
	
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	// Step 2
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	// Step 3
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	// Step 4
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
}

//flexible step reduce (add_2 as in second version, not +2)
kernel void reduce_add_2(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//reduce using local memory (so called privatisation)
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

//a very simple histogram implementation
kernel void hist_simple(global const int* A, global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//calculates the block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N && id < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}