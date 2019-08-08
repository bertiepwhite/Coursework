#include <CL/cl.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

void SelectDevice(cl_platform_id *platformID,cl_device_id *deviceID);

void SelectDevice(cl_platform_id *platformID,cl_device_id *deviceID)
{
    int i,j;
    cl_int err;

    cl_uint chosen_platform = -1;
    cl_uint chosen_device = -1;
    //Platform variables
    cl_platform_id *platforms;
    cl_uint num_platforms = 0;

    //Device Variables
    cl_device_id *devices;
    cl_uint device_num = 0;

    //Selection Variables
    char* device_name;
    size_t device_name_len;

    /***FIND WHAT DEVICES ARE ON SYSTEM AND DISPLAY THEM THERE NAMES***/

    //Get all platforms and create a list of them
    clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);

    err = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Go through the list of platforms and evaluate all the devices on them
    //printf("%i platforms\n", num_platforms);
    for(i=0;i<num_platforms;i++)
    {
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * device_num );

        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_num , devices, NULL);

        //Go through all the devices and print out its name
	       //printf("Platform %i has %i devices\n", i, device_num);
         for (j = 0; j < device_num; j++)
         {
           clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &device_name_len);
           device_name = (char*) malloc(device_name_len);
           clGetDeviceInfo(devices[j], CL_DEVICE_NAME, device_name_len, device_name, NULL);

           //printf("Platform: %i Device: %i --- [%s]",i,j,device_name);
	    // select the first Tesla K20m we encounter
	         if (strcmp("Tesla P100-PCIE-16GB", device_name)==0 && (int)chosen_platform < 0)
           {
	            chosen_platform = i;
	            chosen_device = j;
	            //printf(" [*]");
	         }
	      //printf("\n");
        }
    }

    //printf("Using platform %i device %i\n", chosen_platform, chosen_device);

    {
      int ch_plat;
      ch_plat = (int)chosen_platform;
      int ch_dev;
      ch_dev = (int)chosen_device;
    }

    /***ASSIGNMENT AND OUTPUT***/
    clGetDeviceIDs(platforms[chosen_platform],
                                      CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * device_num );

    err = clGetDeviceIDs(platforms[chosen_platform], CL_DEVICE_TYPE_ALL,
                                                    device_num , devices, NULL);

    //Return the selected platform ID and Device ID back to the main function
    *platformID = platforms[chosen_platform];
    *deviceID = devices[chosen_device];
}

int main( int argc, char *argv[] )
{

  clock_t start = clock(), diff;

  // CL Variables
  char filename[300] = "Gravity_me_cl.cl";

  cl_platform_id platformID;
  cl_device_id deviceID;
  cl_context context;
  cl_program program;
  cl_kernel part_kernel;
  cl_kernel full_kernel;
  cl_kernel init_kernel;
  cl_kernel init_kernel_2;
  cl_kernel Energy_kernel;
  cl_command_queue d_queue;

  cl_int err;

  char* device_name;
  size_t device_name_len;
  device_name = (char*) malloc(device_name_len);

  cl_mem position;
  cl_mem velocity;
  cl_mem masses;
  cl_mem Energy;

  // C Variables
  int chunk = atoi(argv[2]);             // Chunk must perfectly divide N
  int N = atoi(argv[1]);               // Number of particles
  int N_t = atoi(argv[3]);                 // Number of time steps
  double dt = 1.0/(24.0*60.0*60.0);
  double G = 0.00011900032;
  double M_solar = 1.0/0.000003003;
  int i,k_i,t;
  int saveEnergy = 0;
  printf("N:%d, N_t:%d, chunk:%d\n",N,N_t,chunk);
  unsigned int size_output = 3*N;
  unsigned int mem_output  = size_output * sizeof(double);
  unsigned int red_output = N*sizeof(double);
  double Energy_count;

  double* output_position = (double*) malloc(mem_output);
  double* output_velocity = (double*) malloc(mem_output);
  double* output_energies = (double*) malloc(red_output);

  SelectDevice(&platformID,&deviceID);
  /* -------------------------- Other OpenCL Selections ---------------------*/

  // Context Creation
  const cl_context_properties context_config[3] = {CL_CONTEXT_PLATFORM,
                                          (cl_context_properties)platformID,0};
  context = clCreateContext(context_config, 1, &deviceID, NULL, NULL, &err);

  // Device Creation
  clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &device_name_len);
  device_name = (char*) malloc(device_name_len);
  clGetDeviceInfo(deviceID, CL_DEVICE_NAME, device_name_len, device_name, NULL);
  //printf("%s\n",device_name);

  // Command queue creation
  d_queue = clCreateCommandQueue(context, deviceID, 0, &err);

  // Kernel Load
  FILE *kernel_file;
  char *source_str;
  int source_num=0;
  size_t source_size;
  kernel_file = fopen(filename,"r");
  fseek(kernel_file, 0L, SEEK_END);
  source_num = ftell(kernel_file);
  fseek(kernel_file, 0, SEEK_SET);

  source_str = (char*)malloc(source_num);
  source_size = fread(source_str,1,source_num,kernel_file);
  fclose(kernel_file);

  // Creating the kernel, compiling it
  program = clCreateProgramWithSource(context, 1, (const char **) &source_str,
                                            (const size_t *)&source_size, &err);

  //printf("Compiling kernel(s)....\n");

  if (clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL) != CL_SUCCESS)
  {
      size_t log_size;
      clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
                                                          0, NULL, &log_size);
      char *log = (char *)malloc(log_size);
      clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
                                                          log_size, log, NULL);
      //printf("CL Compilation failed:\n%s\n", log);
  }

  position = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_output, NULL, &err);
  velocity = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_output, NULL, &err);
  masses   = clCreateBuffer(context, CL_MEM_READ_WRITE, red_output, NULL, &err);
  Energy   = clCreateBuffer(context, CL_MEM_READ_WRITE, red_output, NULL, &err);

  init_kernel = clCreateKernel(program, "kernel_prog_init", &err);

  k_i = 0;
  err = clSetKernelArg(init_kernel, k_i, sizeof(double), &G);       k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(double), &M_solar); k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(int), &N);          k_i++;

  err = clSetKernelArg(init_kernel, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(cl_mem), &velocity); k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(cl_mem), &masses);   k_i++;

  init_kernel_2 = clCreateKernel(program, "kernel_prog_init_2", &err);

  k_i = 0;
  err = clSetKernelArg(init_kernel_2, k_i, sizeof(int), &N); k_i++;

  err = clSetKernelArg(init_kernel_2, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(init_kernel_2, k_i, sizeof(cl_mem), &velocity); k_i++;
  err = clSetKernelArg(init_kernel_2, k_i, sizeof(cl_mem), &masses);   k_i++;


  part_kernel = clCreateKernel(program, "kernel_prog_part_step", &err);

  k_i = 0;
  err = clSetKernelArg(part_kernel, k_i, sizeof(double), &G);      k_i++;
  err = clSetKernelArg(part_kernel, k_i, sizeof(double), &dt);     k_i++;
  err = clSetKernelArg(part_kernel, k_i, sizeof(int), &N);         k_i++;

  err = clSetKernelArg(part_kernel, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(part_kernel, k_i, sizeof(cl_mem), &velocity); k_i++;
  err = clSetKernelArg(part_kernel, k_i, sizeof(cl_mem), &masses);   k_i++;

  full_kernel = clCreateKernel(program, "kernel_prog_full_step", &err);

  k_i = 0;
  err = clSetKernelArg(full_kernel, k_i, sizeof(double), &G);      k_i++;
  err = clSetKernelArg(full_kernel, k_i, sizeof(double), &dt);     k_i++;
  err = clSetKernelArg(full_kernel, k_i, sizeof(int), &N);         k_i++;

  err = clSetKernelArg(full_kernel, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(full_kernel, k_i, sizeof(cl_mem), &velocity); k_i++;
  err = clSetKernelArg(full_kernel, k_i, sizeof(cl_mem), &masses);   k_i++;

  Energy_kernel = clCreateKernel(program, "kernel_prog_Energy", &err);

  k_i = 0;
  err = clSetKernelArg(Energy_kernel, k_i, sizeof(double), &G);      k_i++;
  err = clSetKernelArg(Energy_kernel, k_i, sizeof(int), &N);         k_i++;
  err = clSetKernelArg(Energy_kernel, k_i, sizeof(cl_mem), &Energy); k_i++;

  err = clSetKernelArg(Energy_kernel, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(Energy_kernel, k_i, sizeof(cl_mem), &velocity); k_i++;
  err = clSetKernelArg(Energy_kernel, k_i, sizeof(cl_mem), &masses);   k_i++;

  size_t localWorkSize[1], globalWorkSize[1];

  globalWorkSize[0] = N;

  localWorkSize[0] = chunk;

  //printf("Running Initialising Kernel\n");
  err = clEnqueueNDRangeKernel(d_queue, init_kernel,   1, NULL, globalWorkSize,
                                              localWorkSize, 0, NULL, NULL);
  err = clFinish(d_queue);
  err = clEnqueueNDRangeKernel(d_queue, init_kernel_2, 1, NULL, globalWorkSize,
                                              localWorkSize, 0, NULL, NULL);
  err = clFinish(d_queue);
  err = clEnqueueNDRangeKernel(d_queue, part_kernel,   1, NULL, globalWorkSize,
                                              localWorkSize, 0, NULL, NULL);
  err = clFinish(d_queue);
  //printf("Finished Initialising\n");

  FILE* f;
  if(saveEnergy==1){
  //printf("Saving to file\n");
  f = fopen("Gravity_output_cl.csv", "w");
  fprintf(f,"E\n" );
  }

  //printf("Running Main Kernel\n");
  for(int t=0;t<N_t;t++)
  {
    if(t%100==0){printf("Iteration: %d/%d\n",t,N_t);}
    err = clEnqueueNDRangeKernel(d_queue, full_kernel,   1, NULL, globalWorkSize,
                                                localWorkSize, 0, NULL, NULL);
    err = clFinish(d_queue);
    if(saveEnergy==1)
    {
      err = clEnqueueNDRangeKernel(d_queue, Energy_kernel,1, NULL, globalWorkSize,
                                                    localWorkSize, 0, NULL, NULL);
      err = clFinish(d_queue);
      clEnqueueReadBuffer( d_queue, Energy, CL_TRUE, 0, red_output,
                                                  output_energies, 0, NULL, NULL);
      Energy_count = 0;
      for(i=0;i<N;i++){Energy_count+=output_energies[i];}
      fprintf(f,"%lf\n",Energy_count);
      }
    }

  if(saveEnergy==1){fclose(f);}

  diff = clock()-start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

  clEnqueueReadBuffer( d_queue, position, CL_TRUE, 0, mem_output,
                                              output_position, 0, NULL, NULL );
  clEnqueueReadBuffer( d_queue, velocity, CL_TRUE, 0, mem_output,
                                              output_velocity, 0, NULL, NULL );



  clReleaseMemObject(position);
  clReleaseMemObject(velocity);
  clReleaseMemObject(masses);

  free(output_position);
  free(output_velocity);
  free(output_energies);

  clReleaseProgram(program);
  clReleaseKernel(part_kernel);
  clReleaseKernel(full_kernel);
  clReleaseKernel(init_kernel);
  clReleaseKernel(init_kernel_2);
  clReleaseKernel(Energy_kernel);
  clReleaseCommandQueue(d_queue);
  clReleaseContext(context);

  //printf("Finished.\n");
  return 0;
}
