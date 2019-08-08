#include <CL/cl.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


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
    printf("%i platforms\n", num_platforms);
    for(i=0;i<num_platforms;i++)
    {
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_num);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * device_num );

        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_num , devices, NULL);

        //Go through all the devices and print out its name
	       printf("Platform %i has %i devices\n", i, device_num);
         for (j = 0; j < device_num; j++)
         {
           clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &device_name_len);
           device_name = (char*) malloc(device_name_len);
           clGetDeviceInfo(devices[j], CL_DEVICE_NAME, device_name_len, device_name, NULL);

           printf("Platform: %i Device: %i --- [%s]",i,j,device_name);
	    // select the first Tesla K20m we encounter
	         if (strcmp("Tesla P100-PCIE-16GB", device_name)==0 && (int)chosen_platform < 0)
           {
	            chosen_platform = i;
	            chosen_device = j;
	            printf(" [*]");
	         }
	      printf("\n");
        }
    }

    printf("Using platform %i device %i\n", chosen_platform, chosen_device);

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

int main()
{
  clock_t start = clock(), diff;

  // CL Variables
  char filename[300] = "3_kernel.cl";

  cl_platform_id platformID;
  cl_device_id deviceID;
  cl_context context;
  cl_program program;
  cl_kernel init_kernel, step_kernel, force_kernel;
  cl_command_queue d_queue;

  cl_int err;

  char* device_name;
  size_t device_name_len;
  device_name = (char*) malloc(device_name_len);

  cl_mem position;
  cl_mem velocity;
  cl_mem force;

  // C Variables
  int chunk = 6;
  int N_l = chunk*chunk;                   // Cube of side N_l makes N particles
  int N_2 = N_l*chunk;
  int N = N_l*N_2*chunk;                     // Number of particles
  int N_t = 4096;                 // Number of time steps
  double dt = 365.25*24*60*60*500;
  double G = 6.67*pow(10,-11);
  double M_solar = 2.0*pow(10,30);
  double M_Gal; M_Gal = M_solar*pow(10,12);
  double M = M_Gal/N;
  double radius = 1.234271032*pow(10,21);
  double l; l = radius/N_l;
  double GmM_rrr,h,x,y,z;
  int i,j,k,index,index_0,k_i;

  unsigned int size_output = 3*N;
  unsigned int mem_output  = size_output * sizeof(double);

  double* output_position  = (double*) malloc(mem_output);

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
  printf("%s\n",device_name);

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

  printf("Compiling kernel(s)....\n");

  if (clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL) != CL_SUCCESS)
  {
      size_t log_size;
      clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
                                                          0, NULL, &log_size);
      char *log = (char *)malloc(log_size);
      clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
                                                          log_size, log, NULL);
      printf("CL Compilation failed:\n%s\n", log);
  }

  position = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_output, NULL, &err);
  velocity = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_output, NULL, &err);
  force    = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_output, NULL, &err);

  force_kernel = clCreateKernel(program, "kernel_force", &err);

  k_i = 0;
  err = clSetKernelArg(force_kernel, k_i, sizeof(double), &G);      k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(double), &M);      k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(double), &l);      k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(double), &dt);     k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(double), &radius); k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(int), &N);         k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(int), &N_2);       k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(int), &N_l);       k_i++;

  err = clSetKernelArg(force_kernel, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(cl_mem), &velocity); k_i++;
  err = clSetKernelArg(force_kernel, k_i, sizeof(cl_mem), &force);    k_i++;


  step_kernel = clCreateKernel(program, "kernel_t_step", &err);

  k_i = 0;
  err = clSetKernelArg(step_kernel, k_i, sizeof(double), &G);      k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(double), &M);      k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(double), &l);      k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(double), &dt);     k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(double), &radius); k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(int), &N);         k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(int), &N_2);       k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(int), &N_l);       k_i++;

  err = clSetKernelArg(step_kernel, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(cl_mem), &velocity); k_i++;
  err = clSetKernelArg(step_kernel, k_i, sizeof(cl_mem), &force);    k_i++;

  init_kernel = clCreateKernel(program, "kernel_prog_init", &err);

  k_i = 0;
  err = clSetKernelArg(init_kernel, k_i, sizeof(double), &G);      k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(double), &M);      k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(double), &l);      k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(double), &dt);     k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(double), &radius); k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(int), &N);         k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(int), &N_2);       k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(int), &N_l);       k_i++;

  err = clSetKernelArg(init_kernel, k_i, sizeof(cl_mem), &position); k_i++;
  err = clSetKernelArg(init_kernel, k_i, sizeof(cl_mem), &velocity); k_i++;


  size_t localWorkSize[3], globalWorkSize[3];

  globalWorkSize[0] = N_2;
  globalWorkSize[1] = N_2;

  localWorkSize[0] = chunk;
  localWorkSize[1] = chunk;

  printf("Running Initialising Kernel\n");
  err = clEnqueueNDRangeKernel(d_queue, init_kernel, 2, NULL, globalWorkSize,
                                                localWorkSize, 0, NULL, NULL);
  err = clFinish(d_queue);
  printf("Finished Initialising\n");


  printf("Running Main Kernel\n");


  for(int t=0;t<N_t;t++)
  {
    if(t%100==0){printf("Iteration: %d/%d\n",t,N_t);}
    err = clEnqueueNDRangeKernel(d_queue, force_kernel, 2, NULL, globalWorkSize,
                                                localWorkSize, 0, NULL, NULL);
    err = clFinish(d_queue);
    err = clEnqueueNDRangeKernel(d_queue, step_kernel, 2, NULL, globalWorkSize,
                                                localWorkSize, 0, NULL, NULL);
    err = clFinish(d_queue);
  }
  printf("Finished Main Run\n");


  diff = clock()-start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

  clEnqueueReadBuffer( d_queue, position, CL_TRUE, 0, mem_output,
                                              output_position, 0, NULL, NULL );

  printf("Saving to file\n");
  FILE *f = fopen("Gravity_output_cl.csv", "w");
  fprintf(f,"x,y,z\n" );
  for(i=0;i<N;i++)
  {
    fprintf(f,"%lf,%lf,%lf\n",output_position[3*i],
                                output_position[3*i+1],output_position[3*i+2]);
  }
  fclose(f);

  clReleaseMemObject(position);
  clReleaseMemObject(velocity);

  free(output_position);

  clReleaseProgram(program);
  clReleaseKernel(step_kernel);
  clReleaseKernel(init_kernel);
  clReleaseKernel(force_kernel);
  clReleaseCommandQueue(d_queue);
  clReleaseContext(context);

  printf("Finished.\n");
  return 0;
}
