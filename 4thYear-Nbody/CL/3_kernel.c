#pragma OPENCL EXTENSION cl_khr_fp64: enable


__kernel void kernel_t_step(const double G,
                            const double M,
                            const double l,
                            const double dt,
                            const double radius,
                            const int N,
                            const int N_2,
                            const int N_l,
                            __global double* position,
                            __global double* velocity,
                            __global double* force
                           )
{
  int index,i,i_gi,j_gi;

  i_gi = get_global_id(0);
  j_gi = get_global_id(1);

  i = i_gi + j_gi*N_2;

  index = 3*i;   position[index] += velocity[index]*dt;
                 velocity[index] += force[index]*dt/M;
                 force[index] = 0;
  index = 3*i+1; position[index] += velocity[index]*dt;
                 velocity[index] += force[index]*dt/M;
                 force[index] = 0;
  index = 3*i+2; position[index] += velocity[index]*dt;
                 velocity[index] += force[index]*dt/M;
                 force[index] = 0;

}


__kernel void kernel_force(const double G,
                          const double M,
                          const double l,
                          const double dt,
                          const double radius,
                          const int N,
                          const int N_2,
                          const int N_l,
                          __global double* position,
                          __global double* force
                          )
{

  double h,GmM_rrr;
  double x,y,z;
  double f_x = 0.0, f_y = 0.0, f_z = 0.0;
  int i_gi, j_gi, k_gi;
  int i,j;
  int index,jndex;

  i_gi = get_global_id(0);
  j_gi = get_global_id(1);

  i = i_gi + j_gi*N_2;

  for(j=0;j<N;j++)
  {
    if( (((i+j)%2==0 && i>j) || ((i+j)%2==1 && i<j)) && i!=j )
    {
    index = 3*i;   jndex = 3*j;   x = position[index] - position[jndex];
    index = 3*i+1; jndex = 3*j+1; y = position[index] - position[jndex];
    index = 3*i+2; jndex = 3*j+2; z = position[index] - position[jndex];

    h = x*x + y*y + z*z + 0.05*l*l;
    GmM_rrr = G*M*M/(h*sqrt(h));
    index = 3*i;   force[index] -= GmM_rrr*x;
    jndex = 3*j;   force[jndex] += GmM_rrr*x;
    index = 3*i+1; force[index] -= GmM_rrr*y;
    jndex = 3*j+1; force[jndex] += GmM_rrr*y;
    index = 3*i+2; force[index] -= GmM_rrr*z;
    jndex = 3*j+2; force[jndex] += GmM_rrr*z;
    }
  }
}





//       KERNEL 2 KERNEL 2 KERNEL 2 KERNEL 2 KERNEL 2







__kernel void kernel_prog_init(const double G,
                               const double M,
                               const double l,
                               const double dt,
                               const double radius,
                               const int N,
                               const int N_2,
                               const int N_l,
                               __global double* position,
                               __global double* velocity
                               )
{
  int k_i = 0;
  double GmM_rrr;
  double x,y,z;
  double h;

  int i_gi, j_gi;
  int i,j;
  int index;

  int i_pos = 0,j_pos = 0,k_pos = 0;
  i_gi = get_global_id(0);
  j_gi = get_global_id(1);

  i = i_gi + j_gi*N_2;

  for(int i_set = 0; i_set < i; i_set++)
  {
    i_pos += 1;
    if(i_pos==N_l){i_pos -= N_l; j_pos += 1;}
    if(j_pos==N_l){j_pos -= N_l; k_pos += 1;}
  }

  z = k_pos*l - radius/2;
  x = i_pos*l - radius/2;
  y = j_pos*l - radius/2;

  index = 3*i;   position[index] = x;
  index = 3*i+1; position[index] = y;
  index = 3*i+2; position[index] = z;

  h = x*x + y*y + z*z;
  if(h>0)
  {
    index = 3*i;   velocity[index] = -y/sqrt(h)*2000;
    index = 3*i+1; velocity[index] =  x/sqrt(h)*2000;
    index = 3*i+2; velocity[index] = 0.0;
  }
  else
  {
    index = 3*i;   velocity[index] = 0.0;
    index = 3*i+1; velocity[index] = 0.0;
    index = 3*i+1; velocity[index] = 0.0;
  }
}
