#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void kernel_prog_full_step(  const double G,
                                      const double dt,
                                      const int N,
                                   __global double* position,
                                   __global double* velocity,
                                   __global double* masses
                                   )
{
  int i = get_global_id(0);
  int j,k;
  double GmM,Fx=0.0,Fy=0.0,Fz=0.0;
  double x,y,z,r;
  int index, jndex;
  for(j=0;j<i;j++)
  {
      index = 3*i;   jndex = 3*j;   x = position[index] - position[jndex];
      index = 3*i+1; jndex = 3*j+1; y = position[index] - position[jndex];
      index = 3*i+2; jndex = 3*j+2; z = position[index] - position[jndex];

      r = sqrt(x*x+y*y+z*z);
      //The tiny offset allows the i = j case to be calculated but go to 0
      // avoiding a gpu expensive if statement
      GmM = G*masses[i]*masses[j]/((r+0.0000000001)*(r+0.0000000001));
      Fx -= GmM*x/r; Fy -= GmM*y/r; Fz -= GmM*z/r;
  }
  for(j=i+1;j<N;j++)
  {
      index = 3*i;   jndex = 3*j;   x = position[index] - position[jndex];
      index = 3*i+1; jndex = 3*j+1; y = position[index] - position[jndex];
      index = 3*i+2; jndex = 3*j+2; z = position[index] - position[jndex];

      r = sqrt(x*x+y*y+z*z);
      //The tiny offset allows the i = j case to be calculated but go to 0
      // avoiding a gpu expensive if statement
      GmM = G*masses[i]*masses[j]/((r+0.0000000001)*(r+0.0000000001));
      Fx -= GmM*x/r; Fy -= GmM*y/r; Fz -= GmM*z/r;
      //if(i==5){printf("%lf",Fx);}
  }

  //if(i==5){printf("%lf",position[3*i+0]);}
  velocity[3*i+0] += 0.5*dt*(Fx/masses[i]);
  position[3*i+0] += velocity[3*i+0]*dt+0.5*dt*dt*(Fx/masses[i]);
  velocity[3*i+0] += 0.5*dt*(Fx/masses[i]);

  velocity[3*i+1] += 0.5*dt*(Fy/masses[i]);
  position[3*i+1] += velocity[3*i+1]*dt+0.5*dt*dt*(Fy/masses[i]);
  velocity[3*i+1] += 0.5*dt*(Fy/masses[i]);

  velocity[3*i+2] += 0.5*dt*(Fz/masses[i]);
  position[3*i+2] += velocity[3*i+2]*dt+0.5*dt*dt*(Fz/masses[i]);
  velocity[3*i+2] += 0.5*dt*(Fz/masses[i]);

}





//       KERNEL 2 KERNEL 2 KERNEL 2 KERNEL 2 KERNEL 2


__kernel void kernel_prog_part_step(
                                    const double G,
                                    const double dt,
                                    const int N,
                                 __global double* position,
                                 __global double* velocity,
                                 __global double* masses
                                 )
{
  int i = get_global_id(0);
  int j,k;
  double GmM,Fx=0.0,Fy=0.0,Fz=0.0;
  double x,y,z,r;

  int index, jndex;
  for(j=0;j<i;j++)
  {

      index = 3*i;   jndex = 3*j;   x = position[index] - position[jndex];
      index = 3*i+1; jndex = 3*j+1; y = position[index] - position[jndex];
      index = 3*i+2; jndex = 3*j+2; z = position[index] - position[jndex];

      r = sqrt(x*x+y*y+z*z);
      //The tiny offset allows the i = j case to be calculated but go to 0
      // avoiding a gpu expensive if statement
      GmM = G*masses[i]*masses[j]/((r+0.0000000001)*(r+0.0000000001));
      Fx -= GmM*x/r; Fy -= GmM*y/r; Fz -= GmM*z/r;
  }
  j++;
  for(j;j<i;j++)
  {

      index = 3*i;   jndex = 3*j;   x = position[index] - position[jndex];
      index = 3*i+1; jndex = 3*j+1; y = position[index] - position[jndex];
      index = 3*i+2; jndex = 3*j+2; z = position[index] - position[jndex];

      r = sqrt(x*x+y*y+z*z);
      //The tiny offset allows the i = j case to be calculated but go to 0
      // avoiding a gpu expensive if statement
      GmM = G*masses[i]*masses[j]/((r+0.0000000001)*(r+0.0000000001));
      Fx -= GmM*x/r; Fy -= GmM*y/r; Fz -= GmM*z/r;
      //if(i==5){("%lf",Fx);}
  }
  //if(i==5){printf("\n");}
  //if(i==5){printf("Before: %lf,%lf\n",position[3*i+0],velocity[3*i+0]);
  //         printf("Test:   %lf,%lf",dt,Fx);}
  position[3*i+0] += velocity[3*i+0]*dt+0.5*dt*dt*(Fx/masses[i]);
  velocity[3*i+0] += 0.5*dt*(Fx/masses[i]);
  //if(i==5){printf("After: %lf,%lf\n",position[3*i+0],velocity[3*i+0]);}

  position[3*i+1] += velocity[3*i+1]*dt+0.5*dt*dt*(Fy/masses[i]);
  velocity[3*i+1] += 0.5*dt*(Fy/masses[i]);

  position[3*i+2] += velocity[3*i+2]*dt+0.5*dt*dt*(Fz/masses[i]);
  velocity[3*i+2] += 0.5*dt*(Fz/masses[i]);
}

__kernel void kernel_prog_init_2(
                                  const int N,
                               __global double* position,
                               __global double* velocity,
                               __global double* masses
                               )
{

  int i = get_global_id(0);
  int j;
  double pCoMx=0.0,pCoMy=0.0,pCoMz=0.0;
  double M = 1.0/0.000003003;
  if(i==1){
    for(j=1;j<N;j++){pCoMx+=velocity[3*j+0]*masses[j];
                     pCoMy+=velocity[3*j+1]*masses[j];
                     pCoMz+=velocity[3*j+2]*masses[j];
                    }
  position[0]=position[1]=position[2]=0.0;
  velocity[0]=-pCoMx/M;velocity[1]=-pCoMy/M;velocity[2]=-pCoMz/M;
  masses[0] = M;
  //printf("SUN: %lf,%lf,%lf,%lf\n",position[0],position[1],velocity[0],velocity[1]);
  }
}

__kernel void kernel_prog_init(const double G,
                               const double M,
                               const int N,
                               __global double* position,
                               __global double* velocity,
                               __global double* masses
                               )
{
  double r, r_ast, mass;
  int i = get_global_id(0);

  double v, GM_r;
  double theta;
  double i_step = 10.0/(double)N;

  r_ast = 40.0+i*i_step;

  if(i==0){mass=0;r=1;}//THIIS IS FIXED IN THE SECOND PHASE OF SETTING UP
  if(i==1){mass=0.055;r=0.39;}//Mercury
  if(i==2){mass=0.815;r=0.72;}//Venus  Retrograde motion
  if(i==3){mass=1.0;r=1.0;}   //Earth
  if(i==4){mass=0.107;r=1.52;}
  if(i==5){mass=317.8;r=5.2;}
  if(i==6){mass=95.1;r=9.58;}
  if(i==7){mass=14.5;r=19.2;}
  if(i==8){mass=17.1;r=30.05;}
  if(i >8){mass=0.001;r=r_ast;}

  GM_r = G*M/r;
  v = sqrt(GM_r);

  position[3*i+0]=r;
  position[3*i+1]=0.0;
  position[3*i+2]=0.0;
  //printf("%lf,%lf\n",r,v);
  velocity[3*i+0]=0.0;
  velocity[3*i+1]=v;
  velocity[3*i+2]=0.0;
  masses[i] = mass;
}



__kernel void kernel_prog_Energy(
                                const double G,
                                const int N,
                             __global double* Energy,
                             __global double* position,
                             __global double* velocity,
                             __global double* masses
                             )
{
  int i = get_global_id(0);
  int j,k;
  double r,v,x,y,z,Energy_count;
  v = velocity[3*i+0]*velocity[3*i+0]  +  velocity[3*i+1]*velocity[3*i+1]  +   velocity[3*i+2]*velocity[3*i+2];
  int index, jndex;
  Energy_count = 0.5*masses[i]*v;
  index = 3*i;   x = position[index] - position[0];
  index = 3*i+1; y = position[index] - position[1];
  index = 3*i+2; z = position[index] - position[2];

  r = sqrt(x*x+y*y+z*z);

  for(j=0;j<i;j++)
  {
    index = 3*i;   jndex = 3*j;   x = position[index] - position[jndex];
    index = 3*i+1; jndex = 3*j+1; y = position[index] - position[jndex];
    index = 3*i+2; jndex = 3*j+2; z = position[index] - position[jndex];

    r = sqrt(x*x+y*y+z*z);
    Energy_count -= G*masses[i]*masses[j]/r;
  }

  for(j=i+1;j<N;j++)
  {
    index = 3*i;   jndex = 3*j;   x = position[index] - position[jndex];
    index = 3*i+1; jndex = 3*j+1; y = position[index] - position[jndex];
    index = 3*i+2; jndex = 3*j+2; z = position[index] - position[jndex];

    r = sqrt(x*x+y*y+z*z);
    Energy_count -= G*masses[i]*masses[j]/r;
  }
  Energy[i] = Energy_count;

  /*
  if(i>0){Energy[i] = 0.5*masses[i]*v-2*G*masses[i]*masses[0]/r;}//The suns contribution
  if(i==0){Energy[i] = 0.5*masses[i]*v;}
  printf("E:%lf, ",Energy[i]);*/
}
