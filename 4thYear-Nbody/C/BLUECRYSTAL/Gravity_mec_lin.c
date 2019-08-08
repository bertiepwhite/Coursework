#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define PI 3.14159265358979323846


// #########################   Functions  ###############################

void initPlanetValues(double* position, double* velocity, double* masses, double mass,
                      double rmean, double G, double M, int i)
{
  double theta;
  double v;
  v = sqrt(G*M/rmean);
  theta = (double)rand()/(double)(RAND_MAX/(2*PI));

  masses[i] = mass;

  position[3*i+0] = rmean*sin(theta); position[3*i+1] = rmean*cos(theta);
  position[3*i+2] = 0.0;

  velocity[3*i+0] = -v*cos(theta); velocity[3*i+1] = v*sin(theta);
  velocity[3*i+2] = 0.0;
}
void initKuiperBelt(double* position, double* velocity, double* masses, double mass,
                    double rmin, double rmax, double G, double M, int N)
{
  double r_step = (rmax-rmin)/(N);
  double theta,r,v,z;
  int i;
  for(i=9;i<N;i++)
  {
    masses[i] = mass;
    theta = (double)rand()/(double)(RAND_MAX/(2*PI));
    z = (double)rand()/(double)(RAND_MAX/(10.0))-5.0;
    r = rmin + (i-9)*r_step;
    position[3*i+0] =  r*sin(theta); position[3*i+1] = r*cos(theta);
    position[3*i+2] = z;
    v = sqrt(G*M/sqrt(r*r+z*z));
    velocity[3*i+0] = -v*cos(theta); velocity[3*i+1] = v*sin(theta);
    velocity[3*i+2] = 0.0;
  }
}

void ForceFind(double* position,double* forces,double* masses,
               double G, int N)
{
  double x,y,z,r;
  double GmM,F;
  int i,j;

  for(i=0;i<3*N;i++)
  {
    forces[i] = 0.0;
  }
  for(i=0;i<N;i++)
  {
    for(j=i+1;j<N;j++)
    {


      x = position[3*i+0] - position[3*j+0];
      y = position[3*i+1] - position[3*j+1];
      z = position[3*i+2] - position[3*j+2];
      r = sqrt(x*x + y*y + z*z);


      GmM = G*masses[i]*masses[j];
      F = GmM /(r*r);

      forces[3*i+0] -= F*x/r; forces[3*i+1] -= F*y/r; forces[3*i+2] -= F*z/r;
      forces[3*j+0] += F*x/r; forces[3*j+1] += F*y/r; forces[3*j+2] += F*z/r;
    }
  }
}

void TimeStep(double* position,double* velocity, double* forces,double* masses,
              double dt,int N)
{
  int index,i,k;
  for(i=0;i<N;i++)
  {
    for(k=0;k<3;k++)
    {
      index = 3*i+k;
      velocity[index] += 0.5*dt*(forces[index]/masses[i]);
      position[index] += velocity[index]*dt+0.5*dt*dt*(forces[index]/masses[i]);
      velocity[index] += 0.5*dt*(forces[index]/masses[i]);
    }
  }
}


double EnergyFind(double* position, double* velocity, double* masses, double G,int N)
{
  double Energy = 0.0;
  double r,v;
  int i,j;
  for(i=0;i<N;i++)
  {
    for(j=0;j<N;j++)
    {
      if(i!=j)
      {
        r = 0;
        r += pow((position[3*i+0]-position[3*j+0]),2);
        r += pow((position[3*i+1]-position[3*j+1]),2);
        r += pow((position[3*i+2]-position[3*j+2]),2);
        r = sqrt(r);
        Energy -= G*masses[i]*masses[j]/r;
      }
    }
    v = 0;
    v += pow((velocity[3*i+0]-velocity[3*j+0]),2);
    v += pow((velocity[3*i+1]-velocity[3*j+1]),2);
    v += pow((velocity[3*i+2]-velocity[3*j+2]),2);
    Energy += 0.5*masses[i]*v;
  }
  return Energy;
}


// #########################   MAIN CODE  ###############################


int main( int argc, char *argv[])
{
  //printf("Initialising\n");
  // ----------------INITIALISING ARRAYS AND COSNTANTS------------------
  //Don't save both
  int save = 0;    //Use this for more time steps
  int saveall = 0; //Use this for less time steps
  int saveEnergy = 0;

  int N = atoi(argv[1]);                   // Number of particles
  int N_t = atoi(argv[2]);            // Number of time ste100
  int t_save = 0;
  double G = 0.00011900032;                             //6.67*pow(10,-11)
  double M_solar = 1.0/0.000003003;                     //2.0*pow(10,30)
  double M_earth = 1.0;                                 //0.000003003*M_solar
  double AU = 1.0; //149597870700.0
  double dt = 1.0/(24*60.0);
  int i,t,k;
  double stepE;

  FILE *f4 = fopen("10000000energies.csv", "w");

  int N_t_save = N_t/50000;

  double position[N*3];
  double velocity[N*3];
  double forces[N*3];
  double masses[N];
  double* Energy;
  double* position_output;


  if(save==1)   {position_output = (double*) malloc(3*N*50000*sizeof(double));}
  if(saveall==1){position_output = (double*) malloc(3*N*N_t*sizeof(double));}
  if(saveEnergy==1){Energy = (double*) malloc(N_t*sizeof(double));}

  clock_t start = clock(), diff;
  //printf("Starting\n");




  //-----------------Initial conditions---------------------------


  i=1;
  //Mercury
  initPlanetValues(position,velocity,masses,0.055*M_earth, 0.39*AU,
                  G,M_solar,i); i++;
  //Venus
  initPlanetValues(position,velocity,masses,0.815*M_earth, 0.72*AU,
                   G,M_solar,i); i++;// The odd minus sign here is for
  //Earth                               // the retrograde motion of venus.                                          //Earth
  initPlanetValues(position,velocity,masses,M_earth,AU,
                  G,M_solar,i); i++;
  //Mars
  initPlanetValues(position,velocity,masses,0.107*M_earth,1.52*AU,
                   G,M_solar,i); i++;
  //Jupiter
  initPlanetValues(position,velocity,masses,317.8*M_earth,5.2*AU,
                   G,M_solar,i); i++;
  //Saturn
  initPlanetValues(position,velocity,masses,95.1*M_earth,9.58*AU,
                   G,M_solar,i); i++;
  //Uranus
  initPlanetValues(position,velocity,masses,14.5*M_earth,19.2*AU,
                   G,M_solar,i); i++;
  //Neptune
  initPlanetValues(position,velocity,masses,17.1*M_earth,30.05*AU,
                   G,M_solar,i); i++;
  initKuiperBelt(position,velocity,masses,0.001*M_earth,40*AU, 50*AU,
                 G,M_solar,N);

  double CoMvx=0.0,CoMvy=0.0,CoMvz=0.0;

  for(i=1;i<N;i++)
  {
    CoMvx += velocity[3*i+0]*masses[i];
    CoMvy += velocity[3*i+1]*masses[i];
    CoMvz += velocity[3*i+2]*masses[i];
  }

  //Sun
  masses[0] = M_solar; position[0] = 0.0; position[1] = 0.0; position[2] = 0.0;
  velocity[0] = -CoMvx/M_solar; velocity[1] = -CoMvy/M_solar;
  velocity[2] = -CoMvz/M_solar;

  ForceFind(position,forces,masses,dt,N);
  for(i=0;i<N;i++)
  {
    for(k=0;k<3;k++)
    {
      position[3*i+k] += velocity[3*i+k]*dt+0.5*dt*dt*(forces[3*i+k]/masses[i]);
      velocity[3*i+k] += 0.5*dt*(forces[3*i+k]/masses[i]);
    }
  }

  // ----------------------MAIN ITTERRATION---------------------------

  //printf("Iterating\n");
  for(t=0;t<N_t;t++)
  {
    //Saving into files
    //if(t%100==0){printf("Iteration: %d/%d\n", t, N_t);}
    if(save==1&&t%N_t_save==0){
      for(i=0;i<3*N;i++){position_output[t_save*3*N+i]=position[i];}
      t_save++;
    }
    if(saveall==1){
      for(i=0;i<3*N;i++){position_output[t_save*3*N+i]=position[i];}
      t_save++;
    }
    if(saveEnergy==1){
      stepE = EnergyFind(position,velocity,masses,G,N);
      Energy[t] = stepE;
    }

    ForceFind(position,forces,masses,G,N);
    TimeStep(position,velocity,forces,masses,dt,N);

  }


  diff = clock()-start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds, for N: %d\n", msec/1000, msec%1000,N);

  if(save==1)
  {
    //printf("Saving Full\n");
    FILE *f3 = fopen("Gravity_output_full.csv", "w");

    for(i=0;i<3*N*50000;i++)
      {fprintf(f3,"%lf\n",position_output[i]);}

    fclose(f3);
  }
  if(saveall==1)
  {
    //printf("Saving Full\n");
    FILE *f3 = fopen("Gravity_output_full.csv", "w");

    for(i=0;i<3*N*N_t;i++)
      {fprintf(f3,"%lf\n",position_output[i]);}

    fclose(f3);
  }
  if(saveEnergy==1)
  {
    //printf("Saving Full\n");


    for(t=0;t<N_t;t++)
      {fprintf(f4,"%lf\n",Energy[t]);}

    fclose(f4);
  }

}
