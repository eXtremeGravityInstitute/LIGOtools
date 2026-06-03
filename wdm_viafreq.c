#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Constants.h"

#include <time.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_spline.h>

#define TPI 6.2831853071795862319959269370884     // 2 Pi
#define SQPI 2.5066282746310002  // sqrt(TPI)
#define AD_SQRT_2 1.41421356237309504880
#define AD_SQRT_2PI 2.50662827463100024161

double anderson_darling_known_normal_statistic(
    const double *x,
    int nsamples,
    double mean,
    double variance);

double anderson_darling_known_normal_pvalue_from_statistic(double a2);

double anderson_darling_known_normal_pvalue(
    const double *x,
    int nsamples,
    double mean,
    double variance,
    double *a2_out);

// Holds fit to AD to pvalue mapping
double AV[101] = {0.00 ,0.02 ,0.04 ,0.06 ,0.08 ,0.10 ,0.12 ,0.14 ,0.16 ,0.18 ,0.20 ,0.22 ,0.24 ,0.26 ,0.28 ,0.30 ,0.32 ,0.34 ,0.36 ,0.38 ,0.40 ,0.42 ,0.44 ,0.46 ,0.48 ,0.50 ,0.52 ,0.54 ,0.56 ,0.58 ,0.60 ,0.62 ,0.64 ,0.66 ,0.68 ,0.70 ,0.72 ,0.74 ,0.76 ,0.78 ,0.80 ,0.82 ,0.84 ,0.86 ,0.88 ,0.90 ,0.92 ,0.94 ,0.96 ,0.98 ,1.00 ,1.02 ,1.04 ,1.06 ,1.08 ,1.10 ,1.12 ,1.14 ,1.16 ,1.18 ,1.20 ,1.22 ,1.24 ,1.26 ,1.28 ,1.30 ,1.32 ,1.34 ,1.36 ,1.38 ,1.40 ,1.42 ,1.44 ,1.46 ,1.48 ,1.50 ,1.52 ,1.54 ,1.56 ,1.58 ,1.60 ,1.62 ,1.64 ,1.66 ,1.68 ,1.70 ,1.72 ,1.74 ,1.76 ,1.78 ,1.80 ,1.82 ,1.84 ,1.86 ,1.88 ,1.90 ,1.92 ,1.94 ,1.96 ,1.98 ,1.999800};
double ADC[101] = {0.000000e+00 ,0.000000e+00 ,0.000000e+00 ,-6.000002e-07 ,-6.590217e-05 ,-9.718721e-04 ,-5.347673e-03 ,-1.702806e-02 ,-3.928110e-02 ,-7.370628e-02 ,-1.203567e-01 ,-1.783287e-01 ,-2.462505e-01 ,-3.227490e-01 ,-4.063929e-01 ,-4.960332e-01 ,-5.905115e-01 ,-6.890052e-01 ,-7.908381e-01 ,-8.952793e-01 ,-1.001889e+00 ,-1.110290e+00 ,-1.220061e+00 ,-1.331033e+00 ,-1.442868e+00 ,-1.555449e+00 ,-1.668667e+00 ,-1.782282e+00 ,-1.896382e+00 ,-2.010684e+00 ,-2.125117e+00 ,-2.239706e+00 ,-2.354675e+00 ,-2.469649e+00 ,-2.584469e+00 ,-2.699301e+00 ,-2.814132e+00 ,-2.929191e+00 ,-3.043736e+00 ,-3.158495e+00 ,-3.273014e+00 ,-3.387418e+00 ,-3.501725e+00 ,-3.616137e+00 ,-3.729707e+00 ,-3.843611e+00 ,-3.957667e+00 ,-4.072023e+00 ,-4.185272e+00 ,-4.298510e+00 ,-4.412142e+00 ,-4.525421e+00 ,-4.639168e+00 ,-4.752583e+00 ,-4.866879e+00 ,-4.979522e+00 ,-5.091913e+00 ,-5.205259e+00 ,-5.318487e+00 ,-5.430978e+00 ,-5.543193e+00 ,-5.655833e+00 ,-5.767883e+00 ,-5.880059e+00 ,-5.992341e+00 ,-6.104338e+00 ,-6.216144e+00 ,-6.327318e+00 ,-6.440462e+00 ,-6.552048e+00 ,-6.663304e+00 ,-6.776420e+00 ,-6.888600e+00 ,-7.001418e+00 ,-7.113071e+00 ,-7.226144e+00 ,-7.336909e+00 ,-7.445952e+00 ,-7.556292e+00 ,-7.668668e+00 ,-7.783096e+00 ,-7.896697e+00 ,-8.007098e+00 ,-8.114231e+00 ,-8.227214e+00 ,-8.335705e+00 ,-8.450189e+00 ,-8.558953e+00 ,-8.666854e+00 ,-8.777520e+00 ,-8.883918e+00 ,-8.994020e+00 ,-9.104360e+00 ,-9.212643e+00 ,-9.318371e+00 ,-9.425640e+00 ,-9.532062e+00 ,-9.645749e+00 ,-9.756448e+00 ,-9.868507e+00 ,-9.987087e+00};

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

int *int_vector(int N);
void free_int_vector(int *v);
double *double_vector(int N);
void free_double_vector(double *v);
double **double_matrix(int N, int M);
void free_double_matrix(double **m, int N);
double ***double_tensor(int N, int M, int L);
void free_double_tensor(double ***t, int N, int M);
double phitilde(double om, double insDOM, double A, double B);
void wavelet(int m, double *waveletE, double *waveletO, int N, double nrm, double dom, double DOM, double A, double B, double insDOM);
void tukey(double *data, double alpha, int N);
void bwbpf(double *in, double *out, int fwrv, int M, int n, double s, double f1, double f2);

// gcc -I/opt/homebrew/include -L/opt/homebrew/lib -o wdm_viafreq wdm_viafreq.c -lm -lgslcblas -lgsl

#define nx 6.0    // filter steepness in frequency
#define mult 16  // over sampling
#define Bfrac 1.0  // fall-off region for frequency filter
#define PS_TIME_WINDOW 11  // Gaussian smoothing window width in time pixels
#define PS_FREQ_WINDOW 11  // Gaussian smoothing window width in frequency pixels
#define PS_TIME_SIGMA 3.0 // Gaussian smoothing sigma in time pixels
#define PS_FREQ_SIGMA 3.0 // Gaussian smoothing sigma in frequency pixels

// window should be 3*sigma+1

int main(int argc, char *argv[])
{
  int i, j, k, l, ii, jj, kk, tf;
    int n, m;
    int N;
  char filename[1024];
  double Tobs, Tchunk, df;
    double fskip = 20.0;  // skip these bins
    double DT, DF;
    double x, y, z, alpha;
    double c, s;
    double xx, yy;
    double A, B, DOM, OM;
    double insDOM;
    double om, fac;
    double f0;
    double *R;
    double *DX;
    double *data, *time, *freq;
    double *wdata;
    double **wave, **wvhat;
    
    double *waveletE, *waveletO;
    
    int K, NC, M,  L, up;
    int Nx, Mx;
    double f, t, T, dom, nrm;
    double **data1, **data2;
    double *hist;
    double *phif;
    int *tim;
    
    double dt;
    int Nt, Nf;
    
    const gsl_rng_type * P;
    gsl_rng * r;

    gsl_rng_env_setup();
    
    P = gsl_rng_default;
    r = gsl_rng_alloc (P);
    
    clock_t start, end;
    double cpu_time_used;

  FILE *in;
  FILE *ifp;
  FILE *out;
    
    if(argc<3)
    {
        printf("./wdm_viafreq filename DF time/freq\n");
        return 1;
    }

    if((PS_TIME_WINDOW % 2) == 0 || (PS_FREQ_WINDOW % 2) == 0)
    {
        printf("PS_TIME_WINDOW and PS_FREQ_WINDOW must be odd\n");
        return 1;
    }
    
    in = fopen(argv[1],"r");
    
    DF = atof(argv[2]);
    
    tf = atoi(argv[3]); // 0 for TD, 1 for FD
   
    if(tf == 0)
    {
       N = -1;
       while(!feof(in))
       {
           fscanf(in,"%lf%lf", &x, &y);
           N++;
       }
       rewind(in);
        
        // the data stream
    data = (double*)malloc(sizeof(double)* (N));
    time = (double*)malloc(sizeof(double)* (N));
    for(i=0; i< N; i++) fscanf(in,"%lf%lf", &time[i], &data[i]);
        
    dt = time[1]-time[0];
    
    Tobs = dt*(double)(N);
    }
    else
    {
              N = 0;
              while(!feof(in))
              {
                  fscanf(in,"%lf%lf%lf", &x, &y, &z);
                  N++;
              }
              rewind(in);
        N = N*2;
        
        // the data stream
        data = (double*)malloc(sizeof(double)* (N));
        freq = (double*)malloc(sizeof(double)* (N/2));
        for(i=1; i< N/2; i++) fscanf(in,"%lf%lf%lf", &freq[i], &data[i], &data[N-i]);
        Tobs = 1.0/freq[1];
        dt = Tobs/(double)(N);
    }
    
    k = (int)(Tobs*8.0);
    x = 0.0;
    for(i=1; i< k; i++)
    {
        x += data[i]*data[i];
    }
    x /= (double)(k);
    
    j = (int)(Tobs*100.0);
    y = 0.0;
    for(i=j; i< j+k; i++)
    {
        y += data[i]*data[i];
    }
    y /= (double)(k);
    
    printf("%f %f\n", x, y);
    
    
    Nf = (int)(1.0/(2.0*dt*DF));
    Nt = N/Nf;
    
    printf("Tobs %f N %d Nf %d Nt %d\n", Tobs, N, Nf, Nt);
  
    DT = dt*(double)(Nf);           // width of wavelet pixel in time
    
    printf("DF %f DT %f\n", DF, DT);
    
    OM = M_PI/dt;
    
    M = Nf;
    
    L = 2*M;
    
    DOM = OM/(double)(M);
    
    insDOM = 1.0/sqrt(DOM);
    
    B = Bfrac*DOM;
    
    A = (DOM-B)/2.0;
    
    dom = 2.0*M_PI/Tobs;
    
    printf("full filter bandwidth %e\n", (A+B)/M_PI);
    
    phif = (double*)malloc(sizeof(double)* (Nt/2+1));

    for(i=0; i<= Nt/2; i++)
    {
        om = (double)(i)*dom;
        phif[i] = phitilde(om, insDOM, A, B);
    }

    
    wave = double_matrix(Nt,Nf);  // wavelet wavepacket transform of the signal
    wvhat = double_matrix(Nt,Nf);  // wavelet wavepacket transform of the signal (quadrature)
    
    DX = double_vector(2*Nt);
    
    // Window the data and FFT
    // Tukey window parameter. Flat for (1-alpha) of data
    alpha = (2.0*(4.0*DT)/Tobs);
    
    start = clock();
    
    if(tf == 0)
    {
    tukey(data, alpha, N);
    gsl_fft_real_radix2_transform(data, 1, N);
        
        /*
        out = fopen("test_f.dat","w");
        for(i=1; i< N/2; i++)
        {
            fprintf(out,"%e %e %e\n", (double)(i)/Tobs, data[i],  data[N-i]);
        }
        fclose(out);
        */
    }

    for(m=0; m< Nf; m++)
     {
         
         
        for(j=-Nt/2; j< Nt/2; j++)
        {
            i = j+Nt/2;
            
            REAL(DX,i) = 0.0;
            IMAG(DX,i) = 0.0;
            
            jj = j + m*Nt/2;
            
            if(jj > 0 && jj < N/2)
            {
                REAL(DX,i) = data[jj]*phif[abs(j)];
                IMAG(DX,i) = data[N-jj]*phif[abs(j)];
            }
        }
         
         gsl_fft_complex_radix2_backward(DX, 1, Nt);
         
        
         for(n=0; n < Nt; n++)
         {
             
             if(m%2 == 0)
             {
                 
                 if((n+m)%2 ==0)
                 {
                      wave[n][m] = REAL(DX,n);
                      wvhat[n][m] = IMAG(DX,n);
                 }
                 else
                 {
                    wave[n][m] = IMAG(DX,n);
                    wvhat[n][m] = REAL(DX,n);
                 }
                 
             }
             else
             {
                 if((n+m)%2 ==0)
                 {
                      wave[n][m] = REAL(DX,n);
                      wvhat[n][m] = -IMAG(DX,n);
                 }
                 else
                 {
                    wave[n][m] = -IMAG(DX,n);
                    wvhat[n][m] = REAL(DX,n);
                 }
                 
             }
             
            
         }
         
       }
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("The transform took %f seconds\n", cpu_time_used);
    
    xx = 0.0;
    yy = 0.0;
    
    double *medpow;
    
    medpow = double_vector(Nf*Nt);
    
    k = 0;
    
       for(i=1; i< 2; i++)
       {
          for(j=0; j< Nt; j++)
           {
          
               if( j > mult && j < Nt-mult-1)
               {
                   xx += wave[j][i];
                   yy += wave[j][i]*wave[j][i];
                   medpow[k] = wave[j][i]*wave[j][i];
                   k++;
               }
               
           }
     }
    
    xx /= (double)(k);
    yy /= (double)(k);
    
    yy = sqrt(yy-xx*xx);
    
    gsl_sort(medpow, 1, k);
    z = sqrt(gsl_stats_median_from_sorted_data(medpow, 1, k)/0.454936);
    // Chi-squared with one dof. 0.454936 is the mapping from median to mean
    
    printf("mean = %e sigma = %e %e\n", xx, yy, z);
    
    // generate N(0,1) data for testing
    /*
    for(i=0; i< Nf; i++)
    {
        for(j=0; j< Nt; j++) wave[j][i] = gsl_ran_gaussian(r,1.0);
    }
     */
    
    k = 0;
    
       for(i=0; i< Nf; i++)
       {
          for(j=0; j< Nt; j++)
           {
          
               if((double)(i)*DF > fskip && j > mult && j < Nt-mult-1)
               {
                   xx += wave[j][i];
                   yy += wave[j][i]*wave[j][i];
                   medpow[k] = wave[j][i]*wave[j][i];
                   k++;
               }
               
           }
     }
    
    xx /= (double)(k);
    yy /= (double)(k);
    
    yy = sqrt(yy-xx*xx);
    
    gsl_sort(medpow, 1, k);
    z = sqrt(gsl_stats_median_from_sorted_data(medpow, 1, k)/0.454936);
    // Chi-squared with one dof. 0.454936 is the mapping from median to mean
    
    printf("mean = %e sigma = %e %e\n", xx, yy, z);
    
    
    // scale to unit variance (accounts for normalization issues)
    // Chi-squared with one dof.
    for(i=0; i< Nf; i++)
    {
        for(j=0; j< Nt; j++)
        {
            wave[j][i] /= z;
        }
    }

    out = fopen("BinaryF.dat","w");

    xx = 0.0;
    yy = 0.0;
    
    x = 0.0;
    y = 0.0;
    k = 0;
       for(i=0; i< Nf; i++)
       {
          for(j=0; j< Nt; j++)
           {
           fprintf(out, "%e %e %.14e\n", (double)(j)*DT, (double)(i)*DF, wave[j][i]);
            //fprintf(out, "%d %d %.14e\n", j, i, wave[j][i]);
               if((double)(i)*DF > 16.0 && j > mult && j < Nt-mult-1)
               {
                   z = fabs(wave[j][i]);
                   if(z > y)
                   {
                       n = j;
                       m = i;
                       y = z;
                   }
                   
                   xx += wave[j][i];
                   yy += wave[j][i]*wave[j][i];
                   k++;
               }
               
           }
               
       fprintf(out, "\n");
               
     }
    fclose(out);
    
    xx /= (double)(k);
    yy /= (double)(k);
    
    printf("mean = %e sigma = %e\n", xx, sqrt(yy-xx*xx));
    
    printf("max %e %d %d\n", y, n, m);
    
    x = pow(10.0,floor(log10(y)));
    
    z = y/x;
    z = ceil(z)*x;
    
    // use 3 sigma for the color range
    z = 3.0*sqrt(yy-xx*xx);
    printf("%e\n", z);
    
    
    out = fopen("tranf.gnu","w");
    fprintf(out,"set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n");
    fprintf(out,"set output 'tranf.png'\n");
     fprintf(out,"set ylabel 'f (Hz)'\n");
    fprintf(out,"set xlabel 't (s)'\n");
    fprintf(out,"set yrange [%e:%e]\n", fskip, 1.0/(2.0*dt));
    fprintf(out,"set ytics (20,200,400,600,800,1000)\n");
    fprintf(out,"set xrange [%e:%e]\n", (double)(mult)*DT,Tobs-(double)(mult)*DT);
    fprintf(out,"set cbrange [%e:%e]\n", -3.0, 3.0);
    fprintf(out,"set pm3d map corners2color c1\n");
    fprintf(out,"set palette defined (0 '#2166ac', 1 '#67a9cf', 2 '#d1e5f0', 3 '#ffffff', 4 '#fddbc7', 5 '#ef8a62', 6 '#b2182b')\n");
    fprintf(out,"splot 'BinaryF.dat' using 1:2:3 notitle\n");
    fclose(out);
    
    system("gnuplot tranf.gnu");
    system("open tranf.png");
    
    // compute the power variation, summed accross frequencies, for each time
    
    double *powt, *pown;
    
    ii = (int)(ceil(fskip/DF)); // start at fskip Hz
    kk = Nf-ii;
    
    powt = double_vector(Nt);
    pown = double_vector(Nt);
   
    
    z = 0.0;
    for(j=0; j< Nt; j++)
      {
          powt[j] = 0.0;
        for(i=ii; i< Nf; i++)
         {
             powt[j] += wave[j][i]*wave[j][i];
         }
          powt[j] /= (double)(kk);
          //printf("%d %f\n", j, powt[j]);
      }
    
    
    // power has a mean of 1 and a variance of 2 since it is a Chi-squared with 1 dof
    // Thus variance of the mean, will be 2/kk;
    
    printf("%d %e\n", kk, sqrt(2.0/(double)(kk)));
    
    // Wilson-Hilferty transform
    for(j=0; j< Nt; j++) pown[j] = (pow(powt[j],1.0/3.0)-(1.0-2.0/((double)(9*kk))))/sqrt(2.0/((double)(9*kk)));
    
    out = fopen("powertime.dat","w");
    for(j=0; j< Nt; j++) fprintf(out,"%e %e %e\n", ((double)(j)+0.5)*DT, powt[j], pown[j]);
    fclose(out);
    
    // now we average the power across all frequencies, and across 8 time pixels
    printf("power average computed across %f seconds\n", 8.0*DT);
    
    k = -1;
    for(j=0; j< Nt; j++)
      {
        if((j%8) == 0)
        {
            k++;
            powt[k] = 0.0;
        }
        for(i=ii; i< Nf; i++)
         {
             powt[k] += wave[j][i]*wave[j][i];
         }
        if(((j+1)%8) == 0) powt[k] /= (double)(8*kk);
      }
    
    z = sqrt(2.0/(double)(kk*8));
    out = fopen("powertimeav.dat","w");
    for(j=0; j<= k; j++) fprintf(out,"%e %e %e %e %e %e %e %e\n", ((double)(j)+0.5)*8.0*DT, powt[j], 1.0-z,1.0+z, 1.0-2.0*z, 1.0+2.0*z, 1.0-3.0*z,1.0+3.0*z);
    fclose(out);

    
    out = fopen("BinaryP.dat","w");

      x = 0.0;
      y = 0.0;
       for(i=0; i< Nf; i++)
       {
          for(j=0; j< Nt; j++)
           {
           fprintf(out, "%e %e %.14e\n", (double)(j)*DT, (double)(i)*DF, wave[j][i]*wave[j][i]);

           }
               
       fprintf(out, "\n");
               
     }
    fclose(out);

    double **spow;
    double weight, weight_sum;
    int t_radius, f_radius;

    spow = double_matrix(Nt,Nf);
    t_radius = PS_TIME_WINDOW/2;
    f_radius = PS_FREQ_WINDOW/2;
    
    x = 0.0;
    y = 0.0;

    for(i=0; i< Nf; i++)
    {
       for(j=0; j< Nt; j++)
        {
            spow[j][i] = 0.0;
            weight_sum = 0.0;

            for(ii=-f_radius; ii<=f_radius; ii++)
            {
                if(i+ii < 0 || i+ii >= Nf) continue;

                for(jj=-t_radius; jj<=t_radius; jj++)
                {
                    if(j+jj < 0 || j+jj >= Nt) continue;

                    weight = exp(-0.5*((double)(jj*jj)/(PS_TIME_SIGMA*PS_TIME_SIGMA)
                                      + (double)(ii*ii)/(PS_FREQ_SIGMA*PS_FREQ_SIGMA)));
                    spow[j][i] += weight*wave[j+jj][i+ii]*wave[j+jj][i+ii];
                    weight_sum += weight;
                }
            }

            spow[j][i] /= weight_sum;
            
            x += spow[j][i];
            y += spow[j][i]*spow[j][i];
            
            
        }
    }
    
    x /= (double)(Nt*Nf);
    y /= (double)(Nt*Nf);
    
    printf("smoothed power mean %f and variance %f sigma %f\n", x, y-x*x, sqrt(y-x*x));

    out = fopen("BinaryPS.dat","w");

       for(i=0; i< Nf; i++)
       {
          for(j=0; j< Nt; j++)
           {
               fprintf(out, "%e %e %.14e\n", (double)(j)*DT, (double)(i)*DF, spow[j][i]-1.0);
           }

       fprintf(out, "\n");

     }
    fclose(out);
    
    x = 9.0; // chi-squared with 1 DOF 99.73% (3-sigma for Gaussian)
    
    out = fopen("tranP.gnu","w");
    fprintf(out,"set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n");
    fprintf(out,"set output 'tranP.png'\n");
    fprintf(out,"unset xtics\n");
    fprintf(out,"set ylabel 's'\n");
    fprintf(out,"set multiplot\n");
    fprintf(out,"set size 0.76, 0.3\n");
    fprintf(out,"set yrange [-4:4]\n");
    fprintf(out,"set ytics (-3,0,3)\n");
    fprintf(out,"set xrange [%e:%e]\n", (double)(mult)*DT,Tobs-(double)(mult)*DT);
    fprintf(out,"set origin 0.031, 0.5\n");
    fprintf(out,"plot 'powertime.dat' using 1:3 notitle with lines\n");
    fprintf(out,"set size 0.95, 0.62\n");
    fprintf(out,"set origin 0.0, 0.0\n");
    fprintf(out,"unset ytics\n");
    fprintf(out,"set ytics (20,200,400,600,800,1000)\n");
    fprintf(out,"set xtics\n");
    fprintf(out,"set pm3d map corners2color c1\n");
    fprintf(out,"set ylabel 'f (Hz)'\n");
    fprintf(out,"set xlabel 't (s)'\n");
    fprintf(out,"set yrange [%e:%e]\n", fskip, 1.0/(2.0*dt));
    fprintf(out,"set cbrange [%e:%e]\n", 0.0, x);
    fprintf(out,"set palette defined (0 '#fff7ec', 1'#fee8c8', 2 '#fdd49e', 3 '#fdbb84', 4 '#fc8d59', 5 '#ef6548', 6 '#d7301f',7 '#990000')\n");
    fprintf(out,"splot 'BinaryP.dat' using 1:2:3 notitle\n");
    fclose(out);
    
    system("gnuplot tranP.gnu");
    system("open tranP.png");

    out = fopen("tranPS.gnu","w");
    fprintf(out,"set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n");
    fprintf(out,"set output 'tranPS.png'\n");
    fprintf(out,"set pm3d map corners2color c1\n");
    fprintf(out,"set ylabel 'f (Hz)'\n");
    fprintf(out,"set xlabel 't (s)'\n");
    fprintf(out,"set yrange [%e:%e]\n", fskip, 1.0/(2.0*dt));
    fprintf(out,"set ytics (20,200,400,600,800,1000)\n");
    fprintf(out,"set xrange [%e:%e]\n", (double)(mult)*DT,Tobs-(double)(mult)*DT);
    fprintf(out,"set cbrange [-0.7:0.7]\n");
    fprintf(out,"set palette defined (0 '#2166ac', 1 '#67a9cf', 2 '#d1e5f0', 3 '#ffffff', 4 '#fddbc7', 5 '#ef8a62', 6 '#b2182b')\n");
    fprintf(out,"splot 'BinaryPS.dat' using 1:2:3 notitle\n");
    fclose(out);

    system("gnuplot tranPS.gnu");
    system("open tranPS.png");
    
    x = 1.0-4.0*z;
    y = 1.0+4.0*z;
    
    out = fopen("power.gnu","w");
    fprintf(out,"set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n");
    fprintf(out,"set output 'power.png'\n");
    fprintf(out,"unset xtics\n");
    fprintf(out,"set ylabel 'P'\n");
    fprintf(out,"set multiplot\n");
    fprintf(out,"set size 0.782, 0.3\n");
    fprintf(out,"set yrange [%f:%f]\n", x, y);
    fprintf(out,"set ytics (0.9,1,1.1)\n");
    fprintf(out,"set xrange [%e:%e]\n", (double)(mult)*DT,Tobs-(double)(mult)*DT);
    fprintf(out,"set origin 0.008, 0.5\n");
    fprintf(out,"plot 'powertimeav.dat' using 1:7:8 notitle with filledcurves lc 'skyblue' fs transparent solid 0.2, 'powertimeav.dat' using 1:5:6 notitle with filledcurves lc 'skyblue' fs transparent solid 0.4, 'powertimeav.dat' using 1:3:4 notitle with filledcurves lc 'skyblue' fs transparent solid 0.6, 'powertimeav.dat' using 1:(1.0) notitle with lines lc rgb 'blue', 'powertimeav.dat' using 1:2 notitle with lines lt -1\n");
    fprintf(out,"set size 0.95, 0.62\n");
    fprintf(out,"set origin 0.0, 0.0\n");
    fprintf(out,"unset ytics\n");
    fprintf(out,"set ytics (20,200,400,600,800,1000)\n");
    fprintf(out,"set xtics\n");
    fprintf(out,"set pm3d map corners2color c1\n");
    fprintf(out,"set ylabel 'f (Hz)'\n");
    fprintf(out,"set xlabel 't (s)'\n");
    fprintf(out,"set yrange [%e:%e]\n", fskip, 1.0/(2.0*dt));
    fprintf(out,"set cbrange [%e:%e]\n", 0.0, 9.0);
    fprintf(out,"set palette defined (0 '#fff7ec', 1'#fee8c8', 2 '#fdd49e', 3 '#fdbb84', 4 '#fc8d59', 5 '#ef6548', 6 '#d7301f',7 '#990000')\n");
    fprintf(out,"splot 'BinaryP.dat' using 1:2:3 notitle\n");
    fclose(out);
    
    system("gnuplot power.gnu");
    system("open power.png");

    free_double_matrix(spow,Nt);
    

    
    
    // now compute variance and AD statistic. Have to choose size of smoothing region in time
    // and frequency
    
    int FS, TS;
    int Nfs, Nts, Nsamp;
    
    FS=4;
    TS=8;

    
    Nfs = Nf/FS;
    Nts = Nt/TS;
    
    Nsamp = FS*TS;  // samples in each block
    
    double *phist;
    int pp;
    phist = double_vector(100);
    for(i=0; i< 100; i++) phist[i] = 0.0;
    
    printf("Test blocks %d Samples per block %d\n", Nts*Nfs, Nsamp);
    
    double **vars;
    double **ads;
    double *samples;
    double mean, var, std;
    
    double S, lx, ly, u, p;
    
    vars = double_matrix(Nts,Nfs);  // variance for each block
    ads = double_matrix(Nts,Nfs);  // AD p-values for each block
    
    // spline fit for the AD to p value mapping
    int NS = 101;
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, NS);
    gsl_spline_init(spline, AV, ADC, NS);
    
    samples=double_vector(Nsamp);
    
    for(i=0; i< Nfs; i++)
    {
       for(j=0; j< Nts; j++)
        {
            k = 0;
            
            for(ii=0; ii< FS; ii++)
            {
                for(jj=0; jj< TS; jj++)
                {
                    samples[k] = wave[j*TS+jj][i*FS+ii];
                    k++;
                }
            }
            
            mean = 0.0;
            var = 0.0;
            
            for(n=0; n<Nsamp; n++)
            {
                mean += samples[n];
                var += samples[n]*samples[n];
            }
            
            mean /= (double)(Nsamp);
            var /= (double)(Nsamp-1);
            var -= mean*mean;
            std = sqrt(var);
            
            vars[j][i] = var;
            
            // make the samples N(0,1)
            // No longer standadizing each block. instead apply AD test for known
            // mean and variance
            /*for(n=0; n<Nsamp; n++)
            {
                samples[n] -= mean;
                samples[n] /= std;
            } */
            
            p = anderson_darling_known_normal_pvalue(samples,Nsamp,0.0,1.0,&A);
            
            pp = (int)(p*100.0);
            phist[pp] += 1.0;
            
            
            if(p < 1.0e-5) p = 1.0e-5; // makes sure they all show up on the plot
            
           // printf("%e %e\n", A, p);
            
            
            ads[j][i] = -log10(p);
            
        }
    }
    
    for(i=0; i< 100; i++) phist[i] /= (double)(Nts*Nfs);
    x = 1.0/sqrt((double)(Nts*Nfs)/100.0);
    
    out = fopen("phist.dat","w");
    for(i=0; i< 100; i++)
    {
           fprintf(out, "%e %e %e %e %e %e %e %e\n", ((double)(i)+0.5)/100.0, phist[i], 1.0e-2*(1.0-x), 1.0e-2*(1.0+x), 1.0e-2*(1.0-2.0*x), 1.0e-2*(1.0+2.0*x), 1.0e-2*(1.0-3.0*x), 1.0e-2*(1.0+3.0*x));
     }
    fclose(out);
    
    out = fopen("phist.gnu","w");
    fprintf(out,"set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n");
    fprintf(out,"set output 'phist.png'\n");
    fprintf(out,"set yrange [%f:%f]\n", 0.0, 1.0e-2*(1.0+6.0*x));
    fprintf(out,"set xrange [0:1]\n");
    fprintf(out,"set xlabel 'p-value'\n");
    fprintf(out,"plot 'phist.dat' using 1:7:8 notitle with filledcurves lc 'skyblue' fs transparent solid 0.2, 'phist.dat' using 1:5:6 notitle with filledcurves lc 'skyblue' fs transparent solid 0.4, 'phist.dat' using 1:3:4 notitle with filledcurves lc 'skyblue' fs transparent solid 0.6, 'phist.dat' using 1:(0.01) notitle with lines lc rgb 'blue', 'phist.dat' using 1:2 notitle with histeps lt -1\n");
    fclose(out);
    system("gnuplot phist.gnu");
    system("open phist.png");
    
    
    out = fopen("variances.dat","w");
       for(i=0; i< Nfs; i++)
       {
          for(j=0; j< Nts; j++)
           {
           fprintf(out, "%e %e %.14e\n", ((double)(TS)*DT)*((double)(j)+0.5), ((double)(FS)*DF)*((double)(i)+0.5), vars[j][i]);
           }
       fprintf(out, "\n");
     }
    fclose(out);
    
    out = fopen("pvalues.dat","w");
       for(i=0; i< Nfs; i++)
       {
          for(j=0; j< Nts; j++)
           {
           fprintf(out, "%e %e %.14e\n", (double)(TS)*((double)(j)+0.5)*DT, ((double)(FS)*DF)*((double)(i)+0.5), ads[j][i]);
           }
       fprintf(out, "\n");
     }
    fclose(out);
    
    double uv,vlow,vhigh;
    uv = sqrt(2.0/(double)Nsamp);  // 1-sigma range about mean for sample variance
    vlow = -3.0*uv;
    vhigh = 3.0*uv;
    
    // the lowest bin is not trustworthy since we filled the arrays below 10 Hz with Gaussian noise in the PSD code
    x = ((double)(FS)*DF)*1.5;
    
    out = fopen("variances.gnu","w");
    fprintf(out,"set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n");
    fprintf(out,"set output 'variances.png'\n");
    fprintf(out,"set pm3d map corners2color c1\n");
     fprintf(out,"set ylabel 'f (Hz)'\n");
    fprintf(out,"set xlabel 't (s)'\n");
    fprintf(out,"set yrange [%e:%e]\n", fskip, 1.0/(2.0*dt));
    fprintf(out,"set ytics (20,200,400,600,800,1000)\n");
    fprintf(out,"set xrange [%e:%e]\n", 1.5, Tobs-1.5);
    fprintf(out,"set cbrange [%e:%e]\n", vlow, vhigh);
    fprintf(out,"set palette defined (0 '#2166ac', 1 '#67a9cf', 2 '#d1e5f0', 3 '#ffffff', 4 '#fddbc7', 5 '#ef8a62', 6 '#b2182b')\n");
    fprintf(out,"splot 'variances.dat' using 1:2:($3-1.0) notitle\n");
    fclose(out);
    
    system("gnuplot variances.gnu");
    system("open variances.png");
    
    out = fopen("pvalues.gnu","w");
    fprintf(out,"set term png enhanced truecolor crop font Helvetica 18  size 1200,800\n");
    fprintf(out,"set output 'pvalues.png'\n");
    fprintf(out,"set pm3d map corners2color c1\n");
     fprintf(out,"set ylabel 'f (Hz)'\n");
    fprintf(out,"set xlabel 't (s)'\n");
    fprintf(out,"set yrange [%e:%e]\n", fskip, 1.0/(2.0*dt));
    fprintf(out,"set ytics (20,200,400,600,800,1000)\n");
    fprintf(out,"set xrange [%e:%e]\n", 1.5, Tobs-1.5);
    fprintf(out,"set cbrange [%e:%e]\n", 0.0, 3.0);
    fprintf(out,"set palette defined (0 '#fff7ec', 1'#fee8c8', 2 '#fdd49e', 3 '#fdbb84', 4 '#fc8d59', 5 '#ef6548', 6 '#d7301f',7 '#990000')\n");
    fprintf(out,"splot 'pvalues.dat' using 1:2:3 notitle\n");
    fclose(out);
    
    system("gnuplot pvalues.gnu");
    system("open pvalues.png");
    
    
    return 0;

}

void tukey(double *data, double alpha, int N)
{
  int i, imin, imax;
  double filter;
  
  imin = (int)(alpha*(double)(N-1)/2.0);
  imax = (int)((double)(N-1)*(1.0-alpha/2.0));
  
    int Nwin = N-imax;

    for(i=0; i< N; i++)
  {
    filter = 1.0;
    if(i<imin) filter = 0.5*(1.0+cos(M_PI*( (double)(i)/(double)(imin)-1.0 )));
    if(i>imax) filter = 0.5*(1.0+cos(M_PI*( (double)(i-imax)/(double)(Nwin))));
    data[i] *= filter;
  }
  
}

void wavelet(int m, double *waveletE, double *waveletO, int N, double nrm, double dom, double DOM, double A, double B, double insDOM)
{
    
    int i;
    double om;
    double x, y, z;
    double *DE, *DO;
    
     DE = (double*)malloc(sizeof(double)* (2*N));
     DO = (double*)malloc(sizeof(double)* (2*N));

         // zero and postive frequencies
          for(i=0; i<= N/2; i++)
           {
            om = (double)(i)*dom;
            
            y = phitilde(om+(double)(m)*DOM, insDOM, A, B);
            z = phitilde(om-(double)(m)*DOM, insDOM, A, B);
               
               x = y+z;
            
               REAL(DE,i) = IRTWO*x;
               IMAG(DE,i) = 0.0;
               
               // sign is opposite to what paper says
                x = y-z;
                                  
                REAL(DO,i) = 0.0;
                IMAG(DO,i) = IRTWO*x;
               

           }
     
           // negative frequencies
            for(i=1; i< N/2; i++)
            {
             om = -(double)(i)*dom;
                         
              y = phitilde(om+(double)(m)*DOM, insDOM, A, B);
              z = phitilde(om-(double)(m)*DOM, insDOM, A, B);
                         
              x = y+z;
    
                REAL(DE,N-i) = IRTWO*x;
                IMAG(DE,N-i) = 0.0;
                
                // sign is opposite to what paper says
                 x = y-z;
                                      
                REAL(DO,N-i) = 0.0;
                IMAG(DO,N-i) = IRTWO*x;
             
             }
    
          gsl_fft_complex_radix2_backward(DE, 1, N);
          gsl_fft_complex_radix2_backward(DO, 1, N);
      
        
                 for(i=0; i < N/2; i++)
                    {
                        waveletE[i] = REAL(DE,N/2+i)/nrm;
                        waveletO[i] = REAL(DO,N/2+i)/nrm;
                    }
                 for(i=0; i< N/2; i++)
                 {
                      waveletE[i+N/2] = REAL(DE,i)/nrm;
                      waveletO[i+N/2] = REAL(DO,i)/nrm;
                 }

      free(DE);
      free(DO);

    
}


double phitilde(double om, double insDOM, double A, double B)
{
    double x, y, z;
    
       z = 0.0;
       
       if(fabs(om) >= A && fabs(om) < A+B)
       {
           x = (fabs(om)-A)/B;
           y = gsl_sf_beta_inc(nx, nx, x);
           z = insDOM*cos(y*PI/2.0);
       }
       
       if(fabs(om) < A) z = insDOM;
    
    return(z);
    
}

double *double_vector(int N)
{
    return malloc( (N+1) * sizeof(double) );
}

void free_double_vector(double *v)
{
    free(v);
}

double **double_matrix(int N, int M)
{
    int i;
    double **m = malloc( (N+1) * sizeof(double *));
    
    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(double));
    }
    
    return m;
}

void free_double_matrix(double **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_double_vector(m[i]);
    free(m);
}

double ***double_tensor(int N, int M, int L)
{
    int i,j;
    
    double ***t = malloc( (N+1) * sizeof(double **));
    for(i=0; i<N+1; i++)
    {
        t[i] = malloc( (M+1) * sizeof(double *));
        for(j=0; j<M+1; j++)
        {
            t[i][j] = malloc( (L+1) * sizeof(double));
        }
    }
    
    return t;
}

void free_double_tensor(double ***t, int N, int M)
{
    int i;
    
    for(i=0; i<N+1; i++) free_double_matrix(t[i],M);
    
    free(t);
}


int *int_vector(int N)
{
    return malloc( (N+1) * sizeof(int) );
}

void free_int_vector(int *v)
{
    free(v);
}



void bwbpf(double *in, double *out, int fwrv, int M, int n, double s, double f1, double f2)
{
    /* Butterworth bandpass filter
     n = filter order 4,8,12,...
     s = sampling frequency
     f1 = upper half power frequency
     f2 = lower half power frequency  */
    
    if(n % 4){ printf("Order must be 4,8,12,16,...\n"); return;}
    
    int i, j;
    double a = cos(PI*(f1+f2)/s)/cos(PI*(f1-f2)/s);
    double a2 = a*a;
    double b = tan(PI*(f1-f2)/s);
    double b2 = b*b;
    double r;
    
    n = n/4;
    double *A = (double *)malloc(n*sizeof(double));
    double *d1 = (double *)malloc(n*sizeof(double));
    double *d2 = (double *)malloc(n*sizeof(double));
    double *d3 = (double *)malloc(n*sizeof(double));
    double *d4 = (double *)malloc(n*sizeof(double));
    double *w0 = (double *)malloc(n*sizeof(double));
    double *w1 = (double *)malloc(n*sizeof(double));
    double *w2 = (double *)malloc(n*sizeof(double));
    double *w3 = (double *)malloc(n*sizeof(double));
    double *w4 = (double *)malloc(n*sizeof(double));
    double x;
    
    for(i=0; i<n; ++i)
    {
        r = sin(PI*(2.0*(double)i+1.0)/(4.0*(double)n));
        s = b2 + 2.0*b*r + 1.0;
        A[i] = b2/s;
        d1[i] = 4.0*a*(1.0+b*r)/s;
        d2[i] = 2.0*(b2-2.0*a2-1.0)/s;
        d3[i] = 4.0*a*(1.0-b*r)/s;
        d4[i] = -(b2 - 2.0*b*r + 1.0)/s;
        w0[i] = 0.0;
        w1[i] = 0.0;
        w2[i] = 0.0;
        w3[i] = 0.0;
        w4[i] = 0.0;
    }
    
    for(j=0; j< M; ++j)
    {
        if(fwrv == 1) x = in[j];
        if(fwrv == -1) x = in[M-j-1];
        for(i=0; i<n; ++i)
        {
            w0[i] = d1[i]*w1[i] + d2[i]*w2[i]+ d3[i]*w3[i]+ d4[i]*w4[i] + x;
            x = A[i]*(w0[i] - 2.0*w2[i] + w4[i]);
            w4[i] = w3[i];
            w3[i] = w2[i];
            w2[i] = w1[i];
            w1[i] = w0[i];
        }
        if(fwrv == 1) out[j] = x;
        if(fwrv == -1) out[M-j-1] = x;
    }
    
    free(A);
    free(d1);
    free(d2);
    free(d3);
    free(d4);
    free(w0);
    free(w1);
    free(w2);
    free(w3);
    free(w4);
    
    return;
}

static int ad_compare_doubles(const void *a, const void *b)
{
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double ad_upper_tail_series(double u)
{
    const double u2 = u * u;
    return 1.0 - 1.0 / u2 + 3.0 / (u2 * u2) - 15.0 / (u2 * u2 * u2)
           + 105.0 / (u2 * u2 * u2 * u2);
}

static void ad_gaussian_log_terms(double z, double *log_cdf, double *log_tail)
{
    const double cdf = 0.5 * erfc(-z / AD_SQRT_2);

    if (cdf > 0.0 && cdf < 0.999) {
        *log_cdf = log(cdf);
        *log_tail = log1p(-cdf);
        return;
    }

    if (cdf <= 0.0) {
        const double s = ad_upper_tail_series(-z);
        *log_cdf = -0.5 * z * z + log(s / (-z * AD_SQRT_2PI));
        *log_tail = -exp(*log_cdf);
        return;
    }

    const double s = ad_upper_tail_series(z);
    const double tail = s * exp(-0.5 * z * z) / (z * AD_SQRT_2PI);
    *log_cdf = -tail;
    *log_tail = -0.5 * z * z + log(s / (z * AD_SQRT_2PI));
}

double anderson_darling_known_normal_pvalue_from_statistic(double a2)
{
    double cdf;

    if (a2 <= 0.0) {
        return 1.0;
    }

    if (a2 < 2.0) {
        cdf = exp(-1.2337141 / a2) / sqrt(a2)
              * (2.00012 + 0.247105 * a2 - 0.0649821 * a2 * a2
                 + 0.0347962 * a2 * a2 * a2 - 0.0116720 * pow(a2, 4.0)
                 + 0.00168691 * pow(a2, 5.0));
    } else {
        cdf = exp(-exp(1.0776 - 2.30695 * a2 + 0.43424 * a2 * a2
                       - 0.082433 * a2 * a2 * a2 + 0.008056 * pow(a2, 4.0)
                       - 0.0003146 * pow(a2, 5.0)));
    }

    if (cdf <= 0.0) {
        return 1.0;
    }
    if (cdf >= 1.0) {
        return 0.0;
    }
    return 1.0 - cdf;
}

double anderson_darling_known_normal_statistic(const double *x, int nsamples, double mean, double variance)
{
    double *z;
    double inv_std;
    double sum = 0.0;
    int i;

    if (x == NULL || nsamples < 2 || variance <= 0.0) {
        return NAN;
    }

    z = (double *)malloc((size_t)nsamples * sizeof(double));
    if (z == NULL) {
        return NAN;
    }

    inv_std = 1.0 / sqrt(variance);
    for (i = 0; i < nsamples; i++) {
        z[i] = (x[i] - mean) * inv_std;
    }

    qsort(z, (size_t)nsamples, sizeof(double), ad_compare_doubles);

    for (i = 0; i < nsamples; i++) {
        double lx, ly;
        const double k = (double)(i + 1);
        ad_gaussian_log_terms(z[i], &lx, &ly);
        sum += ((2.0 * k - 1.0) * lx + (2.0 * nsamples - 2.0 * k + 1.0) * ly)
               / (double)nsamples;
    }

    free(z);
    return -(double)nsamples - sum;
}

double anderson_darling_known_normal_pvalue(
    const double *x,
    int nsamples,
    double mean,
    double variance,
    double *a2_out)
{
    const double a2 = anderson_darling_known_normal_statistic(x, nsamples, mean, variance);

    if (a2_out != NULL) {
        *a2_out = a2;
    }
    if (!isfinite(a2)) {
        return NAN;
    }
    return anderson_darling_known_normal_pvalue_from_statistic(a2);
}




