#include <omp.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include "BayesLine.h"

// Linux
// gcc -std=gnu99 -fopenmp -w -o BWtest BWtest.c BayesLine.c -lm -lgslcblas -lgsl


// Compile line for OSX with a HomeBrew install of gsl
// gcc  -I/opt/homebrew/include -L/opt/homebrew/lib -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -Xclang -fopenmp -lomp -o BWtest BWtest.c BayesLine.c -lm -lgslcblas -lgsl

static void tukey(double *data, double alpha, int N);
static double tukey_power_correction(double alpha, int N);
void bwbpf(double *in, double *out, int fwrv, int M, int n, double s, double f1, double f2);

int main(int argc,char **argv)
{
    int ND, N, i, j, k, kk, imin, dec;
    double Tobs;
    double alpha, x, y, z, hr, hi, fac, window_power_correction, psd_scale;
    double *data, *dataf, *fdata, *times, *dcopy;
    double *psd, *invpsd, *splinePSD, *fprop;
    double *timeX, *dataX;
    double fmin, fmax, dt, fny;
    double fmn, fmx;
    int wdth, nf, nnu;
    double numin, numax;
    char ch;
    char filename[1024];
    
    struct BayesLineParams *bptr  = calloc(1, sizeof(struct BayesLineParams));
    
    FILE *in;
    FILE *out;
    FILE *white;
    
    if(argc!=2)
    {
        printf("./BLtest file\n");
        return 1;
    }
    
    fmin = 20.0;
    fmax = 1024.0;

    in = fopen(argv[1],"r");
    ND = 0;
    while((ch=fgetc(in))!=EOF) if(ch=='\n') ND += 1;
    rewind(in);

    printf("Number of points in data = %d\n", ND);
    
    timeX = (double*)malloc(sizeof(double)* (ND));
    dataX = (double*)malloc(sizeof(double)* (ND));
    
    for (i = 0; i < ND; ++i)
    {
        fscanf(in,"%lf%lf", &timeX[i], &dataX[i]);
    }
    fclose(in);
    
    dt = (timeX[ND-1]-timeX[0])/(double)(ND);
    Tobs = rint((double)(ND)*dt);  // duration
    dt = Tobs/(double)(ND);
    fny = 1.0/(2.0*dt);  // Nyquist
    
    printf("%f %e %f %f\n", Tobs, dt, fny, fmax);
    
    dec = (int)(fny/fmax);
    
    if(dec > 8) dec = 8;
    printf("Down sample = %d\n", dec);
    
    N = ND/dec;

    printf("Number of points used in analysis = %d\n", N);
    
    if(dec > 1)
    {
        
       // fmn = 1.0/Tobs;
        fmn = fmin/2.0;
        fmx = fmax;
        
        // apply 8th order zero phase bandpass filter
        bwbpf(dataX, dataX, 1, ND, 8, 1.0/dt, fmx, fmn);
        bwbpf(dataX, dataX, -1, ND, 8, 1.0/dt, fmx, fmn);
        
    }
    
    times = (double*)malloc(sizeof(double)* (N));
    data = (double*)malloc(sizeof(double)* (N));
    dcopy = (double*)malloc(sizeof(double)* (N));
    dataf = (double*)malloc(sizeof(double)* (N));
    fdata = (double*)malloc(sizeof(double)* (N));
    
    // decimate
    for (i = 0; i < N; ++i)
    {
        times[i] = timeX[i*dec];
        data[i] = dataX[i*dec];
    }
    
    free(timeX);
    free(dataX);
    
    alpha = (2.0*t_rise/Tobs);
    
    tukey(data, alpha, N);
    
    for(i=0; i< N; i++) dataf[i] = data[i];
    
    gsl_fft_real_radix2_transform(dataf, 1, N);
    
    // keep a copy of the raw FFTed data
    for(i=0; i< N; i++) dcopy[i] = dataf[i];
    
    dt = Tobs/(double)(N);
    
    // switch to FFTW storage format
    fdata[0] = 0.0;
    fdata[1] = 0.0;
    for (i = 1; i < N/2; ++i)
    {
        fdata[2*i] = dataf[i];
        fdata[2*i+1] = dataf[N-i];
    }
    

    psd = malloc((N/2)*sizeof(double));
    invpsd = malloc((N/2)*sizeof(double));
    splinePSD = malloc((N/2)*sizeof(double));
    fprop = malloc((N/2)*sizeof(double));
    
    bptr->maxBLLines = 1000;
    
    BayesLineSetup(bptr, fdata, fmin, fmax, dt, Tobs);
    
    BayesLineBurnin(bptr, data, fdata, "H1", fprop, 1);
    
    
    // Note the BayesLineBurnin code is passing fdata back with
    // signals and glitches removed.
    
    out = fopen("fprop.dat","w");
    for (i = 0; i < N/2-bptr->data->imin; ++i)
    {
        fprintf(out,"%e %e\n", (double)(i+bptr->data->imin)/Tobs, fprop[i]);
    }
    fclose(out);
    
    BayesLineRJMCMC(bptr, fdata, psd, invpsd, splinePSD, N, 100000, 1.0, 1, fprop, 1);
    
    // BayesLineRJMCMC returns the PSD in BayesWave internal units. With the
    // dt/sqrt(2)-scaled Fourier coefficients used by BayesLine, those units are
    // Tobs/4 times the physical one-sided PSD. Convert only the standalone
    // driver outputs back to duration-independent PSD units.
    
    
    window_power_correction = tukey_power_correction(alpha, N);
    psd_scale = 4.0*window_power_correction/Tobs;
    
    out = fopen("psd.dat","w");
    fprintf(out,"%e %e\n", 0.0, psd[1]*psd_scale);
    for (i = 1; i < N/2; ++i)
    {
        fprintf(out,"%e %e\n", (double)(i)/Tobs, psd[i]*psd_scale);
    }
    fclose(out);
    
    out = fopen("psd_components.dat","w");
    for (i = bptr->data->imin; i < N/2; ++i)
    {
        fprintf(out,"%e %e %e\n", (double)(i)/Tobs, bptr->Sbase[i-bptr->data->imin]*psd_scale, bptr->Sline[i-bptr->data->imin]*psd_scale);
    }
    fclose(out);
    
    
    // with glitches removed
    out = fopen("periodogram.dat","w");
    for (i = 1; i < N/2; ++i)
    {
        fprintf(out,"%e %e\n", (double)(i)/Tobs, psd_scale*2.0*(fdata[2*i]*fdata[2*i]+fdata[2*i+1]*fdata[2*i+1]));
    }
    fclose(out);
    
    out = fopen("frequency_data.dat","w");
    x = sqrt(16.0*window_power_correction/Tobs);
    fprintf(out,"%e %e %e\n", 0.0, 0.0, 0.0);
    for (i = 1; i < N/2; ++i)
    {
        fprintf(out,"%e %e %e\n", (double)(i)/Tobs, fdata[2*i]*x, fdata[2*i+1]*x);
    }
    fclose(out);
    
    gsl_rng *r = bptr->r;
   
    white = fopen("whitelsf_noglitch.dat","w");
    out = fopen("p_linesub.dat","w");
    x = sqrt(1.0/2.0);
    // fill with Gaussian below fmin. Keeps wavelet filters happy
    for (i = 1; i < bptr->data->imin; ++i)
    {
        fprintf(white, "%e %e %e\n", (double)(i)/bptr->data->Tobs, gsl_ran_gaussian(r, x), gsl_ran_gaussian(r, x));
    }
    for (i = bptr->data->imin; i < N/2; ++i)
    {
        x = bptr->Sline[i-bptr->data->imin]/(bptr->Snf[i-bptr->data->imin]);
        hr = 0.0;
        hi = 0.0;
        if(x > 1.0e-2)
        {
            y = x*fdata[2*i];
            z = x*fdata[2*i+1];
            // Note, the BW conventions differ from the lmcmc log likelihood conventions by a factor of 2
            // The BayseLine burin code has the standard 1-sided power defintion, so lmcmc matches that
            lmcmc(1000, 2.0*bptr->Sbase[i-bptr->data->imin], 2.0*bptr->Sline[i-bptr->data->imin], fdata[2*i], fdata[2*i+1], y, z, &hr, &hi, r);
        }
        x = sqrt(bptr->Sbase[i-bptr->data->imin]);
        fprintf(white, "%e %e %e\n", (double)(i)/bptr->data->Tobs, (fdata[2*i]-hr)/x, (fdata[2*i+1]-hi)/x);
        fprintf(out, "%e %e\n", (double)(i)/bptr->data->Tobs, 2.0*((fdata[2*i]-hr)*(fdata[2*i]-hr)+(fdata[2*i+1]-hi)*(fdata[2*i+1]-hi)));
    }
    fclose(white);
    fclose(out);
    
    
    white = fopen("whitelsf.dat","w");
     fac = sqrt(Tobs*Tobs/((double)(N)*(double)(N)*2.0));
     for (i = 0; i < N; ++i) dataf[i] *= fac;
    
    // data with glitches
    out = fopen("periodogram_raw.dat","w");
    for (i = 1; i < N/2; ++i)
    {
        fprintf(out,"%e %e\n", (double)(i)/Tobs, 2.0*window_power_correction/Tobs*2.0*(dataf[i]*dataf[i]+dataf[N-i]*dataf[N-i]));
    }
    fclose(out);
    
    x = sqrt(1.0/2.0);
    white = fopen("whitelsf.dat","w");
    // fill with Gaussian below fmin. Keeps wavelet filters happy
    for (i = 1; i < bptr->data->imin; ++i)
    {
        fprintf(white, "%e %e %e\n", (double)(i)/bptr->data->Tobs, gsl_ran_gaussian(r, x), gsl_ran_gaussian(r, x));
    }
    for (i = bptr->data->imin; i < N/2; ++i)
    {
        x = bptr->Sline[i-bptr->data->imin]/(bptr->Snf[i-bptr->data->imin]);
        hr = 0.0;
        hi = 0.0;
        if(x > 1.0e-2)
        {
            y = x*dataf[i];
            z = x*dataf[N-i];
            // Note, the BW conventions differ from the lmcmc log likelihood conventions by a factor of 2
            // The BayseLine burin code has the standard 1-sided power defintion, so lmcmc matches that
            lmcmc(1000, 2.0*bptr->Sbase[i-bptr->data->imin], 2.0*bptr->Sline[i-bptr->data->imin], dataf[i], dataf[N-i], y, z, &hr, &hi, r);
        }
        x = sqrt(bptr->Sbase[i-bptr->data->imin]);
        fprintf(white, "%e %e %e\n", (double)(i)/bptr->data->Tobs, (dataf[i]-hr)/x, (dataf[N-i]-hi)/x);
       
    }
    fclose(white);
    
    // make a Qscan of the data
    // switch back to GSL storage format
    dataf[0] = 0.0;
    dataf[N/2] = 0.0;
    for (i = 1; i < N/2; ++i)
    {
        dataf[i] = fdata[2*i];
        dataf[N-i] = fdata[2*i+1];
    }
    
    Qscan(dataf, psd, 16.0, fmin, fmax, dt, N);
    system("gnuplot Qscan.gnu");
    system("mv Qscan.png Qscan_clean.png");

    fac = dt/sqrt(2.0);
    for(i=1; i< N; i++) dcopy[i] *= fac;
    Qscan(dcopy, psd, 16.0, fmin, fmax, dt, N);
    system("gnuplot Qscan.gnu");
    system("mv Qscan.png Qscan_raw.png");
    
    system("magick -delay 100 Qscan_raw.png Qscan_clean.png -loop 0 blink.gif");
    
    return 0;
}

static void tukey(double *data, double alpha, int N)
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

static double tukey_power_correction(double alpha, int N)
{
  int i, imin, imax;
  double filter;
  double sumw2;
  
  imin = (int)(alpha*(double)(N-1)/2.0);
  imax = (int)((double)(N-1)*(1.0-alpha/2.0));
  
  int Nwin = N-imax;
  
  sumw2 = 0.0;
  for(i=0; i< N; i++)
  {
    filter = 1.0;
    if(i<imin) filter = 0.5*(1.0+cos(M_PI*( (double)(i)/(double)(imin)-1.0 )));
    if(i>imax) filter = 0.5*(1.0+cos(M_PI*( (double)(i-imax)/(double)(Nwin))));
    sumw2 += filter*filter;
  }
  
  return (double)(N)/sumw2;
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
    double a = cos(M_PI*(f1+f2)/s)/cos(M_PI*(f1-f2)/s);
    double a2 = a*a;
    double b = tan(M_PI*(f1-f2)/s);
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
        r = sin(M_PI*(2.0*(double)i+1.0)/(4.0*(double)n));
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
