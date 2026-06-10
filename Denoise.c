/**************************************************************************
 
 Copyright (c) 2025 Neil Cornish
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 ************************************************************************/

//OSX Homebrew
//gcc -O3 -I/opt/homebrew/include -L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp -lomp -w -o Denoise Denoise.c -lgsl -lgslcblas -lm


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>

#include <omp.h>

#ifndef _OPENMP
#define omp ignore
#endif

#define TSUN 4.92569043916e-6
#define PI 3.1415926535897931159979634685442
#define PI2 9.869604401089357992304940125905
#define TPI 6.2831853071795862319959269370884
#define RT2PI 2.5066282746310005024
#define RTPI 1.772453850905516
#define RT2 1.414213562373095
#define IRT2 0.707106781186547549
#define TRT2 2.8284271247461901
#define LN2 0.6931471805599453
#define MSUN_SI 1.988546954961461467461011951140572744e30

#define sthresh 9.0
#define warm 5.0
#define Qs 8.0
#define t_rise 1.0
#define Mline 2048
#define Nlegmax 64
#define test 1
#define addmul 1
#define maxmod 10000
#define verbose 0
#define printQscan 0
#define dfmin 8.0
#define dfmax 256.0
#define stiffness 1.0e4
#define smooth 16.0
#define etol 0.05
#define tol 0.2
#define linemul 9.0
#define ompflag 0
#define Qprint 8.0
#define finc 1.0
static const int qscan_subscale = 40;

struct Point
{ int x;
  int y;
};

struct PSDProducts
{
    int dec;
    int N;
    int Nf;
    double Tobs;
    double *times;
    double *downsampled_data;
    double *freqs;
    double *dfreq;
    double *asd;
    double *smasd;
    char clean_filename[1024];
};

int *int_vector(int N);
void free_int_vector(int *v);
double *double_vector(int N);
void free_double_vector(double *v);
double **double_matrix(int N, int M);
void free_double_matrix(double **m, int N);
int **int_matrix(int N, int M);
void free_int_matrix(int **m, int N);

void tukey(double *data, double alpha, int N);
void medspecspline(double *data, double *S, double *SN, double *SM, double df, int N);
void whiten(double *data, double *Sn, int N);
void max_array_element(double *max, int *index, double *array, int n);
void phase_blind_time_shift(double *corr, double *corrf, double *data1, double *data2, int n);
double f_nwip(double *a, double *b, int n);
void TransformC(double *a, double *freqs, double **tf, double **tfR, double **tfI, double Q, double Tobs, int n, int m);
void layerC(double *a, double f, double *tf, double *tfR, double *tfI, double Q, double Tobs, double fix, int n);
void SineGaussianC(double *hs, double *sigpar, double Tobs, int N);
void specest(double *data, int N, double dt, double fmin, double fmx, double *SN, double *SM, double *PS);
void clean(double *D, double *Draw, double *glitch, double *sqf, double *freqs, double *Sn, double *specD, double *sspecD, double df, double Q, double Tobs, double scale, double alpha, int Nf, int N, int imin, int imax, double *SNR, int pflag);
void bwbpf(double *in, double *out, int fwrv, int M, int n, double s, double f1, double f2);
double Getscale(double *freqs, double Q, double Tobs, double fmx, int n, int m);
void recursive_phase_evolution(double dre, double dim, double *cosPhase, double *sinPhase);
void SineGaussianF(double *hs, double *sigpar, double Tobs, int N);
void isums(double x, double *is);
void solve(int n, double **M, double *av, double *yv);
void splinespace(int Ns, int istart, int iend, double *SM, double *freqs, double Tobs, double *splineA, double *splinef, int *Nk);
void makespec(double *SM, double *freqs, int istart, int iend, int Nknot, double *splinef, double *splineA);
void floodFill(int **img, int m, int n, int x, int y, int newClr);
void qscanf(double *data, double *smasd, double Q, double Tobs, double fmin, double fmax, int NT, int dec, double *feature);
void nongaussian(double *data, double *freqs, double *smasd, double **tfD, double **tfDR, double fmax, double Q, double Tobs, int NT, int Nf, int dec, double *feature);
int estimate_psd_products(const char *input_file, double fmax, struct PSDProducts *products);
void free_psd_products(struct PSDProducts *products);
int extract_features_from_products(const struct PSDProducts *products);
void set_clean_filename(const char *input_file, char *output_file, size_t output_size);


int main(int argc, char *argv[])
{
    struct PSDProducts products = {0};
    int status;
    double fmax;

    if(argc != 3)
    {
        printf("./Denoise file fmax\n");
        return 1;
    }

    fmax = atof(argv[2]);

    status = estimate_psd_products(argv[1], fmax, &products);
    if(status != 0)
    {
        free_psd_products(&products);
        return status;
    }

    status = extract_features_from_products(&products);
    free_psd_products(&products);

    return status;
}

int estimate_psd_products(const char *input_file, double fmax, struct PSDProducts *products)
{
    int i, ND, N, dec;
    double *timeF, *dataF, *dfreq;
    double *times, *data;
    double x, y, dt, df, Tobs, fny, fmn, fmx, fmin;
    double *SN, *SM, *PS;
    double alpha;
    double *freqs;
    clock_t start, end;
    double cpu_time_used;
    FILE *in;

    int Nthread = 4;
    omp_set_num_threads(Nthread);

    printf("starting PSD estimation\n");

    in = fopen(input_file,"r");
    if(in == NULL)
    {
        printf("input data does not exist\n");
        return 2;
    }

    ND = -1;
    while(!feof(in))
    {
        fscanf(in,"%lf%lf", &x, &y);
        ND++;
    }
    rewind(in);
    if(verbose == 1) printf("Number of points in data = %d\n", ND);

    x = log2((double)(ND));
    if(x-floor(x) > 0.0)
    {
        printf("data provided does not lead to 2^n samples\n");
        fclose(in);
        return 1;
    }

    timeF = (double*)malloc(sizeof(double)*ND);
    dataF = (double*)malloc(sizeof(double)*ND);

    for (i = 0; i < ND; ++i)
    {
        fscanf(in,"%lf%lf", &timeF[i], &dataF[i]);
    }
    fclose(in);

    dt = (timeF[ND-1]-timeF[0])/(double)(ND);
    Tobs = rint((double)(ND)*dt);
    dt = Tobs/(double)(ND);
    fny = 1.0/(2.0*dt);

    printf("%f %e %f %f\n", Tobs, dt, fny, fmax);

    dec = (int)(fny/fmax);
    if(dec > 8) dec = 8;
    if(dec < 1) dec = 1;

    if(verbose == 1) printf("Down sample = %d\n", dec);

    N = ND/dec;
    if(verbose == 1) printf("Number of points used in analysis = %d\n", N);

    if(dec > 1)
    {
        fmn = 1.0/Tobs;
        fmx = fmax;
        bwbpf(dataF, dataF, 1, ND, 8, 1.0/dt, fmx, fmn);
        bwbpf(dataF, dataF, -1, ND, 8, 1.0/dt, fmx, fmn);
    }

    times = (double*)malloc(sizeof(double)*N);
    data = (double*)malloc(sizeof(double)*N);
    dfreq = (double*)malloc(sizeof(double)*N);

    for (i = 0; i < N; ++i)
    {
        times[i] = timeF[i*dec];
        data[i] = dataF[i*dec];
    }

    products->times = times;
    products->downsampled_data = double_vector(N);
    for (i = 0; i < N; ++i) products->downsampled_data[i] = data[i];

    free(timeF);
    free(dataF);

    dt = Tobs/(double)(N);
    fny = 1.0/(2.0*dt);
    fmin = 1.0/Tobs;
    df = 1.0/Tobs;

    alpha = (2.0*t_rise/Tobs);
    tukey(data, alpha, N);

    for (i = 0; i < N; ++i) dfreq[i] = data[i];
    gsl_fft_real_radix2_transform(dfreq, 1, N);

    if(verbose == 1) printf("Data volume %f seconds, %f Hz\n", Tobs, fny);

    freqs = double_vector(N/2);
    for (i = 0; i < N/2; ++i) freqs[i] = (double)(i)/Tobs;

    SM = double_vector(N/2);
    SN = double_vector(N/2);
    PS = double_vector(N/2);

    start = clock();
    specest(data, N, dt, fmin, fmax, SN, SM, PS);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("spectrum took %f seconds\n", cpu_time_used);

    products->dec = dec;
    products->N = N;
    products->Nf = N/2;
    products->Tobs = Tobs;
    products->freqs = freqs;
    products->dfreq = dfreq;
    products->asd = SN;
    products->smasd = SM;
    set_clean_filename(input_file, products->clean_filename, sizeof(products->clean_filename));

    free(PS);
    free(data);

    return 0;
}

int extract_features_from_products(const struct PSDProducts *products)
{
    FILE *out;
    int i, j, Nf, NT;
    double x, fac, Tobs, dt;
    double *data;
    double *dataw;
    double *asd;
    double *smasd;
    double *feature;

    Nf = products->Nf;
    NT = products->N;
    Tobs = products->Tobs;

    printf("Observation time %f\n", Tobs);
    printf("Frequency bins = %d\n", Nf);

    data = (double*)malloc(sizeof(double)*NT);
    dataw = (double*)malloc(sizeof(double)*NT);
    asd = (double*)malloc(sizeof(double)*Nf);
    smasd = (double*)malloc(sizeof(double)*Nf);
    feature = (double*)malloc(sizeof(double)*NT);

    for(i = 0; i < NT; i++)
    {
        data[i] = products->dfreq[i];
        dataw[i] = 0.0;
        feature[i] = 0.0;
    }

    for(i = 1; i < Nf; i++)
    {
        asd[i] = sqrt(products->asd[i]);
        smasd[i] = sqrt(products->smasd[i]);
    }

    fac = (double)(NT)/sqrt(Tobs);
    for(i = 1; i < Nf; i++)
    {
        asd[i] *= fac;
        smasd[i] *= fac;
    }

    x = 0.0;
    j = 0;
    for(i = 1; i < Nf; i++)
    {
        data[i] /= asd[i];
        data[NT-i] /= asd[i];
        x += 2.0*(data[i]*data[i]+data[NT-i]*data[NT-i]);
        j++;
    }
    printf("white var %e\n", x/(double)(j));

    dt = Tobs/(double)(NT);

    dataw[0] = 0.0;
    for(i = 1; i < NT; i++) dataw[i] = data[i];

    gsl_fft_halfcomplex_radix2_inverse(dataw,1,NT);

    out = fopen("wdata.dat", "w");
    x = sqrt((double)(2.0*NT));
    for(i = 0; i < NT; i++)
    {
        fprintf(out, "%e %e\n", (double)(i)*dt, dataw[i]*x);
    }
    fclose(out);

    qscanf(data, smasd, 8.0, Tobs, products->freqs[1], products->freqs[Nf-1], NT, products->dec, feature);

    out = fopen(products->clean_filename, "w");
    if(out == NULL)
    {
        printf("could not open %s for writing\n", products->clean_filename);
        free(data);
        free(dataw);
        free(asd);
        free(smasd);
        free(feature);
        return 2;
    }
    for(i = 0; i < NT; i++)
    {
        fprintf(out, "%.17e %.17e\n", products->times[i], products->downsampled_data[i]-feature[i]);
    }
    fclose(out);
    printf("Cleaned data written to %s\n", products->clean_filename);

    free(data);
    free(dataw);
    free(asd);
    free(smasd);
    free(feature);

    return 0;
}

void free_psd_products(struct PSDProducts *products)
{
    if(products->times != NULL) free_double_vector(products->times);
    if(products->downsampled_data != NULL) free_double_vector(products->downsampled_data);
    if(products->freqs != NULL) free_double_vector(products->freqs);
    if(products->dfreq != NULL) free(products->dfreq);
    if(products->asd != NULL) free_double_vector(products->asd);
    if(products->smasd != NULL) free_double_vector(products->smasd);

    products->times = NULL;
    products->downsampled_data = NULL;
    products->freqs = NULL;
    products->dfreq = NULL;
    products->asd = NULL;
    products->smasd = NULL;
    products->clean_filename[0] = '\0';
}

void set_clean_filename(const char *input_file, char *output_file, size_t output_size)
{
    const char *base;
    const char *slash;
    const char *dot;

    slash = strrchr(input_file, '/');
    base = slash == NULL ? input_file : slash+1;
    dot = strrchr(base, '.');

    if(dot != NULL && dot != base)
    {
        int prefix_length = (int)(dot-input_file);
        snprintf(output_file, output_size, "%.*s_clean%s", prefix_length, input_file, dot);
    }
    else
    {
        snprintf(output_file, output_size, "%s_clean", input_file);
    }
}

/*******************************************************************************************

Copyright (c) 2022 Neil Cornish

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

**********************************************************************************************/

//##############################################
//OPEN MP

gsl_rng **rvec;
//##############################################

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


void medspecspline(double *data, double *S, double *SN, double *SM, double df, int N)
{
    double Df, Dfmax, x, y;
    int mw, mww, k, i, j, jt;
    int mm, kk, nc, ns, nw, nww;
    double med;
    double *chunk;
    double *sc, *fc;
    double fend;
    double fspace = 2.0; // spline spacing
    double fw; // half window size for median
    gsl_spline   *spline;
    gsl_interp_accel *acc;
    
    // log(2) is median/2 of chi-squared with 2 dof

    for(i=1; i< N/2; i++) S[i] = 2.0*(data[i]*data[i]+data[N-i]*data[N-i]);
    S[0] = S[1];
    
    k = (int)((double)(N/2)*df/fspace)+2; // allocate room for spline
    sc = double_vector(k);
    fc = double_vector(k);
    
    fw = fspace;

    fc[0] = df;
    
    fend = (double)(N/2-1)*df;
    
    i = 0;
    do
    {
        i++;
        fc[i] = fspace*(double)(i);
       // printf("%d %f\n", i, fc[i]);
    }while(fc[i] < fend);
    nc = i+1;
    
    mm = (int)(fspace/df);
    mw = (int)(fw/df);
    chunk = double_vector(2*mw);
    
    for(j=1; j < nc-1; j++)
    {
        // reset to larger median windows
        if(fc[j] >= 30.0-0.5*fspace && fc[j] < 30.0+0.5*fspace)
        {
            free(chunk);
            fw *= 2.0;
            mw = (int)(fw/df);
            chunk = double_vector(2*mw);
        }
        
        // reset to larger median windows
        if(fc[j] >= 100.0-0.5*fspace && fc[j] < 100.0+0.5*fspace)
        {
            free(chunk);
            fw *= 2.0;
            mw = (int)(fw/df);
            chunk = double_vector(2*mw);
        }
        
        // reset to larger median windows
        if(fc[j] >= 200.0-0.5*fspace && fc[j] < 200.0+0.5*fspace)
        {
            free(chunk);
            fw *= 2.0;
            mw = (int)(fw/df);
            chunk = double_vector(2*mw);
        }
        
        // reset to smaller median windows
        if(fc[j]+fw > fend)
        {
            free(chunk);
            fw = fend-fc[j];
            mw = (int)(fw/df);
            chunk = double_vector(2*mw);
        }
        
       // printf("%d %e %e %d\n", j, fc[j], fw, mw);
        
        for(i=0; i< 2*mw; i++)
        {
            chunk[i] = S[j*mm-mw+i];
        }
       
        gsl_sort(chunk, 1, 2*mw);
        sc[j] = log(gsl_stats_median_from_sorted_data(chunk, 1, 2*mw)/LN2);
        
    }
    
    sc[0] = sc[1];
    sc[nc-1] = sc[nc-2];

    /*
    FILE *out;
    out = fopen("ms.dat", "w");
    for(i=0; i< nc; i++) fprintf(out, "%e %e\n", fc[i], sc[i]);
    fclose(out);
     */
     
    free_double_vector(chunk);
    
    // Allocate spline
    spline = gsl_spline_alloc(gsl_interp_akima, nc);
    acc = gsl_interp_accel_alloc();
    gsl_spline_init(spline,fc,sc,nc);
    
    SM[1] = exp(sc[0]);
    SM[N/2-1] = exp(sc[nc-1]);
    for (i = 2; i < N/2-1; ++i)
    {
        SM[i] = exp(gsl_spline_eval(spline,(double)(i)*df,acc));
    }
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    
    free(fc);
    free(sc);
    
    // zap the lines.
    for(i=1;i< N/2;i++)
    {
        SN[i] = SM[i];
        x = S[i]/SM[i];
        if(x > linemul)
        {
            SN[i] = S[i];
        }
    }
    
    
}



void whiten(double *data, double *Sn, int N)
{
    double f, x, y, fix;
    int i;
    
    data[0] = 0.0;
    data[N/2] = 0.0;
    
    for(i=1; i< N/2; i++)
    {
        x = 1.0/sqrt(Sn[i]);
        data[i] *= x;
        data[N-i] *= x;
    }
    
}


void phase_blind_time_shift(double *corr, double *corrf, double *data1, double *data2, int n)
{
    int nb2, i, l, k, j;
    int imax, imin;
    
    nb2 = n / 2;
    
    corr[0] = 0.0;
    corrf[0] = 0.0;
    corr[nb2] = 0.0;
    corrf[nb2] = 0.0;
    
    for (i=1; i < nb2; i++)
    {
        l=i;
        k=n-i;
        
        corr[l]    = (data1[l]*data2[l] + data1[k]*data2[k]);
        corr[k]    = (data1[k]*data2[l] - data1[l]*data2[k]);
        corrf[l] = corr[k];
        corrf[k] = -corr[l];
    }
    
    gsl_fft_halfcomplex_radix2_inverse(corr, 1, n);
    gsl_fft_halfcomplex_radix2_inverse(corrf, 1, n);
    
    
}



double f_nwip(double *a, double *b, int n)
{
    int i, j, k;
    double arg, product;
    double ReA, ReB, ImA, ImB;
    
    arg = 0.0;
    for(i=1; i<n/2; i++)
    {
        j = i;
        k = n-j;
        ReA = a[j]; ImA = a[k];
        ReB = b[j]; ImB = b[k];
        product = ReA*ReB + ImA*ImB;
        arg += product;
    }
    
    return(arg);
    
}

void TransformC(double *a, double *freqs, double **tf, double **tfR, double **tfI, double Q, double Tobs, int n, int m)
{
    int j;
    double fix;
    
    // [0] t0 [1] f0 [2] Q [3] Amp [4] phi
    
    fix = sqrt((double)(n/2));
    
    #pragma omp parallel for
    for(j = 0; j < m; j++)
    {
        layerC(a, freqs[j], tf[j], tfR[j], tfI[j], Q, Tobs, fix, n);
    }
    
}


void layerC(double *a, double f, double *tf, double *tfR, double *tfI, double Q, double Tobs, double fix, int n)
{
    int i;
    double *AC, *AF;
    double *b;
    double *params;
    double bmag;
    
    params= double_vector(6);
    AC=double_vector(n);  AF=double_vector(n);
    b = double_vector(n);
    
    params[0] = 0.0;
    params[1] = f;
    params[2] = Q;
    params[3] = 1.0;
    params[4] = 0.0;
    
    SineGaussianC(b, params, Tobs, n);
    
    bmag = sqrt(f_nwip(b, b, n)/(double)n);
    
    bmag /= fix;
    
    phase_blind_time_shift(AC, AF, a, b, n);
    
    for(i = 0; i < n; i++)
    {
        tfR[i] = AC[i]/bmag;
        tfI[i] = AF[i]/bmag;
        tf[i] = tfR[i]*tfR[i]+tfI[i]*tfI[i];
    }
    
    free_double_vector(AC);  free_double_vector(AF);
    free_double_vector(b);  free_double_vector(params);
    
}

void SineGaussianC(double *hs, double *sigpar, double Tobs, int N)
{
    double f0, t0, Q, sf, sx, Amp;
    double fmx, fmn;//, fac;
    double phi, f;
    double tau;
    double re,im;
    double q, p, u;
    double A, B, C;
    double Am, Bm, Cm;
    double a, b, c;
    double dt, fac;
    double cosPhase_m, sinPhase_m, cosPhase_p, sinPhase_p;
    
    int i, imid, istart,istop,imin,imax,iend,even,odd;
    
    t0  = sigpar[0];
    f0  = sigpar[1];
    Q   = sigpar[2];
    Amp = sigpar[3];
    phi = sigpar[4];
    
    tau = Q/(TPI*f0);
    
    fmx = f0 + 3.0/tau;  // no point evaluating waveform past this time (many efolds down)
    fmn = f0 - 3.0/tau;  // no point evaluating waveform before this time (many efolds down)
    
    i = (int)(f0*Tobs);
    imin = (int)(fmn*Tobs);
    imax = (int)(fmx*Tobs);
    if(imax - imin < 10)
    {
        imin = i-5;
        imax = i+5;
    }
    
    if(imin < 0) imin = 1;
    if(imax > N/2) imax = N/2;
    
    hs[0] = 0.0;
    hs[N/2] = 0.0;
    
    for(i = 1; i < N/2; i++)
    {
        hs[i] = 0.0;
        hs[N-i] = 0.0;
    }
    
    dt = Tobs/(double)(N);
    fac = sqrt(sqrt(2.0)*PI*tau/dt);
    
    /* Use recursion relationship  */
    
    imid = (int)(f0*Tobs);
    
    p = PI*PI*tau*tau/(Tobs*Tobs);
    
    Bm = exp(-p*(((double)(imid)-f0*Tobs)*((double)(imid)-f0*Tobs)));
    Cm = 1.0;
    
    b = exp(-p*(1.0+2.0*((double)(imid)-f0*Tobs)));
    c = exp(-2.0*p);
    
    // start in the middle and work outwards
    
    B = Bm;
    C = Cm;

    
    for(i = imid; i < imax; i++)
    {
 
        f = (double)(i)/Tobs;
        
        sf = fac*B;
        
        hs[i] = sf;
        hs[N-i] = 0.0;
        
        B *= (C*b);
        C *= c;
        
    }
    
    // reset to midpoint
    
    B = Bm;
    C = Cm;
    
    b = exp(p*(-1.0+2.0*((double)(imid)-f0*Tobs)));
    // c unchanged
    
    for(i = imid; i > imin; i--)
    {

        f = (double)(i)/Tobs;
        
        sf = fac*B;
        
        hs[i] = sf;
        hs[N-i] = 0.0;
        
        B *= (C*b);
        C *= c;
        

    }
    
    
}

void specest(double *data, int N, double dt, double fmin, double fmx, double *SN, double *SM, double *PS)
{
    int i, j, k, M, Nf, Nstep,  ii, m;
    int jj, kk, Nlines;
    int oflag, flag;
    int imin, imax;
    double SNR, max;
    double junk, Tobs, fix, f, t, t0, df, x, y, z, dx;
    double fmn, dfx, Q, fny, delt, scale, dlnf;
    double *freqs, *ref;
    double *Draw;
    double *D, *times;
    double *Sn;
    double *specD, *sspecD;
    double *sdata, *glitch;
    double *intime, *sqf;
    double *pspline, *dspline;
    double sigmean, sigmedian;
    int octave_count;
    int mmax;
    double SNRsq, SNRold, pH, pL, pmax;
    double SNRH, SNRL, pw, alpha;
    double s1, s2, ascale, fac, Dfmax;
    double *linew;
    int pflag;
    
    int sflag;
    
    int modelprint;

    char filename[1024];
    char command[1024];
    
    double fwindow = 8.0;
    
    FILE *out;
    
    int Nthread = 4;
    
    omp_set_num_threads(Nthread);
    
    // usually this is set to zero. Setting to 1 prints out the Qscans at each iteration
    sflag = 0;
    
    Q = Qs;  // Q of transform
    Dfmax = smooth;  // width of smoothing window in Hz
    Tobs = (double)(N)*dt;  // duration
    
    df = 1.0/Tobs;  // frequency resolution
    fny = 1.0/(2.0*dt);  // Nyquist
    
    if(fmx > fny) fmx = fny;
    
    D = (double*)malloc(sizeof(double)* (N));
    Draw = (double*)malloc(sizeof(double)* (N));
    glitch = (double*)malloc(sizeof(double)* (N));
    
    // Copy data over
    for(i=0; i< N; i++)
    {
     Draw[i] = data[i];
     D[i] = data[i];
    }
    
    // Time series data is corrupted by the Tukey window at either end
    // imin and imax define the "safe" region to use
    imin = (int)(2.0*t_rise/dt);
    imax = N-imin;
    
    // Prepare to make spectogram
    
    // logarithmic frequency spacing
    dx = log(2.0)/(double)(qscan_subscale);  // log frequency increment
    Nf = (int)floor(log(fmx/fmin)/dx)+1;
    freqs = (double*)malloc(sizeof(double)* (Nf));   // frequencies used in the analysis
    sqf = (double*)malloc(sizeof(double)* (Nf));
    dlnf = dx;
    x = log(fmin);
    for(i=0; i< Nf; i++)
    {
        freqs[i] = exp(x);
        sqf[i] = sqrt(freqs[i]);
        x += dx;
    }
    
   // printf("frequency layers = %d %f %f %f\n", Nf, freqs[0], freqs[Nf-1], exp(x));
    
    if(verbose == 1) printf("frequency layers = %d fmin %e fmax %e\n", Nf, fmin, fmx);
    
    scale = Getscale(freqs, Q, Tobs, fmx, N, Nf);
    
    sspecD = (double*)malloc(sizeof(double)*(N/2));
    specD = (double*)malloc(sizeof(double)*(N/2));
    Sn = (double*)malloc(sizeof(double)*(N/2));
    
    fac = Tobs/((double)(N)*(double)(N));
    

    SNRold = 0.0;
    clean(D, Draw, glitch, sqf, freqs, Sn, specD, sspecD, df, Q, Tobs, scale, alpha, Nf, N, imin, imax, &SNR, sflag);
    
    if(sflag == 1)
    {
        sprintf(command, "mv Qtransform.dat Qtransform_0.dat");
        system(command);
        sprintf(command, "mv Qtransform_clean.dat Qtransform_clean_0.dat");
        system(command);
        sprintf(command, "mv wglitch.dat wglitch_0.dat");
        system(command);
        sprintf(filename, "spec_0.dat");
        out = fopen(filename,"w");
        for (i = 1; i < N/2; ++i)
        {
            fprintf(out,"%e %e\n", (double)(i)/Tobs, specD[i]*fac);
        }
        fclose(out);
        printf("0 glitch SNR %f\n", SNR);
    }
    
    
    // if big glitches are present we need to rinse and repeat
    i = 0;
    while(i < 10 && (SNR-SNRold) > 10.0)
    {
        SNRold = SNR;
        clean(D, Draw, glitch, sqf, freqs, Sn, specD, sspecD, df, Q, Tobs, scale, alpha, Nf, N, imin, imax, &SNR, sflag);
        i++;
        if(sflag == 1)
        {
            sprintf(command, "mv Qtransform.dat Qtransform_%d.dat", i);
            system(command);
            sprintf(command, "mv Qtransform_clean.dat Qtransform_clean_%d.dat", i);
            system(command);
            sprintf(command, "mv wglitch.dat wglitch_%d.dat", i);
            system(command);
            sprintf(filename, "spec_%d.dat", i);
            out = fopen(filename,"w");
            for (j = 1; j < N/2; ++j)
            {
                fprintf(out,"%e %e\n", (double)(j)/Tobs, specD[j]*fac);
            }
            fclose(out);
            printf("%d glitch SNR %f\n", i, SNR);
        }
    }
    
    out = fopen("glitchSNR.dat","w");
    fprintf(out,"%f\n", SNR);
    fclose(out);

    
    // pass back the cleaned data. This cleaned data is not passed forward
    // from the spectral estimation code. Only used inside the MCMC
    for(i=0; i< N; i++)
    {
     data[i] = D[i];
    }
    
    gsl_fft_real_radix2_transform(D, 1, N);
    
    // Form spectral model for whitening data (lines plus a smooth component)
    medspecspline(D, Sn, specD, sspecD, df, N);
 
    
    for (i = 0; i < N/2; ++i) SM[i] = sspecD[i]*fac;
    for (i = 0; i < N/2; ++i) SN[i] = specD[i]*fac;
    for (i = 0; i < N/2; ++i) PS[i] = Sn[i]*fac;
    
    
    // print the cleaned data
    for(i=0; i< N; i++)
    {
     D[i] = data[i];
    }
    
    // save Q-scan data to file
    if(printQscan == 1)
    {
    clean(D, Draw, glitch, sqf, freqs, Sn, specD, sspecD, df, Qprint, Tobs, scale, alpha, Nf, N, imin, imax, &SNR, 1);
    }
    
    free(D);
    free(Draw);
    free(glitch);
    free(sspecD);
    free(specD);
    free(Sn);
    free(freqs);
    free(sqf);

    
}





void clean(double *D, double *Draw, double *glitch, double *sqf, double *freqs, double *Sn, double *specD, double *sspecD, double df, double Q, double Tobs, double scale, double alpha, int Nf, int N, int imin, int imax, double *SNR, int pflag)
{
    
    int i, j, k, kmin;
    int i1, i2;
    int flag;
    int ii, jj;
    double x, y, dt;
    double S;
    double fac;
    double t, f, fmn;
    
    // allocate some arrays
    double **tfDR, **tfDI;
    double **tfD;
    double **live;
    double **live2;
    
    double *Dtemp, *Drf;
    
    clock_t start, end;
    double cpu_time_used;
    
     double itime, ftime, exec_time;
    
    FILE *out;
    
    live= double_matrix(Nf,N);
    live2= double_matrix(Nf,N);
    tfDR = double_matrix(Nf,N);
    tfDI = double_matrix(Nf,N);
    tfD = double_matrix(Nf,N);
    
    dt = Tobs/(double)(N);

    Dtemp = (double*)malloc(sizeof(double)*(N));
    
    for (i = 0; i < N; i++) Dtemp[i] = Draw[i];
    
    // D holds the previously cleaned time-domain data
    // Draw holds the raw time-domain data
    
    // D is used to compute the spectrum. A copy of Draw, Dtemp, is then whitened using this spectrum and glitches are then identified
    // The glitches are then re-colored, subtracted from Draw to become the new D
    
    // FFT
    gsl_fft_real_radix2_transform(D, 1, N);
    gsl_fft_real_radix2_transform(Dtemp, 1, N);
    
    // Form spectral model for whitening data (lines plus a smooth component)
    medspecspline(D, Sn, specD, sspecD, df, N);
    
    // whiten data
    whiten(Dtemp, specD, N);
    
    x = 0.0;
    j = 0;
    for (i = 1; i < N/2; i++)
    {
        x += 2.0*(Dtemp[i]*Dtemp[i]+Dtemp[N-i]*Dtemp[N-i]);
        j++;
    }
    //printf("white var %e \n", x/(double)(j));
    
    // Wavelet transform
    TransformC(Dtemp, freqs, tfD, tfDR, tfDI, Q, Tobs, N, Nf);

    
    if(pflag == 1)
    {
        i1 = (int)((Tobs/2.0-2.0)/dt);
        i2 = (int)((Tobs/2.0+2.0)/dt);
        out = fopen("Qtransform.dat","w");
        for(j = 0; j < Nf; j++)
        {
            f = freqs[j];
            
            for(i = i1; i < i2; i++)
            {
                t = (double)(i)*dt-Tobs/2.0;
                fprintf(out,"%e %e %e\n", t, f, tfD[j][i]);
            }
            
            fprintf(out,"\n");
        }
        fclose(out);
        
        
        
    }
    
    k = 0;
    //  apply threshold
    for(j = 0; j < Nf; j++)
    {
        for (i = 0; i < N; i++)
        {
            live[j][i] = -1.0;
            if(tfD[j][i] > sthresh) live[j][i] = 1.0;
            if(tfD[j][i] > sthresh) k++;
            live2[j][i] = live[j][i];
        }
    }
   
    
    // dig deeper to extract clustered power
    for(j = 1; j < Nf-1; j++)
    {
        for (i = 1; i < N-1; i++)
        {
            
            flag = 0;
            for(jj = -1; jj <= 1; jj++)
            {
                for(ii = -1; ii <= 1; ii++)
                {
                    if(live[j+jj][i+ii] > 0.0) flag = 1;
                }
            }
            if(flag == 1 && tfD[j][i] > warm) live2[j][i] = 1.0;
        }
    }
    
    
    // build the excess power model
    for (i = 0; i < N; i++)
    {
        Dtemp[i] = 0.0;
    }
    
    k = 0;
    for(j = 0; j < Nf; j++)
    {
        for (i = imin; i < imax; i++)
        {
            if(live2[j][i] > 0.0) Dtemp[i] += scale*sqf[j]*tfDR[j][i];
        }
    }
    
    if(pflag == 1)
       {
        out = fopen("wglitch.dat", "w");
        for(i=i1; i< i2; i++)
        {
            fprintf(out,"%e %e\n", (double)(i)*dt-Tobs/2.0, Dtemp[i]);
         }
       fclose(out);
       }
    
    // Compute the excess power (relative to the current spectral model
    S = 0.0;
    for (i = imin; i < imax; ++i) S += Dtemp[i]*Dtemp[i];
    S = sqrt(S);
    
   
   printf("Excess SNR at Q %f = %f\n", Q, S);
    
    
    *SNR = S;
    
    if(pflag == 1)
    {
        int l;
        k = 0;
        l = 0;
        x = 0.0;
        for(j = 0; j < Nf; j++)
        {
            f = freqs[j];
            
            for(i = 0; i < N; i++)
            {
                t = (double)(i)*dt;
                if(t > t_rise && t < Tobs-t_rise)
                {
                    x += tfD[j][i];
                    k++;
                    if(tfD[j][i] > 9.0) l++;
                }
            }
        
        }
        
        // The spectrogram for whiten unit variance Gaussian data should follow
        // a chi-squared distribiton with two degrees of freedom. Mean = 2.
        // A threshold of 9 should correspond to 0.01111 of the samples
        printf("%f %f\n", x/(double)(k), (double)(l)/(double)(k));
    }
    

    //Unwhiten and subtract the excess power so we can compute a better spectral estimate
    // Back to frequency domain
    
    gsl_fft_real_radix2_transform(Dtemp, 1, N);
    
    
    fmn = freqs[0];
    
    // only use smooth spectrum in the un-whitening
    Dtemp[0] = 0.0;
    for(i=1; i< N/2; i++)
    {
        f = (double)(i)/Tobs;
        //y = 1.0;
        //if(f < fmn) y = 0.5*(1.0+tanh(8.0*(f-0.5*fmn)/fmn));
        //x = y*sqrt(sspecD[i]);
        x = sqrt(sspecD[i]);
        Dtemp[i] *= x;
        Dtemp[N-i] *= x;
    }
    Dtemp[N/2] = 0.0;
    
    gsl_fft_halfcomplex_radix2_inverse(Dtemp, 1, N);
    
    
    x = sqrt((double)(2*N));
    
   
    // data minus the glitch model
    for(i=0; i< N; i++)
    {
        glitch[i] = Dtemp[i]/x;
        D[i] = Draw[i]-glitch[i];
    }
    
    
    if(pflag == 1)
    {
        
        for(i = 0; i < N; i++) Dtemp[i] = D[i];
        gsl_fft_real_radix2_transform(Dtemp, 1, N);
        whiten(Dtemp, specD, N);
        TransformC(Dtemp, freqs, tfD, tfDR, tfDI, Q, Tobs, N, Nf);
        
        out = fopen("Qtransform_clean.dat","w");
        for(j = 0; j < Nf; j++)
        {
            f = freqs[j];
            
            for(i = i1; i < i2; i++)
            {
                t = (double)(i)*dt-Tobs/2.0;
                fprintf(out,"%e %e %e\n", t, f, tfD[j][i]);
            }
            
            fprintf(out,"\n");
        }
        fclose(out);
        
    }
    
    free(Dtemp);
    free_double_matrix(live,Nf);
    free_double_matrix(live2,Nf);
    free_double_matrix(tfDR,Nf);
    free_double_matrix(tfDI,Nf);
    free_double_matrix(tfD,Nf);
    
    
    return;
    
    
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

double Getscale(double *freqs, double Q, double Tobs, double fmx, int n, int m)
{
    double *data, *intime, *ref, **tfR, **tfI, **tf;
    double f, t0, delt, t, x, fix, dt;
    double scale, sqf;
    int i, j;
    
    FILE *out;
    
    data = double_vector(n);
    ref = double_vector(n);
    intime = double_vector(n);
    
    tf = double_matrix(m,n);
    tfR = double_matrix(m,n);
    tfI = double_matrix(m,n);
    
    f = fmx/4.0;
    t0 = Tobs/2.0;
    delt = Tobs/8.0;
    dt = Tobs/(double)(n);
    
    for(i=0; i< n; i++)
    {
        t = (double)(i)*dt;
        x = (t-t0)/delt;
        x = x*x/2.0;
        data[i] = cos(TPI*t*f)*exp(-x);
        ref[i] = data[i];
    }
    
    gsl_fft_real_radix2_transform(data, 1, n);

    TransformC(data, freqs, tf, tfR, tfI, Q, Tobs, n, m);
    
    for(i = 0; i < n; i++) intime[i] = 0.0;
    
    for(j = 0; j < m; j++)
    {
        
        f = freqs[j];
        
        
         sqf = sqrt(f);
        
        for(i = 0; i < n; i++)
        {
            intime[i] += sqf*tfR[j][i];
        }
        
    }
    
    x = 0.0;
    j = 0;
    for(i=0; i< n; i++)
    {
        if(fabs(ref[i]) > 0.01)
        {
            j++;
            x += intime[i]/ref[i];
        }
    }
    
    x /= sqrt((double)(2*n));
    
    scale = (double)j/x;
    
    free_double_vector(data);
    free_double_vector(ref);
    free_double_vector(intime);
    free_double_matrix(tf,m);
    free_double_matrix(tfR,m);
    free_double_matrix(tfI,m);
    
    return scale;
    
}


void recursive_phase_evolution(double dre, double dim, double *cosPhase, double *sinPhase)
{
    /* Update re and im for the next iteration. */
    double cosphi = *cosPhase;
    double sinphi = *sinPhase;
    double x, y;
    
    x = (cosphi*dre + sinphi*dim);
    y = (sinphi*dre - cosphi*dim);
    
    double newRe = cosphi - x;
    double newIm = sinphi - y;
    
    *cosPhase = newRe;
    *sinPhase = newIm;
    
}

void SineGaussianF(double *hs, double *sigpar, double Tobs, int N)
{
    double f0, t0, Q, sf, sx, Amp;
    double fmx, fmn;//, fac;
    double phi, f;
    double tau;
    double re,im;
    double q, p, u;
    double A, B, C;
    double Am, Bm, Cm;
    double a, b, c;
    double cosPhase_m, sinPhase_m, cosPhase_p, sinPhase_p;
    
    int i, imid, istart,istop,imin,imax,iend,even,odd;
    
    t0  = sigpar[0];
    f0  = sigpar[1];
    Q   = sigpar[2];
    Amp = sigpar[3];
    phi = sigpar[4];
    
    tau = Q/(TPI*f0);
    
    fmx = f0 + 3.0/tau;  // no point evaluating waveform past this time (many efolds down)
    fmn = f0 - 3.0/tau;  // no point evaluating waveform before this time (many efolds down)
    
    i = (int)(f0*Tobs);
    imin = (int)(fmn*Tobs);
    imax = (int)(fmx*Tobs);
    if(imax - imin < 10)
    {
        imin = i-5;
        imax = i+5;
    }
    
    if(imin < 0) imin = 1;
    if(imax > N/2) imax = N/2;
    
    hs[0] = 0.0;
    hs[N/2] = 0.0;
    
    for(i = 1; i < N/2; i++)
    {
        hs[i] = 0.0;
        hs[N-i] = 0.0;
    }
    
    /* Use recursion relationship  */
    
    //incremental values of exp(iPhase)
    double dim = sin(TPI*t0/Tobs);
    double dre = sin(0.5*(TPI*t0/Tobs));
    dre = 2.0*dre*dre;
    
    double amplitude = 0.5*(Amp)*RTPI*tau;
    double pi2tau2   = PI*PI*tau*tau;
    double Q2        = Q*Q/f0;
    
    
    imid = (int)(f0*Tobs);
    
    
    q = Q*Q/(f0*Tobs);
    p = PI*PI*tau*tau/(Tobs*Tobs);
    u = PI*PI*tau*tau/Tobs*(2.0*f0*Tobs-1.0);
    
    Am = exp(-q*(double)(imid));
    Bm = exp(-p*(((double)(imid)-f0*Tobs)*((double)(imid)-f0*Tobs)));
    Cm = 1.0;
    
    a = exp(-q);
    b = exp(-p*(1.0+2.0*((double)(imid)-f0*Tobs)));
    c = exp(-2.0*p);
    
    // sine and cosine of phase at reference frequency
    f = (double)(imid)/Tobs;
    double phase = TPI*f*t0;
    double cosPhase_m0  = cos(phase-phi);
    double sinPhase_m0  = sin(phase-phi);
    double cosPhase_p0  = cos(phase+phi);
    double sinPhase_p0  = sin(phase+phi);
    
    // start in the middle and work outwards
    
    A = Am;
    B = Bm;
    C = Cm;
    
    cosPhase_m  = cosPhase_m0;
    sinPhase_m  = sinPhase_m0;
    cosPhase_p  = cosPhase_p0;
    sinPhase_p  = sinPhase_p0;
    
    for(i = imid; i < imax; i++)
    {
        even = 2*i;
        odd = even + 1;
        f = (double)(i)/Tobs;
        
        sf = amplitude*B;
        sx = A;
        re = sf*(cosPhase_m+sx*cosPhase_p);
        im = -sf*(sinPhase_m+sx*sinPhase_p);
        
        hs[i] = re;
        hs[N-i] = im;
        
        A *= a;
        B *= (C*b);
        C *= c;
        
        /* Now update re and im for the next iteration. */
        recursive_phase_evolution(dre, dim, &cosPhase_m, &sinPhase_m);
        recursive_phase_evolution(dre, dim, &cosPhase_p, &sinPhase_p);
    }
    
    // reset to midpoint
    
    A = Am;
    B = Bm;
    C = Cm;
    
    cosPhase_m  = cosPhase_m0;
    sinPhase_m  = sinPhase_m0;
    cosPhase_p  = cosPhase_p0;
    sinPhase_p  = sinPhase_p0;
    
    a = 1.0/a;
    b = exp(p*(-1.0+2.0*((double)(imid)-f0*Tobs)));
    // c unchanged
    
    // interate backwards in phase
    dim *= -1.0;
    
    for(i = imid; i > imin; i--)
    {
        even = 2*i;
        odd = even + 1;
        f = (double)(i)/Tobs;
        
        sf = amplitude*B;
        sx = A;
        re = sf*(cosPhase_m+sx*cosPhase_p);
        im = -sf*(sinPhase_m+sx*sinPhase_p);
        
        hs[i] = re;
        hs[N-i] = im;
        
        A *= a;
        B *= (C*b);
        C *= c;
        
        /* Now update re and im for the next iteration. */
        recursive_phase_evolution(dre, dim, &cosPhase_m, &sinPhase_m);
        recursive_phase_evolution(dre, dim, &cosPhase_p, &sinPhase_p);
    }
    
    
}


void splinespace(int Ns, int istart, int iend, double *SM, double *freqs, double Tobs, double *splineA, double *splinef, int *Nk)
{
    int i, j, k, ii, jj, kk, kkold, n, Nx;
    int inc;
    double *lSM, *Sfit;
    double *im, *ym, *a;
    double *scale;
    double y2, cnt, xtol;
    double **M;
    double chisqquad, chisqlin, chisqconst;
    double x, z;
    FILE *out;
    
    im = double_vector(3);
    a = double_vector(2);
    ym = double_vector(2);
    M = double_matrix(2,2);
    
    lSM = double_vector(Ns);
    Sfit = double_vector(Ns);
    
    for(i=istart; i<= iend; i++) lSM[i] = log(SM[i]);
    
    Nx = 1;
    splinef[0] = freqs[istart];
    splineA[0] = lSM[istart];
    
    
    inc = (int)(Tobs*dfmin);
    // total number of chunks
    k = (iend-istart)/inc;
    
    scale = double_vector(k);

    
    for(i=0; i< k; i++)
    {
        
        y2 = 0.0;
        for(jj=0; jj< 2; jj++) ym[jj] = 0.0;
        cnt = 0.0;

        for(j=0; j< inc; j++)
        {
            cnt += 1.0;
            ii = istart+i*inc+j;
            y2 += lSM[ii]*lSM[ii];
            ym[0] += lSM[ii];
            ym[1] += cnt*lSM[ii];
        }
        
        
        isums(cnt, im);
        
        for(ii=0; ii< 2; ii++)
         {
             for(jj=0; jj< 2; jj++) M[ii][jj] = im[ii+jj];
         }
        
        n = 2;
        solve(n, M, a, ym);
        
        chisqlin = y2;
        for(ii=0; ii< n; ii++) chisqlin -= 2.0*a[ii]*ym[ii];
        for(ii=0; ii< n; ii++)
          {
              for(jj=0; jj< n; jj++) chisqlin += a[ii]*a[jj]*im[ii+jj];
          }
          
        z = chisqlin/cnt;
        
        scale[i] = z;
        
      }
    
    
    //
    
    // decide on the fit tolerance
    gsl_sort(scale, 1, k);
    jj = (int)(0.95*(double)(k));
    xtol = scale[jj];
    
    inc = (int)(Tobs*finc);
    // total number of chunks
    k = (iend-istart)/inc;
    
    
    y2 = 0.0;
    for(jj=0; jj< 2; jj++) ym[jj] = 0.0;
    cnt = 0.0;
    
    kkold = istart;
    
    for(i=0; i< k; i++)
    {

        for(j=0; j< inc; j++)
        {
            cnt += 1.0;
            ii = istart+i*inc+j;
            y2 += lSM[ii]*lSM[ii];
            ym[0] += lSM[ii];
            ym[1] += cnt*lSM[ii];
        }
        
        
        isums(cnt, im);
        
        for(ii=0; ii< 2; ii++)
         {
             for(jj=0; jj< 2; jj++) M[ii][jj] = im[ii+jj];
         }
        
        n = 2;
        solve(n, M, a, ym);
        
        chisqlin = y2;
        for(ii=0; ii< n; ii++) chisqlin -= 2.0*a[ii]*ym[ii];
        for(ii=0; ii< n; ii++)
          {
              for(jj=0; jj< n; jj++) chisqlin += a[ii]*a[jj]*im[ii+jj];
          }
        
        x = cnt/Tobs;
        z = chisqlin/cnt;
        
        if(x >= dfmin)
         {
             
          if(z > xtol || x >= dfmax)
           {
               
            kk = istart+(i+1)*inc-1;
        
            splinef[Nx] = freqs[kk];
            splineA[Nx] = lSM[kk];
               
            // store the linear fit for this segment
            cnt = 0.0;
            for(ii=kkold; ii< kk; ii++)
            {
              cnt += 1.0;
              Sfit[ii] = a[0] + a[1]*cnt;
            }
               
            kkold = kk;
        
            Nx++;
               
        
        // restart the spacing
            y2 = 0.0;
            for(jj=0; jj< 2; jj++) ym[jj] = 0.0;
            cnt = 0.0;
            
          }
        }
        
        
        
      }
    
    if(kk < iend)
    {
    splinef[Nx] = freqs[iend];
    splineA[Nx] = lSM[iend];
    Nx++;
    }
    
    for(ii=kk; ii<= iend; ii++)
    {
      cnt += 1.0;
      Sfit[ii] = a[0] + a[1]*cnt;
    }
    
    if(verbose == 1) printf("knots %d\n", Nx);
    
    if(Nx < 7) // need to add in some additional spline points
    {
        do
        {
            // find the knots with the largest spacing
            z = 0.0;
            for(i=0; i< Nx-1; i++)
            {
                x = splinef[i+1]-splinef[i];
                if(x > z)
                {
                    j = i;
                    z = x;
                }
            }
            
            x = (splinef[j+1]+splinef[j])/2.0;  // midway between the two points with largest spacing
            i = (int)(x*Tobs); // where it sits in the frequency array
            
            // add in a new spline knot
            splinef[Nx] = freqs[i];
            splineA[Nx] = lSM[i];
            Nx++;
            
            //sort the knots by frequency and apply the same ordering to Amplitudes
            gsl_sort2(splinef, 1, splineA, 1, Nx);
            
            
        }while(Nx < 7);
        
    }

    
    *Nk = Nx;
    
    if(verbose == 1) printf("There are %d smooth control points\n", Nx);
    
    free_double_matrix(M,2);
    free_double_vector(lSM);
    free_double_vector(Sfit);
    free_double_vector(a);
    free_double_vector(im);
    free_double_vector(ym);
    
}

void makespec(double *SM, double *freqs, int istart, int iend, int Nknot, double *splinef, double *splineA)
{
    int i, j;
    double f, y;
    gsl_spline   *aspline;
    gsl_interp_accel *acc;
    
    // smooth spectrum
    // Allocate spline
    aspline = gsl_spline_alloc(gsl_interp_akima, Nknot);
    acc = gsl_interp_accel_alloc();
    gsl_spline_init(aspline,splinef,splineA,Nknot);
    SM[istart] = exp(splineA[0]);
    SM[iend] = exp(splineA[Nknot-1]);
    for (i = istart+1; i < iend; ++i)
    {
        SM[i] = exp(gsl_spline_eval(aspline,freqs[i],acc));
    }
    gsl_spline_free (aspline);
    gsl_interp_accel_free (acc);
    
}

void isums(double x, double *is)
{
    double x2, x3, x4, x5;
    
    x2 = x*x;
    x3 = x2*x;
    
    is[0] = x;
    is[1] = (x2+x)/2.0;
    is[2] = (2.0*x3+3.0*x2+x)/6.0;
 
}

void solve(int n, double **M, double *av, double *yv)
{
    // solves the system M a = y for a
    
    int i, j, s;
    
    gsl_matrix *m = gsl_matrix_alloc (n, n);
    
    for (i = 0 ; i < n ; i++)
    {
        for (j = 0 ; j < n ; j++)
        {
            gsl_matrix_set(m, i, j, M[i][j]);
        }
    }
    
     gsl_vector *y = gsl_vector_alloc (n);
     
      for(i=0; i< n; i++) gsl_vector_set(y, i, yv[i]);
     
       gsl_vector *x = gsl_vector_alloc (n);
     
       gsl_permutation * p = gsl_permutation_alloc (n);
     
       gsl_linalg_LU_decomp (m, p, &s);
     
       gsl_linalg_LU_solve (m, p, y, x);
    
       for(i=0; i< n; i++) av[i] = gsl_vector_get(x, i);
     
       gsl_permutation_free (p);
       gsl_vector_free(x);
       gsl_vector_free(y);
       gsl_matrix_free(m);
    
       return;
}












//uses frequency domain data
void qscanf(double *data, double *smasd, double Q, double Tobs, double fmin, double fmax, int NT, int dec, double *feature)
{
    double dt;
    double *freqs;
    double x, dx, dlnf;
    double t, f;
    int octaves, Nf, i, j, k, l;
    double **tfDR, **tfDI, **tfD;
    double fac;
    
    FILE *out;
    
    dt = Tobs/(double)(NT);
    
    // logarithmic frequency spacing
    
    octaves = (int)(rint(log(fmax/fmin)/log(2.0))); // number of octaves
    Nf = qscan_subscale*octaves+1;
    freqs = (double*)malloc(sizeof(double)* (Nf));   // frequencies used in the analysis
    dx = log(2.0)/(double)(qscan_subscale);
    dlnf = dx;
    
    x = log(fmin);
    for(i=0; i< Nf; i++)
    {
        freqs[i] = exp(x);
        x += dx;
    }

    printf("frequency layers = %d fmin %e fmax %e\n", Nf, fmin, fmax);
    
    // arrays to hold the Qscan data
    tfDR = double_matrix(Nf,NT); // real (cosine)
    tfDI = double_matrix(Nf,NT); // imaginary (sine)
    tfD = double_matrix(Nf,NT);  // power
    
    // Wavelet transform
    TransformC(data, freqs, tfD, tfDR, tfDI, Q, Tobs, NT, Nf);
    
    // extract any non-Gaussian features
    nongaussian(data, freqs, smasd, tfD, tfDR, fmax, Q, Tobs,  NT, Nf, dec, feature);

    // save the Qscan data to file
    out = fopen("Qtransform.dat","w");
    k = 0;
    l = 0;
    x = 0.0;
    for(j = 0; j < Nf; j++)
    {
        f = freqs[j];
        
        for(i = 0; i < NT; i++)
        {
           
            t = (double)(i)*dt;
            
            if(t > 1.0 && t < Tobs-1.0)
            {
                fprintf(out,"%e %e %e\n", t, f, tfD[j][i]);
                x += tfD[j][i];
                k++;
                if(tfD[j][i] > 9.0) l++;
            }
        }
        
        fprintf(out,"\n");
    }
    fclose(out);
    
    // The spectrogram for whiten unit variance Gaussian data should follow
    // a chi-squared distribiton with two degrees of freedom. Mean = 2.
    // A threshold of 9 should correspond to 0.01111 of the samples
    printf("%f %f\n", x/(double)(k), (double)(l)/(double)(k));
    
    // Wavelet transform
    TransformC(data, freqs, tfD, tfDR, tfDI, Q, Tobs, NT, Nf);
    
    // save the Qscan data to file
    out = fopen("Qtransform_clean.dat","w");
    for(j = 0; j < Nf; j++)
    {
        f = freqs[j];
        
        for(i = 0; i < NT; i++)
        {
            t = (double)(i)*dt;
            if(t > 1.0 && t < Tobs-1.0) fprintf(out,"%e %e %e\n", t, f, tfD[j][i]);
        }
        
        fprintf(out,"\n");
    }
    fclose(out);
    
    free(freqs);
    free_double_matrix(tfDR,Nf);
    free_double_matrix(tfDI,Nf);
    free_double_matrix(tfD,Nf);
    
}

void nongaussian(double *data, double *freqs, double *smasd, double **tfD, double **tfDR, double fmax, double Q, double Tobs, int NT, int Nf, int dec, double *feature)
{
    int **live, **live2;
    int **cluster;
    double *Dtotal;
    int i, j, k, ii, jj, kk, ci, cf, flag, cmax;
    double sthresh_local, warm_local, S, scale, dt;
    double SNRcut;
    double **ttD;
    double sqf;
    double *cSNRsq;
    double Smax;
    int jmax;
    int *count, *list, *num;
    int *rlist;
    int *flist, *tlist;
    double **DT;
    double *Dblock;
    int imin, imax;
    double tmin, tmax, fmin, x;
    int itmin, itmax, ifmin, ifmax;
    
    FILE *out;
    
    live= int_matrix(Nf,NT);
    live2= int_matrix(Nf,NT);
    cluster = int_matrix(Nf,NT);
    
    dt = Tobs/(double)(NT);
    
    sthresh_local = 10.0;
    warm_local = 6.0;
    
    SNRcut = 5.0;
    
    ttD = double_matrix(Nf,NT); // used to reconstruct time domain signal
    
    // empirically determine the scaling for the time domain reconstruction
    scale = Getscale(freqs, Q, Tobs, fmax, NT, Nf);
    
    printf("%d %d\n", NT, Nf);

    for(j = 0; j < Nf; j++)
    {
        sqf = scale*sqrt(freqs[j]); // scaling used to convert to time doman
        for (i = 0; i < NT; i++)
        {
            ttD[j][i] = sqf*tfDR[j][i];
        }
    }
    
    
    k = 0;
    //  apply threshold
    for(j = 0; j < Nf; j++)
    {
        for (i = 0; i < NT; i++)
        {
            cluster[j][i] = 0;
            live[j][i] = -1;
            if(tfD[j][i] > sthresh_local)
            {
                live[j][i] = 1;
                k++;
            }
            live2[j][i] = live[j][i];
        }
    }
    
    printf("%d initial hot pixels\n", k);

    
    // dig deeper to extract clustered power
    for(j = 1; j < Nf-1; j++)
    {
        for (i = 1; i < NT-1; i++)
        {
            
            flag = 0;
            for(jj = -1; jj <= 1; jj++)
            {
                for(ii = -1; ii <= 1; ii++)
                {
                    if(live[j+jj][i+ii] == 1) flag = 1;
                }
            }
            
            if(flag == 1 && tfD[j][i] > warm_local) live2[j][i] = 1;
            
        }
    }
    
    k = 0;
    for(j = 0; j < Nf; j++)
    {
        for (i = 0; i < NT; i++)
        {
            if(live2[j][i] == 1) k++;
        }
    }
    printf("%d hot pixels\n", k);
    
    
    //  identify clusters
    int newClr;
    int size;
    int cnt;
    
    size = Nf*NT;
    newClr = 1;
    
    do
     {
        // find a new hot (=1) pixel and color new cluster
        flag = 0;
        cnt = 0;
        do
        {
            i = cnt%Nf;
            j = cnt/Nf;
            if(live2[i][j] == 1) flag = 1;
            cnt++;
        }while(flag == 0 && cnt < size);
        
        if(flag == 1)
        {
            
            newClr++;
            floodFill(live2, Nf, NT, i, j, newClr);
        }
        
        }while(flag == 1);
    
    cf = newClr-1;
    printf("%d clusters\n", cf);
    
    
    cSNRsq = double_vector(cf);
    DT = double_matrix(cf,NT); //  time domain signal for each cluster
    Dtotal = double_vector(NT);
    num = int_vector(cf);

     for(j = 0; j < cf; j++)
      {
         num[j] = 0;
      for (i = 0; i < NT; i++)
       {
        DT[j][i] = 0.0;
       }
     }
    
    // exclude boundaries impacted by Tukey window
    imin = (int)(t_rise/Tobs*(double)(NT));
    imax = (int)((Tobs-t_rise)/Tobs*(double)(NT));

    for(j = 0; j < Nf; j++)
    {
        for (i = imin; i < imax; i++)
        {
            if(live2[j][i] > 0)
            {
                jj = live2[j][i]-2; // label of the cluster
                DT[jj][i] += ttD[j][i];
                num[jj]++; // counts how many pixels in a cluster
            }
         }
     }


  Smax = 0.0;
   for(j = 0; j < cf; j++)
    {
        cSNRsq[j] = 0.0;
     for (i = imin; i < imax; i++)
      {
          cSNRsq[j] += DT[j][i]*DT[j][i];
      }
        cSNRsq[j] =  sqrt(cSNRsq[j]);
        if(cSNRsq[j] > Smax)
        {
            Smax = cSNRsq[j];
            jmax = j;
        }
        printf("SNR of cluster %d = %f\n", j, cSNRsq[j]);
   }
    
    printf("Loudest cluster is #%d SNR = %f\n", jmax, Smax);
    
    itmin = NT;
    itmax = 0;
    ifmin = Nf;
    ifmax = 0;
    k = 0;
    
    // frequency and time indixes of pixels in the loudest cluster
    flist = int_vector(num[jmax]);
    tlist = int_vector(num[jmax]);
    
    for(j = 0; j < Nf; j++)
    {
        for (i = imin; i < imax; i++)
        {
            if(live2[j][i] > 0)
              {
                  jj = live2[j][i]-2; // label of the final cluster
                  if(jj == jmax)
                  {
                      flist[k] = j;
                      tlist[k] = i;
                      k++;
                      
                      if(j < ifmin)
                      {
                          ifmin = j;
                          fmin = freqs[j];
                      }
                      if(j > ifmax)
                      {
                          ifmax = j;
                          fmax = freqs[j];
                      }
                      if(i < itmin)
                      {
                          itmin = i;
                          tmin = dt*(double)(i);
                      }
                      if(i > itmax)
                      {
                          itmax = i;
                          tmax = dt*(double)(i);
                      }
                  }
              }
        }
    }
    
    
    printf("tmin %f tmax %f fmin %f fmax %f\n", tmin, tmax, fmin, fmax);
    
 
    // total excess
    for (i = 0; i < NT; i++)
    {
        Dtotal[i] = 0.0;
    }
    for(j = 0; j < cf; j++)
    {
          for (i = 0; i < NT; i++)
           {
               Dtotal[i] += DT[j][i];
           }
    }

    // the downsampling changes the scaling of the reconstruction
    // this factor corrects for it
    // The SNRs etc calculated in Extract are designed to match those in PSD.c
    x = sqrt((double)(dec));
    
    out = fopen("excess.dat", "w");
    for(i=0; i< NT; i++)
    {
        fprintf(out,"%e %e\n", (double)(i)*dt, Dtotal[i]/x);
    }
    fclose(out);
    
    
    for (i = 0; i < NT; i++)
    {
        Dtotal[i] = 0.0;
    }
    // build the excess power model with threshold
    k = 0;
    for(j = 0; j < cf; j++)
    {
        if(cSNRsq[j] > SNRcut)
        {
            k++;
          for (i = 0; i < NT; i++)
           {
               Dtotal[i] += DT[j][i];
           }
        }
    }
    
    printf("%d significant cluster(s)\n", k);

    
    
    out = fopen("features.dat", "w");
    for(i=0; i< NT; i++)
    {
        fprintf(out,"%e %e\n", (double)(i)*dt, Dtotal[i]/x);
    }
    fclose(out);
    
    
    // Compute the excess power (relative to the current spectral model
    S = 0.0;
    for (i = 0; i < NT; ++i) S += Dtotal[i]*Dtotal[i];
    S = sqrt(S);
    
    printf("Excess SNR = %f\n", S);
    
    
    gsl_fft_real_radix2_transform(Dtotal, 1, NT);
    
    x = sqrt((double)(2*NT));
    // subtract the features (whitened, frequency domain)
    for(i=0; i< NT; i++) data[i] -= Dtotal[i]/x;
    
    
    S = 0.0;
    for (i = 1; i < NT/2; ++i)
    {
        S += 2.0*(Dtotal[i]*Dtotal[i]+Dtotal[NT-i]*Dtotal[NT-i]);
    }
    S = sqrt(S/(double)(NT));
    
    printf("Excess computed in frequency domain SNR = %f\n", S);
    
    // we don't try and reconstruct properly below 20 Hz as the un-whitening amplifies junk
    double f;
    double fcut = 20.0;
    
    // only use smooth spectrum in the un-whitening
    Dtotal[0] = 0.0;
    for(i=1; i< NT/2; i++)
    {
        x = smasd[i];
        f = (double)(i)/Tobs;
        if(f < fcut) x *= exp((f-fcut));
        Dtotal[i] *= x;
        Dtotal[NT-i] *= x;
    }
    Dtotal[NT/2] = 0.0;
    
    // re-colored template, frequency domain
    out = fopen("colored_template.dat", "w");
    for(i=1; i< NT/2; i++)
    {
        fprintf(out,"%e %e %e\n", (double)(i)/Tobs, Dtotal[i],  Dtotal[NT-i]);
    }
    fclose(out);

    

    gsl_fft_halfcomplex_radix2_inverse(Dtotal, 1, NT);
    
    // undoes the overall scaling from Getscale
    x = sqrt((double)(2*NT));
   
    // re-colored signal
    out = fopen("features_colored.dat", "w");
    for(i=0; i< NT; i++)
    {
        fprintf(out,"%e %e\n", (double)(i)*dt, Dtotal[i]/x);
    }
    fclose(out);
    
    for(i=0; i< NT; i++) feature[i] = Dtotal[i]/x;
    
    
    
    free(Dtotal);
    free(cSNRsq);
    free(num);
    free(flist);
    free(tlist);
    free_double_matrix(DT,cf);
    free_double_matrix(ttD,Nf);
    free_int_matrix(cluster,Nf);
    free_int_matrix(live,Nf);
    free_int_matrix(live2,Nf);
    
}


// Fill the image img[x][y] and all its same colored
// surrounding with the given new color
// Fill the image img[x][y] and all its same colored
// surrounding with the given new color
void floodFill(int **img, int m, int n, int x, int y, int newClr) {
    
    int prevClr = img[x][y];
    if (prevClr == newClr) return;
    
    int size = m*n;

    struct Point *queue = malloc(size * sizeof(struct Point));
    
    int front = 0, rear = 0;

    
    queue[rear].x = x;
    queue[rear].y = y;
    rear++;
    img[x][y] = newClr;

    while (front < rear) {
        struct Point p = queue[front++];
        x = p.x; y = p.y;

        // Check if the surrounding pixels are valid and enqueue
        if (x + 1 < m && img[x + 1][y] == prevClr) {
            img[x + 1][y] = newClr;
            queue[rear].x = x+1;
            queue[rear].y = y;
            rear++;
        }
        if (x - 1 >= 0 && img[x - 1][y] == prevClr) {
            img[x - 1][y] = newClr;
            queue[rear].x = x-1;
            queue[rear].y = y;
            rear++;
        }
        if (y + 1 < n && img[x][y + 1] == prevClr) {
            img[x][y + 1] = newClr;
            queue[rear].x = x;
            queue[rear].y = y+1;
            rear++;
        }
        if (y - 1 >= 0 && img[x][y - 1] == prevClr) {
            img[x][y - 1] = newClr;
            queue[rear].x = x;
            queue[rear].y = y-1;
            rear++;
        }
        if (x + 1 < m && y + 1 < n && img[x + 1][y + 1] == prevClr) {
            img[x + 1][y + 1] = newClr;
            queue[rear].x = x+1;
            queue[rear].y = y+1;
            rear++;
        }
        if (x - 1 >= 0 && y + 1 < n && img[x - 1][y + 1] == prevClr) {
            img[x - 1][y + 1] = newClr;
            queue[rear].x = x-1;
            queue[rear].y = y+1;
            rear++;
        }
        if (x - 1 >= 0 && y - 1 >= 0 && img[x - 1][y - 1] == prevClr) {
            img[x - 1][y - 1] = newClr;
            queue[rear].x = x-1;
            queue[rear].y = y-1;
            rear++;
        }
        if (x + 1 < m && y - 1 >= 0 && img[x + 1][y - 1] == prevClr) {
            img[x + 1][y - 1] = newClr;
            queue[rear].x = x+1;
            queue[rear].y = y-1;
            rear++;
        }
        
        
    }

    free(queue);
}




int *int_vector(int N)
{
    return malloc( (N+1) * sizeof(int) );
}

void free_int_vector(int *v)
{
    free(v);
}

int **int_matrix(int N, int M)
{
    int i;
    int **m = malloc( (N+1) * sizeof(int *));

    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(int));
    }

    return m;
}

void free_int_matrix(int **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_int_vector(m[i]);
    free(m);
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
