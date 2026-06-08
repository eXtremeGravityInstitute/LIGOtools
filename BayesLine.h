/*
 *  Copyright (C) 2018 Neil J. Cornish, Tyson B. Littenberg
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <assert.h>

#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>

#define TPI 6.2831853071795862319959269370884
#define lAwidth 2.3    // 2.3 is one decade
#define ZETA 0.5
#define kappa_BL 0.8
#define ufrac 0.4    // rj unifrom draw fraction 
#define FSTEP 10.0   // the stencil separation in Hz for the spline model. Going below 2 Hz is dangerous - will fight with line model
#define HMean 1      //0:Average 1:Harmonic Mean
#define linemul 9.0  // how much above the Gaussian floor a line needs to be
#define dfmin 2.0    // minimum smooth spacing
#define lowlat 1     //run on low latency settings to get  quick start psd model
#define t_rise 1.0  // Tukey roll-off in seconds

//static const gsl_rng_type *rngtype;
//static const gsl_rng *rng;

//##############################################
//OPEN MP

//gsl_rng **rvec;
//##############################################

typedef struct
{
  int n;
  int size;

  int *larray;

  double *Q;
  double *A;
  double *f;
  double *nu;
    
  int nf;
  int nnu;
  int wdth;
  int imin;
  double numin;
  double numax;
  double lnumin;
  double lnumax;
  double ***ltemplate;
  double *lt;

}lorentzianParams;

typedef struct
{
  int tmax;
  int ncut;
  int nmin;
  int sgmts;
  int n;
  int imin;
  int imax;

  double df;
  double fny;
  double Tobs;
  double fmin;
  double fmax;
  double flow;
  double fgrid;
  double fstep;
  double fhigh;
  double cadence;

}dataParams;

typedef struct
{
  int n;
  double *points;
  double *data;

}splineParams;

typedef struct
{
  double SAmin;
  double SAmax;
  double LQmin;
  double LQmax;
  double LAmin;
  double LAmax;

  //double *invsigma; //variances for each frequency bin
  double *sigma; //variances for each frequency bin
  double *upper; //variances for each frequency bin
  double *lower; //variances for each frequency bin
  double *mean;     //means for each frequency bin
  // double *upperSb; //variances for each frequency bin

}BayesLinePriors;

struct BayesLineParams
{
  dataParams *data;
  splineParams *spline;
  splineParams *spline_x;
  lorentzianParams *lines_x;
  BayesLinePriors *priors;

  double *Snf;
  double *Sna;
  double *fa;
  double *freq;
  double *power;
  double *spow;
  double *sfreq;
  double *Sbase;
  double *Sline;
  int maxBLLines;
  int constantLogLFlag;
  int flatPriorFlag;
  
  double TwoDeltaT;
  gsl_rng *r;
};


void BayesLineSetup(struct BayesLineParams *bptr, double *freqData, double fmin, double fmax, double deltaT, double Tobs);
void BayesLineFree(struct BayesLineParams *bptr);

void BayesLineRJMCMC(struct BayesLineParams *bayesline, double *freqData, double *psd, double *invpsd, double *splinePSD, int N, int cycle, double beta, int priorFlag, double *fprior, int SplineFlag);
void BayesLineInitialize(struct BayesLineParams *bayesline);

void BayesLineLorentzSplineMCMC        (struct BayesLineParams *bayesline, double heat, int steps, int priorFlag, double *dan, double *fprior, int SplineFlag);


double loglike_pm        (double *respow, double *Sn, double *Snx, int ilow, int ihigh);
double loglike_single    (double *respow, double *Sn, double *Snx, int ilowx, int ihighx, int ilowy, int ihighy);

double qdraw(double *fprop, double pmax, double flow, double fhigh, int ncut, double Tobs, gsl_rng *r);
double lprop(double f, double *fprop, dataParams *data);

void full_spectrum_single(double *Sn, double *Snx, dataParams *data, lorentzianParams *line_x, lorentzianParams *line_y, int ii, int *ilowx, int *ihighx, int *ilowy, int *ihighy);
void full_spectrum_add_or_subtract(double *Snew, double *Sold, dataParams *restrict data, lorentzianParams *restrict lines, int ii, int *ilow, int *ihigh, int flag);
void full_spectrum_spline(double *Sline, dataParams *restrict data, lorentzianParams *restrict lines);



void AkimaSplineGSL_one(int N, double *x, double *y, double xint, double *yint);
void CubicSplineGSL_one(int N, double *x, double *y, double xint, double *yint);
void CubicSplineGSL(int imin, int imax, int N, double *x, double *y, double *xint, double *yint);
void AkimaSplineGSL(int imin, int imax, int N, double *x, double *y, double *xint, double *yint);
void getrangeakima(int iu, int nsy, double *x, dataParams *restrict data, int Nend, int *imin, int *imax);
double delta_loglike(double *respow, double *Sn, double *Snx, int imin, int imax);

double rjdraw(double model, double sp, double prange, double pmin, gsl_rng *r);
double rjden(double model, double ref, double sp, double prange);
    
void create_dataParams(dataParams *data, double *f, int n, int N, double Tobs, int max_lines);

void create_lorentzianParams(lorentzianParams *lines, int size);
void create_lorentzianLook(lorentzianParams *lines);
void copy_lorentzianParams(lorentzianParams *origin, lorentzianParams *copy);
void copy_lorentzianLook(lorentzianParams *origin, lorentzianParams *copy);
void destroy_lorentzianParams(lorentzianParams *lines);

void create_splineParams(splineParams *spline, int size);
void copy_splineParams(splineParams *origin, splineParams *copy);
void destroy_splineParams(splineParams *spline);

void copy_bayesline_params(struct BayesLineParams *origin, struct BayesLineParams *copy);
void print_line_model(FILE *fptr, struct BayesLineParams *bayesline);
void print_spline_model(FILE *fptr, struct BayesLineParams *bayesline);
void parse_line_model(FILE *fptr, struct BayesLineParams *bayesline);
void parse_spline_model(FILE *fptr, struct BayesLineParams *bayesline);


double cut_lorentz(lorentzianParams *restrict ll, int N, double *SM, double *SL, double *PG, double *lf, double *lnu, double *lS, double Tobs);
void MaxLine(double *PS, double *SNA, double *SMA, double *lf, double *lh, double *lQ, double *lw, double *splinef, double *splineA, int *Nknt, int *Nln, double Tobs, double fmin, double fmax);
void blstart(lorentzianParams *line, double *data, double *residual, int N, double dt, double fmin, double fstep, int *Nsp, double *dspline, double *pspline, int max_lines);
void lmcmc(int M, double SM, double SL, double dr, double di, double xr, double xi, double *hr, double *hi, gsl_rng *r);
void faststart(lorentzianParams *line, double *data, int N, double *splinef, double *splineA, int *Nknt, int max_lines, double Tobs, double fmin, double fmax, double fstep, double fac);
void BayesLineBurnin(struct BayesLineParams *bayesline, double *timeData, double *freqData, char *ifo, double *fprior, int SplineFlag);
void lorentzgrid(int nf, int nnu, int wdth, double numin, double numax, double ***ltemplate, int N, double Tobs, double alpha);
void wlorentz(double f0, double nu, double S, int Tobs, int J, int N, int K, double *TWL, double *line);
void lorentzraw(double f0, double nu, double S, int J, double *freqs, double *pf);
double ***double_tensor(int N, int M, int L);
void free_double_tensor(double ***t, int N, int M);
void setup_lorentzian_lookup(lorentzianParams *lines, int N, double Tobs, int imin);
void add_windowed_lorentzian(double *Slines, double Tobs, lorentzianParams *lines, int ii, int imin, int imax);
int llook(lorentzianParams *restrict ll, double f0, double nu, double A, double Tobs);
double Lpeak(lorentzianParams *restrict ll, double *PG, double *SM, int N, int spread, double f0, double nu, double *A, double Tobs);
void Qscan(double *dataf, double *psd, double Q, double fmin, double fmax, double dt, int N);

