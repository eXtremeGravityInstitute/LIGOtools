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

/*********************************************************************************/
/*                                                                               */
/*     BayesLine fits the LIGO/Virgo power spectra using a model made up         */
/*     of N Lorentzian lines (described by central frequency f, quality          */
/*     factor Q and amplitude A) and cubic spline with M control points.         */
/*     The number of terms in each model, N, M, are free to vary via RJMCMC      */
/*     updates. The code initializes the models in a non-Markovian fashion,      */
/*     then refines the models with a full Markovian RJMCMC subroutine. This     */
/*     subroutine (LorentzMCMC) can be called by other codes to update the       */
/*     spectral fit to the residuals (data - signal model). Doing this is        */
/*     highly recommended, as it ensures that the spectral model is not          */
/*     eating any of the gravitational wave signal. Since the code is            */
/*     transdimensional (and very free in its movement between dimensions)       */
/*     it will not "over-fit" the spectral model.                                */
/*                                                                               */
/*********************************************************************************/


#include "BayesLine.h"
#include <omp.h>

#ifndef _OPENMP
#define omp ignore
#endif

//##############################################
//OPEN MP
gsl_rng **rvec;
//##############################################
/*
static void system_pause()
{
  printf("Press Any Key to Continue\n");
  getchar();
}
*/

static int *int_vector(int N)
{
    return malloc( (N+1) * sizeof(int) );
}
static double *double_vector(int N)
{
  return malloc( (N+1) * sizeof(double) );
}

static void free_double_vector(double *v)
{
  free(v);
}

static double **double_matrix(int N, int M)
{
  int i;
  double **m = malloc( (N+1) * sizeof(double *));
  
  for(i=0; i<N+1; i++)
  {
    m[i] = malloc( (M+1) * sizeof(double));
  }
  
  return m;
}

static void free_double_matrix(double **m, int N)
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




double qdraw(double *fprop, double pmax, double flow, double fhigh, int ncut, double Tobs, gsl_rng *r)
{
    int i;
    double alpha, beta;
    double f;
    
    do
    {
     f = flow + (fhigh-flow)*gsl_rng_uniform(r);  // draw a frequency
     i = (int)(floor((f-flow)*Tobs)); // map to a bin
     if(i <0) i = 0;
     if(i > ncut-1) i = ncut-1;
     beta = fprop[i]/pmax;
     alpha = gsl_rng_uniform(r);
    }while(alpha > beta);
    
    return f;
    
}

double lprop(double f, double *fprop, dataParams *data)
{
  int i;

  int ssize    = data->ncut;
  double flow  = data->flow;
  double Tobs  = data->Tobs;

  i = (int)((f-flow)*Tobs);
  if(i < 0) i=0;
  else if(i > ssize-1) i=ssize-1;

  return(log(fprop[i]));
}
static double logprior(double *lower, double *upper, double *Snf, int ilow, int ihigh)
{
  //double x;
  double lgp,dS;
  int i;

  //leaving off normalizations since they cancel in Hastings ratio
  lgp = 0;
  for(i=ilow; i<ihigh; i++)
  {
    if(Snf[i]>upper[i])
    {
      dS = log(Snf[i]) - log(upper[i]);
      lgp -= 0.5*dS*dS;
    }
    if(Snf[i]<lower[i])
    {
      dS = log(Snf[i]) - log(lower[i]);
      lgp -= 0.5*dS*dS;
    }
  }
  return (lgp);
}

/*
static double logprior_gaussian_model(double *mean, double *sigma, double *Snf, double *spline_f, int spline_n, double *lines_f, int lines_n, dataParams *data)
{
  double x,f;
  double lgp;
  int i,n;

  double flow  = data->flow;
  double Tobs  = data->Tobs;

  lgp = 0.0;

  //Lorentzian model
  for(n=0; n<lines_n; n++)
  {
    f = lines_f[n];
    i = (int)((f-flow)*Tobs);
    x = (mean[i] - Snf[i])/sigma[i];
    lgp -= x*x;


//    printf("    line %i:  f=%g, PSD=%g, , P=%g, sigma=%g, logP=%g\n",n,f,Snf[i],mean[i],fabs(x),lgp);
  }

  //Spline model
  for(n=0; n<spline_n; n++)
  {
    f = spline_f[n];
    i = (int)((f-flow)*Tobs);
    x = (mean[i] - Snf[i])/sigma[i];
    lgp -= x*x;

//    printf("    spline %i:  f=%g, PSD=%g, , P=%g, sigma=%g, logP=%g\n",n,f,Snf[i],mean[i],fabs(x),lgp);
  }

//  system_pause();

  return (0.5*lgp);
}
*/

static double logprior_gaussian(double *mean, double *sigma, double *Snf, int ilow, int ihigh)
{
  double x;
  double lgp;
  int i;

  //leaving off normalizations since they cancel in Hastings ratio
  lgp = 0;
  
  for(i=ilow; i<ihigh; i++)
  {
     x = (mean[i] - Snf[i])/sigma[i];
     lgp -= x*x;
  }
  
  return (0.0*lgp);
}

static double loglike(double *respow, double *Snf, int ncut)
{
  double lgl, x;
  int i;

  // leavimng out the log(2Pi) terms since they cancel in Hastings ratio
  lgl = 0.0;
  for(i=0; i< ncut; i++)
  {
    x = respow[i]/Snf[i];
    lgl -= (x+log(Snf[i]));
  }

  return(lgl);
}

double delta_loglike(double *respow, double *Sn, double *Snx, int imin, int imax)
{
  double lgl, x;
  int i;

  // leaving out the log(2Pi) terms since they cancel in Hastings ratio
  lgl = 0.0;
  for(i=imin; i< imax; i++)
  {
     x = respow[i]/Sn[i]-respow[i]/Snx[i];
     lgl -= (x+log(Sn[i]/Snx[i]));
  }
 
  return(lgl);
}
double loglike_pm(double *respow, double *Sn, double *Snx, int ilow, int ihigh)
{
  double lgl, x;
  int i;

  // leavimng out the log(2Pi) terms since they cancel in Hastings ratio
  lgl = 0.0;
  for(i=ilow; i< ihigh; i++)
  {
    x = respow[i]/Sn[i]-respow[i]/Snx[i];
    lgl -= (x+log(Sn[i]/Snx[i]));
  }

  return(lgl);
}



double loglike_single(double *respow, double *Sn, double *Snx, int ilowx, int ihighx, int ilowy, int ihighy)
{
  double lgl, x;
  int i;
  int ilow, ihigh;
  int imid1, imid2;

  ilow = ilowx;
  if(ilowy < ilow) ilow = ilowy;

  if(ilow == ilowx)
  {
    if(ihighx <= ilowy)  // separate regions
    {
      imid1 = ihighx;
      imid2 = ilowy;
      ihigh = ihighy;
    }

    if(ihighx > ilowy) // overlapping regions
    {
      if(ihighx < ihighy)
      {
        imid1 = ihighx;
        imid2 = ihighx;
        ihigh = ihighy;
      }
      else
      {
        imid1 = ilowy;
        imid2 = ilowy;
        ihigh = ihighx;
      }
    }
  }

  if(ilow == ilowy)
  {
    if(ihighy <= ilowx)  // separate regions
    {
      imid1 = ihighy;
      imid2 = ilowx;
      ihigh = ihighx;
    }

    if(ihighy > ilowx) // overlapping regions
    {
      if(ihighy < ihighx)
      {
        imid1 = ihighy;
        imid2 = ihighy;
        ihigh = ihighx;
      }
      else
      {
        imid1 = ilowx;
        imid2 = ilowx;
        ihigh = ihighy;
      }
    }
  }

  // leavimng out the log(2Pi) terms since they cancel in Hastings ratio
  lgl = 0.0;
  for(i=ilow; i< imid1; i++)
  {
    x = respow[i]/Sn[i]-respow[i]/Snx[i];
    lgl -= (x+log(Sn[i]/Snx[i]));
  }
  for(i=imid2; i< ihigh; i++)
  {
    x = respow[i]/Sn[i]-respow[i]/Snx[i];
    lgl -= (x+log(Sn[i]/Snx[i]));
  }

  return(lgl);
}

void full_spectrum_add_or_subtract(double *Snew, double *Sold, dataParams *restrict data, lorentzianParams *restrict lines, int ii, int *ilow, int *ihigh, int flag)
{
  int i;
  double dS,f2,f4;
  double deltf;
  double fsq, x, z, deltafmax, spread;
  double amplitude;
  int istart, istop, imid, idelt;

  double A = lines->A[ii];
  double f = lines->f[ii];
  double nu = lines->nu[ii];
  
  int    ncut = data->ncut;
  double Tobs = data->Tobs;
  double flow = data->flow;

    
  // copy over current model
  memcpy(Snew, Sold, ncut*sizeof(double));
   
  imid = llook(lines, f, nu, A, Tobs);
  imid -= lines->imin; // The Sline etc arrays are offset so 0 is the minimum frequency bin
  idelt = lines->wdth/2;

  istart = imid-idelt;
  istop = imid+idelt;
  if(istart < 0) istart = 0;
  if(istop > ncut) istop = ncut;
    
  *ilow  = istart;
  *ihigh = istop;

  // add or remove the old line
  for(i=istart; i<istop; i++)
  {
    dS = lines->lt[i-istart];
    switch(flag)
    {
      case 1: //add new line
        Snew[i] += dS;
        break;
      case -1: //remove line
        Snew[i] -= dS;
        break;
       default:
        break;
    }
  }
    
}

void full_spectrum_single(double *Sn, double *Snx, dataParams *data, lorentzianParams *line_x, lorentzianParams *line_y, int ii, int *ilowx, int *ihighx, int *ilowy, int *ihighy)
{

  double *Stemp = malloc((size_t)(sizeof(double)*(data->ncut)));

  full_spectrum_add_or_subtract(Stemp, Snx, data, line_x, ii, ilowx, ihighx,-1);
  full_spectrum_add_or_subtract(Sn, Stemp, data, line_y, ii, ilowy, ihighy, 1);

  free(Stemp);
}



void full_spectrum_spline(double *Sline, dataParams *restrict data, lorentzianParams *restrict lines)
{
  int i, j, k;
  int istart, istop;

  double *Stemp = malloc((size_t)(sizeof(double)*(data->ncut)));

  for(i=0; i<data->ncut; i++) Sline[i] = 0.0;
  for(k=0; k<lines->n; k++)
  {
    memcpy(Stemp,Sline,data->ncut*sizeof(double));
    full_spectrum_add_or_subtract(Sline, Stemp, data, lines, k, &istart, &istop, 1);
  }

  free(Stemp);
}

void AkimaSplineGSL_one(int N, double *x, double *y, double xint, double *yint)
{
  int n;
   
  /* set up GSL akima spline */
  gsl_spline *aspline = gsl_spline_alloc(gsl_interp_akima, N);
  gsl_interp_accel *acc    = gsl_interp_accel_alloc();
  
  /* get derivatives */
  gsl_spline_init(aspline,x,y,N);
    
  /*  GSL akima spline throws errors if
     interpolated points are at end of
     spline control points*/
     
    if     (xint<x[0])
      *yint = y[0];
    else if(xint>x[N-1])
      *yint = y[N-1];
    else
      *yint=gsl_spline_eval(aspline,xint,acc);   
    
  gsl_spline_free (aspline);
  gsl_interp_accel_free (acc);

}

void CubicSplineGSL_one(int N, double *x, double *y, double xint, double *yint)
{
  int n;
   
  /* set up GSL akima spline */
  gsl_spline *cspline = gsl_spline_alloc(gsl_interp_cspline, N);
  gsl_interp_accel *acc    = gsl_interp_accel_alloc();
  
  /* get derivatives */
  gsl_spline_init(cspline,x,y,N);
    
  /*  GSL cubic spline throws errors if
     interpolated points are at end of
     spline control points*/
     
    if     (xint<x[0])
      *yint = y[0];
    else if(xint>x[N-1])
      *yint = y[N-1];
    else
      *yint=gsl_spline_eval(cspline,xint,acc);
  
  gsl_spline_free (cspline);
  gsl_interp_accel_free (acc);

}


void CubicSplineGSL(int imin, int imax, int N, double *x, double *y, double *xint, double *yint)
{
  int n;

  /* set up GSL cubic spline */
  gsl_spline       *cspline = gsl_spline_alloc(gsl_interp_cspline, N);
  gsl_interp_accel *acc    = gsl_interp_accel_alloc();

  /* get derivatives */
  gsl_spline_init(cspline,x,y,N);

  /* interpolate */
  for(n=imin; n<imax; n++)
  {
    /*
     GSL cubic spline throws errors if
     interpolated points are at end of
     spline control points
     */
    if     (xint[n]<x[0])
      yint[n] = y[0];
    else if(xint[n]>x[N-1])
      yint[n] = y[N-1];
    else
      yint[n]=gsl_spline_eval(cspline,xint[n],acc);
  }

  gsl_spline_free (cspline);
  gsl_interp_accel_free (acc);

}

void AkimaSplineGSL(int imin, int imax, int N, double *x, double *y, double *xint, double *yint)
{
  int n;
   
  /* set up GSL akima spline */
  gsl_spline *aspline = gsl_spline_alloc(gsl_interp_akima, N);
  gsl_interp_accel *acc    = gsl_interp_accel_alloc();
  
  /* get derivatives */
  gsl_spline_init(aspline,x,y,N);
  
  /* interpolate */
  for(n=imin; n<imax; n++)
  {
    /*
     GSL akima spline throws errors if
     interpolated points are at end of
     spline control points
     */
     
    if     (xint[n]<x[0])
      yint[n] = y[0];
    else if(xint[n]>x[N-1])
      yint[n] = y[N-1];
    else
      yint[n]=gsl_spline_eval(aspline,xint[n],acc);
    
  }
 
  gsl_spline_free (aspline);
  gsl_interp_accel_free (acc);

}

void getrangeakima(int iu, int nsy, double *x, dataParams *restrict data, int Nend, int *imin, int *imax)
{
  int Nst;
  Nst = 3;

  if(iu>Nst-1)
  {
      *imin = (int)((floor)((x[iu-Nst]-x[0])*data->Tobs));
  }
  else
  {
     *imin = (int)(0);
  }
  
  if(iu<nsy-Nst)
  {
     *imax = (int)((ceil)((x[iu+Nst]-x[0])*data->Tobs +1));
  }
  else
  {
     *imax = (int)(Nend);
  }
}

void print_line_model(FILE *fptr, struct BayesLineParams *bayesline)
{
  int j;
  
  fprintf(fptr,"%i ", bayesline->lines_x->n);
  for(j=0; j< bayesline->lines_x->n; j++) fprintf(fptr,"%lg %lg %lg ", bayesline->lines_x->f[j], bayesline->lines_x->A[j], bayesline->lines_x->nu[j]);
  fprintf(fptr,"\n");

}
void print_spline_model(FILE *fptr, struct BayesLineParams *bayesline)
{
  int j;

  fprintf(fptr,"%i ", bayesline->spline_x->n);
  for(j=0; j<bayesline->spline_x->n; j++) fprintf(fptr,"%.4lf %lg ",bayesline->spline_x->points[j],bayesline->spline_x->data[j]);
  fprintf(fptr,"\n");
}

void parse_line_model(FILE *fptr, struct BayesLineParams *bayesline)
{
  int j;

  fscanf(fptr,"%i",&bayesline->lines_x->n);
  for(j=0; j< bayesline->lines_x->n; j++) fscanf(fptr,"%lg %lg %lg",&bayesline->lines_x->f[j],&bayesline->lines_x->A[j],&bayesline->lines_x->nu[j]);
}

void parse_spline_model(FILE *fptr, struct BayesLineParams *bayesline)
{
  int j;

  fscanf(fptr,"%i",&bayesline->spline_x->n);
  for(j=0; j<bayesline->spline_x->n; j++)fscanf(fptr,"%lg %lg",&bayesline->spline_x->points[j],&bayesline->spline_x->data[j]);
}


void create_dataParams(dataParams *data, double *f, int n, int N, double Tobs, int max_lines)
{
  int nmin;

  data->Tobs = Tobs;

  //minimum frequency
  data->fmin = f[0];
    
  // number of frequencies in the data (N/2)
  data->n = N/2;

  // frequency resolution
  data->df = 1.0/data->Tobs;
    
    // sample cadence in Hz, this should be read in from the frame file
  data->cadence = (double)(N)/data->Tobs;
    
    // Nyquist frequency
  data->fny = data->cadence/2.0;
    
  //minimum Fourier bin
  nmin = (int)floor(f[0]*data->Tobs + 0.5);
  data->nmin = nmin;

  // maximum analysis frequency (exclusive)
  data->fmax = (double)(nmin+n)*data->df;
  if(data->fmax > data->fny) data->fmax = data->fny;
    
  // maximum frequency set (initializing to fmax)
  data->fhigh = data->fmax;

  // size of segments in Hz
  // If frequency snippets are too large need longer initial runs to get convergence
  data->fstep = 1./data->Tobs;//data->fmin;//16;//((data->fmax-data->fmin)/128);//256./data->Tobs;//30;//FSTEP;//9.0;//30.0;

  // This sets the maximum number of Lorentzian lines.
  data->tmax = max_lines;

  // approximate number of segments
  data->sgmts = (int)((f[n-1]-f[0])/data->fstep)+2;

  // the stencil separation in Hz for the spline model. Going below 2 Hz is dangerous - will fight with line model
  data->fgrid = 2.0;//data->fstep/2.;//15.0;//FSTEP;///4;//4.0;//FSTEP;
}

// Only a subset of the lorentzian parameters get allocated here.
// The rest are allocated when the lookup table is read in
void create_lorentzianParams(lorentzianParams *lines, int size)
{
  lines->n    = 0;
  lines->size = size;
  lines->nf   = 0;
  lines->nnu  = 0;
  lines->wdth = 0;
  lines->imin = 0;
  lines->numin  = 0.0;
  lines->numax  = 0.0;
  lines->lnumin = 0.0;
  lines->lnumax = 0.0;

  lines->larray = malloc((size_t)(sizeof(int)*size));

  lines->f = malloc((size_t)(sizeof(double)*size));
  lines->Q = malloc((size_t)(sizeof(double)*size));
  lines->A = malloc((size_t)(sizeof(double)*size));
  lines->nu = malloc((size_t)(sizeof(double)*size));
  lines->lt = NULL;
  lines->ltemplate = NULL;
	    
}

// The rest of the lorentzian model is allocated here after the lookup table is read in
void create_lorentzianLook(lorentzianParams *lines)
{

  lines->ltemplate = double_tensor(lines->nf+1,lines->nnu+1,lines->wdth);
  lines->lt = malloc(lines->wdth*sizeof(double));
    
}

void setup_lorentzian_lookup(lorentzianParams *lines, int N, double Tobs, int imin)
{
  int i, j, k;
  int nf = 0;
  int nnu = 0;
  int wdth = 0;
  int read_lookup = 0;
  double numin = 0.0;
  double numax = 0.0;
  double alpha = (2.0*t_rise/Tobs);
  double ***ltemplate = NULL;
  char filename[1024];
  FILE *in;
  FILE *out;

  sprintf(filename, "lookup_%d_%.2f.dat", (int)(Tobs), t_rise);
  in = fopen(filename,"r");
  if(in)
  {
    if(fscanf(in,"%d%d%d%lf%lf\n", &nf, &nnu, &wdth, &numin, &numax) == 5)
    {
      ltemplate = double_tensor(nf+1,nnu+1,wdth);
      read_lookup = 1;

      for(i=0; i<=nf; i++)
      {
        for(j=0; j<=nnu; j++)
        {
          for(k=0; k<wdth; k++)
          {
            if(fscanf(in,"%lf", &ltemplate[i][j][k]) != 1) read_lookup = 0;
          }
        }
      }
    }
    fclose(in);

    if(!read_lookup && ltemplate != NULL)
    {
      free_double_tensor(ltemplate, nf+1, nnu+1);
      ltemplate = NULL;
    }
  }

  if(!read_lookup)
  {
    numin = 1.0e-4;
    numax = 1.0e-1;
    nf = 40;
    nnu = 20;
    wdth = (int)(16.0*Tobs);
    ltemplate = double_tensor(nf+1,nnu+1,wdth);

    lorentzgrid(nf, nnu, wdth, numin, numax, ltemplate, N, Tobs, alpha);

    out = fopen(filename,"w");
    if(out)
    {
      fprintf(out,"%d %d %d %e %e\n", nf, nnu, wdth, numin, numax);
      for(i=0; i<=nf; i++)
      {
        for(j=0; j<=nnu; j++)
        {
          for(k=0; k<wdth; k++)
          {
            fprintf(out,"%.16e\n", ltemplate[i][j][k]);
          }
        }
      }
      fclose(out);
    }
  }

  lines->imin = imin;
  lines->nf = nf;
  lines->nnu = nnu;
  lines->wdth = wdth;
  lines->numin = numin;
  lines->numax = numax;
  lines->lnumin = log(numin);
  lines->lnumax = log(numax);

  create_lorentzianLook(lines);
  for(i=0; i<=nf; i++)
  {
    for(j=0; j<=nnu; j++)
    {
      for(k=0; k<wdth; k++)
      {
        lines->ltemplate[i][j][k] = ltemplate[i][j][k];
      }
    }
  }

  free_double_tensor(ltemplate, nf+1, nnu+1);
}

void add_windowed_lorentzian(double *Slines, double Tobs, lorentzianParams *lines, int ii, int imin, int imax)
{
  int i, j, k;

  k = llook(lines, lines->f[ii], lines->nu[ii], lines->A[ii], Tobs);
  for(i=0; i<lines->wdth; i++)
  {
    j = k-lines->wdth/2+i;
    if(j>=imin && j<imax) Slines[j] += lines->lt[i];
  }
}

// all the lorentz parameters are copied here, except the lookup
void copy_lorentzianParams(lorentzianParams *origin, lorentzianParams *copy)
{
  copy->n    = origin->n;
  copy->size = origin->size;

  int n, m, k;
 for(n=0; n<origin->n; n++)
  {
    copy->larray[n] = origin->larray[n];

    copy->f[n] = origin->f[n];
    copy->Q[n] = origin->Q[n];
    copy->A[n] = origin->A[n];
    copy->nu[n] = origin->nu[n];
  }
    
    copy->imin= origin->imin;
    copy->nf = origin->nf;
    copy->nnu = origin->nnu;
    copy->wdth = origin->wdth;
    
    copy->numin = origin->numin;
    copy->numax = origin->numax;
    copy->lnumin = origin->lnumin;
    copy->lnumax = origin->lnumax;
    
}

// the lookup is copied here. Need to have copied the parameters first
void copy_lorentzianLook(lorentzianParams *origin, lorentzianParams *copy)
{

  int n, m, k;
    
    for(n=0; n<origin->wdth; n++)
    {
        copy->lt[n] = origin->lt[n];
    }

    for (n = 0; n <= origin->nf; ++n)
         {
            for (m = 0; m <= origin->nnu; ++m)
            {
                for (k = 0; k < origin->wdth; ++k)
                {
                    copy->ltemplate[n][m][k] = origin->ltemplate[n][m][k];
                }
            }
        }

}


// all the lorentz arrays are freed here
void destroy_lorentzianParams(lorentzianParams *lines)
{
  if(lines==NULL) return;

  free(lines->larray);
  free(lines->f);
  free(lines->Q);
  free(lines->A);
  free(lines->nu);
  if(lines->lt!=NULL) free(lines->lt);
  if(lines->ltemplate!=NULL) free_double_tensor(lines->ltemplate,lines->nf+1,lines->nnu+1);
  free(lines);
}

void create_splineParams(splineParams *spline, int size)
{
  spline->n = size;

  spline->data   = malloc((size_t)(sizeof(double)*size));
  spline->points = malloc((size_t)(sizeof(double)*size));
}

void copy_splineParams(splineParams *origin, splineParams *copy)
{
  int n;

  copy->n = origin->n;

  for(n=0; n<origin->n; n++)
  {
    copy->data[n]   = origin->data[n];
    copy->points[n] = origin->points[n];
  }
}

void destroy_splineParams(splineParams *spline)
{
  if(spline==NULL) return;

  free(spline->data);
  free(spline->points);
  free(spline);
}

void BayesLineLorentzSplineMCMC(struct BayesLineParams *bayesline, double heat, int steps, int priorFlag, double *dan, double *fprop, int SplineFlag)
{
  int nsy;
  double logLx, logLy=0.0, logH;
  int ilowx, ihighx, ilowy, ihighy, imin, imax;
  int i, j, k=0, ki=0, ii=0, jj=0, ji, newki, mc, iu;
  int sl;
  int check=0;
  int lbl;
  double alpha, alpha1;
  double lSAmax, lSAmin;
  double lQmin, lQmax;
  double lAmin, lAmax;
  double lnumin, lnumax;
  double numin, numax;
  int *ac, *cc;
  double *Sn, *Sbase, *Sbasex, *Sline, *Snx;
  double *xint;
  double e1, e2, e3, e4;
  double x2, x3, x4;
  double s2, s3, s4;
  int typ=0;
  double xsm, pnorm, pmax, fcl, fch, dff;
  double baseav;
  double logpx=0.0, logpy=0.0, logqx=0.0,logqy=0.0,x, y, z, beta;
  double logPsy=0.0,logPsx=0.0;
  double Ac, newfreq, newfreql;
  double *sdatay;
  double *spointsy;
  double mdl, sp, prange;
  double shiftval;
  int *foc;
   
  /* Make local pointers to BayesLineParams structure members */
  dataParams *data           = bayesline->data;
  lorentzianParams *lines_x  = bayesline->lines_x;
  splineParams *spline       = bayesline->spline;
  splineParams *spline_x     = bayesline->spline_x;
  BayesLinePriors *priors    = bayesline->priors;
  double *Snf                = bayesline->Snf;
  double *freq               = bayesline->sfreq;
  double *power              = bayesline->spow;
  gsl_rng *r                 = bayesline->r;

  int ncut = data->ncut;
  int tmax = data->tmax;

  double flow  = data->flow;
  double fhigh = data->fhigh;

  int nspline   = spline->n;

  double *sdata = spline->data;
  double *sdatax = spline_x->data;

  double *spoints  = spline->points;
  double *spointsx = spline_x->points;
    
    FILE *temp;

  Snx    = malloc((size_t)(sizeof(double)*(ncut)));
  Sn     = malloc((size_t)(sizeof(double)*(ncut)));
  Sbasex = malloc((size_t)(sizeof(double)*(ncut)));
  Sbase  = malloc((size_t)(sizeof(double)*(ncut)));
  Sline  = malloc((size_t)(sizeof(double)*(ncut)));

  sdatay   = malloc((size_t)(sizeof(double)*(nspline)));
  spointsy = malloc((size_t)(sizeof(double)*(nspline)));

  foc     = malloc((size_t)(sizeof(int)*(tmax)));
    
    baseav = 0.0;
    xint = malloc((size_t)(sizeof(double)*(ncut)));

    //Interpolate {spointsx,sdatax} ==> {freq,xint}
    if(SplineFlag == 1) AkimaSplineGSL(0,ncut,spline_x->n,spointsx,sdatax,freq,xint);
    else if(SplineFlag == 0) CubicSplineGSL(0,ncut,spline_x->n,spointsx,sdatax,freq,xint);
    
    for(i=0; i<ncut; i++)
    {
      Sbase[i] = exp(xint[i]);
      Sbasex[i] = Sbase[i];
      if(HMean ==0) baseav += xint[i];
      if(HMean ==1) baseav += pow(xint[i],-1.0);
    }

    baseav /= (double)(ncut);
    if(HMean ==1) baseav = pow(baseav,-1.0);

  // This keeps track of whos who in the Lorentzian model
  // Necessary complication when using delta likelihood
  // calculations that only update a single line
  lorentzianParams *lines_y = malloc(sizeof(lorentzianParams));
  create_lorentzianParams(lines_y,tmax);
  // need a full copy here. Need to copy paramters before allocating
  // and copying lookup table
  copy_lorentzianParams(lines_x, lines_y);
  create_lorentzianLook(lines_y);
  copy_lorentzianLook(lines_x, lines_y);
  
  // maxima and minima for the noise spectal slopes and amplitudes
  // uniform priors in slope and log amplitude
  lAmin = log(priors->LAmin);
  lAmax = log(priors->LAmax);
  lnumin = lines_x->lnumin;
  lnumax = lines_x->lnumax;
  numin = lines_x->numin;
  numax = lines_x->numax;
  // lSAmin = log(priors->SAmin);
  // lSAmax = log(priors->SAmax);
    
    // scalings for the line updates
    s2 = 0.5;
    s3 = 1.0e-3;
    s4 = 0.1/data->Tobs;

  for(i=0; i<lines_x->n; i++)
  {
    lines_x->larray[i] = i;
    lines_y->larray[i] = i;
  }

    /*
 temp=fopen("start_psd_bw.dat","w");
 for(i=0; i<ncut; i++) fprintf(temp,"%lg %lg %lg %lg\n",freq[i],Snf[i],priors->lower[i],priors->upper[i]);
 fclose(temp);
     */

  full_spectrum_spline(Sline, data, lines_x);
  for(i=0; i< ncut; i++) 
  {
      Sn[i] = Sbase[i]+Sline[i];
  }

  for(i=0; i<ncut; i++) Snx[i] = Sn[i];

  
  if(!bayesline->constantLogLFlag)
    logLx = loglike(power, Sn, ncut);
  else
    logLx = 1.0;

  if(priorFlag==1)
  {
    logPsx = logprior(priors->lower, priors->upper, Snx, 0, ncut);
  }
  if(priorFlag==2)
  {
    logPsx = logprior_gaussian(priors->mean, priors->sigma, Snx, 0, ncut);
  }

   
  pmax = -1.0;
  for(i=0; i< ncut; i++)
  {
     if(fprop[i] > pmax) pmax = fprop[i];
  }
    
    ac = int_vector(10);
    cc = int_vector(10);
    
    for(i=0; i< 10; i++)
    {
        ac[i] = 0;
        cc[i] = 1;
    }


  if(dfmin == 0.0) shiftval = (double)(gsl_rng_uniform(r));
  else shiftval = (double)(gsl_rng_uniform(r)*(dfmin));
  if(shiftval < (1.0/data->Tobs)) shiftval = 1.0/data->Tobs;
  if(shiftval >=dfmin) shiftval = dfmin;
    
    if(SplineFlag == 0)
    { 
        if(dfmin<2.0) printf("Warning: The spline points are too close.\n Cubic Spline interpolation can result in Wiggles.\n");
    }
    
    //Added
   for(i=0; i< lines_x->n; i++)
   {
      lines_y->nu[i] = lines_x->nu[i];
      lines_y->f[i] = lines_x->f[i];
      lines_y->A[i] = lines_x->A[i];
   }
   lines_y->n = lines_x->n;
   nsy = spline_x->n;
  
    for(i=0; i<spline_x->n; i++)
    {
      sdatay[i] = sdatax[i];
      spointsy[i] = spointsx[i];
    }
    
  for(mc=0; mc < steps; mc++)
  {
    typ=-1;
    check = 0;

    //copy over current state
    lines_y->n = lines_x->n;
    nsy = spline_x->n;
    
    for(i=0; i< lines_x->n; i++)
    {
      lines_y->nu[i] = lines_x->nu[i];
      lines_y->f[i] = lines_x->f[i];
      lines_y->A[i] = lines_x->A[i];
    }
      
    for(i=0; i<nsy; i++)
    {
      sdatay[i] = sdatax[i];
      spointsy[i] = spointsx[i];
    }
    
    beta = gsl_rng_uniform(r);
    logpx = 0.0; logpy = 0.0; logqx = 0.0; logqy = 0.0;

    if(beta > 0.5)  // update the smooth part of the spectrum
    {
        sl = 0;
        
      alpha = gsl_rng_uniform(r);

      logpx = 0.0; logpy = 0.0; logqx = 0.0; logqy = 0.0;
      
      if(alpha > 0.5)  // try a transdimensional move
      {
          lbl = 0;
         
        //decide between adding or removing spline point
        alpha1 = gsl_rng_uniform(r);
          
        if(alpha1 > 0.5)  // try and add a new term
        {
          nsy = spline_x->n+1;
          typ = 5;
        }
        else // try and remove term
        {
          nsy = spline_x->n-1;
          typ = 6;
        }
        
        //Updated birth-death proposal
        if(nsy >=7 && nsy <= nspline-1)
        {
          //Spline death move
          if(typ==6)
          {
            ki=1+(int)(gsl_rng_uniform(r)*(double)(spline_x->n-2)); // pick a term to try and kill - cant be first or last term
            k = 0;
            for(j=0; j<spline_x->n; j++)
            {
              if(j != ki)
              {
                sdatay[k] = sdatax[j];
                spointsy[k] = spointsx[j];
                k++;
              }
            }
            iu = ki;
            newfreq = spointsx[iu];
            
            //Interpolate to find the value it would have, if that point was not there
            if(SplineFlag == 1) AkimaSplineGSL_one(nsy,spointsy,sdatay,spointsx[ki],&mdl);
            else if(SplineFlag == 0) CubicSplineGSL_one(nsy,spointsy,sdatay,spointsx[ki],&mdl);
            
            //Proposal density
            ji = (int)((newfreq-flow)*data->Tobs);
            sp = fabs((log(priors->lower[ji]))*1.0e-3);
            prange = (log(priors->lower[ji]*100.0) - log(priors->lower[ji]));
            
            logqx = rjden(mdl,sdatax[ki],sp,prange);
            logpx = -log(prange);
            logqy = 0.0; logpy = 0.0;
              
          }//end death move

          if(typ==5)
          {
            //Add a new point at a random freq location

            //pick a freq location
             newfreq = flow + (gsl_rng_uniform(r)*((fhigh-flow)));
             spointsy[nsy-1] = newfreq;
             
             if(SplineFlag == 1) AkimaSplineGSL_one(spline_x->n,spointsx,sdatax,spointsy[nsy-1],&mdl);
             else if(SplineFlag == 0) CubicSplineGSL_one(spline_x->n,spointsx,sdatax,spointsy[nsy-1],&mdl);
             
            
             ji = (int)((newfreq-flow)*data->Tobs);
             sp = fabs((log(priors->lower[ji]))*1.0e-3);
             prange = (log(priors->lower[ji]*100.0) - log(priors->lower[ji]));
             sdatay[nsy-1] = rjdraw(mdl,sp,prange,log(priors->lower[ji]),r);
                
             logqy = rjden(mdl, sdatay[nsy-1], sp, prange);
             logpy = -log(prange);
             logqx = 0.0; logpx = 0.0;
              
             gsl_sort2(spointsy, 1, sdatay, 1, nsy);
             for(j=0;j<nsy;j++)
             {
                 if(spointsy[j] == newfreq) iu = j;
             }
              if(iu < 0 || iu >=nsy) printf("Warning: updated point is out of bounds.\n");
          
           }//end birth move
       }
       else check = 1;
    }
    else if (alpha > 0.3 && spline_x->n >2)//Shift the points laterally
    {
        typ = 7;
        nsy = spline_x->n;
        
        lbl = 1;
        
        //pick a freq to swap, check if active, add 1/T to that freq, sort and find the label
        
        ki=1+(int)(gsl_rng_uniform(r)*(double)(spline_x->n-2));  // pick a point to swap
        
        alpha1 = gsl_rng_uniform(r);
        if(alpha1 > 0.5)  // shift term to right, seperation between spline points should be greater than minimum spline spacing
        {
            newfreq = spointsx[ki] + shiftval;
            if(spointsx[ki+1] < newfreq) check = 1;
        }
        else // shift term to left
        {
            newfreq = spointsx[ki] - shiftval;
            if(newfreq < spointsx[ki-1])  check = 1;
        }         
        if(check == 0)
        {
            spointsy[ki] = newfreq; // swap it with a new value
            gsl_sort(spointsy,1,nsy);
            ji = (int)((newfreq - flow)*data->Tobs);
            for(j=0; j<nsy; j++)
            {
                if(spointsy[j] == newfreq)
                {
                    sdatay[j] = sdatax[j];
                    iu = j;
                }
            }
        }
      } //end lateral shift move
      else  // regular MCMC update
      {
        typ = 4;
        nsy = spline_x->n;
          
          lbl = 2;
          
        //pick a term to update
        ii=(int)(gsl_rng_uniform(r)*(double)(spline_x->n));

        // use a variety of jump sizes by using a sum of gaussians of different width
        e1 = 0.0005;
        alpha = gsl_rng_uniform(r);
        if(alpha > 0.8)
        {
          e1 = 0.002;
        }
        else if(alpha > 0.6)
        {
          e1 = 0.005;
        }
        else if(alpha > 0.4)
        {
          e1 = 0.05;
        }
        iu = ii;
          
        // propose new value for the selected term
        alpha = gsl_rng_uniform(r);
        if(alpha >0.5)   
        {
            sdatay[ii] = sdatax[ii]+gsl_ran_gaussian(r, e1);
            if(ii >0 && ii< spline_x->n -1) spointsy[ii] = spointsx[ii] + gsl_ran_gaussian(r,e1);
        }
        else
        {
            if(ii >0 && ii< spline_x->n -1) spointsy[ii] = flow + (fhigh-flow)*gsl_rng_uniform(r);
            ji = ((spointsy[ii]-flow)*data->Tobs);
            sdatay[ii] = log(priors->lower[ji]) +(log(priors->lower[ji]*100.0) - log(priors->lower[ji]))*gsl_rng_uniform(r);
        }
         
        newfreq = spointsy[ii];
        logpx = 0.0; logpy = 0.0; logqx=0.0; logqy = 0.0;
          
      }
    
        if(check==0)
        {
            //check minimum spacing between two active points
            for(i =1; i<nsy; i++)
            {
                if( (spointsy[i]-spointsy[i-1])<dfmin) check = 1;  
            }
            
          for(i=0; i<nsy; i++)
          {
              ji = (int)((spointsy[i] - flow)*data->Tobs); //Make sure all active points lie within the prior envelope
              if(sdatay[i] > log(priors->lower[ji]*100.0)) {check = 1; }
              if(sdatay[i] < log(priors->lower[ji]))       {check = 1; }
              if(spointsy[i]<flow || spointsy[i]>fhigh)    {check = 1; }
          }
            
        }
        
    }
    else    // update the line model
    {
        sl = 1;
        
      if(tmax == 0) check = 1;
      else
      {
          alpha1 = gsl_rng_uniform(r);
          if(alpha1 > 0.5)  // try a transdimensional move
          {
              lbl = 3;
              
            alpha = gsl_rng_uniform(r);
            if(alpha > 0.5) // add
            {
                typ = 2;
                lines_y->n = lines_x->n+1;
                
                if(lines_y->n <= tmax)
                {
               
                ii = (int)(lines_y->n-1);
                // gets put into Ny-1 since index runs from 0 to < Ny
                alpha = gsl_rng_uniform(r);
                    
                lines_y->f[ii] = qdraw(fprop, pmax, flow, fhigh, ncut, data->Tobs, r);
                lines_y->A[ii] = exp(lAmin+(lAmax-lAmin)*gsl_rng_uniform(r));
                lines_y->nu[ii] = exp(lnumin+(lnumax-lnumin)*gsl_rng_uniform(r));
                k = (int)((lines_y->f[ii]-flow)*data->Tobs);
                newfreql = lines_y->f[ii];

                logqy = log(fprop[k]);
                logpy = -log((double)(ncut)); 
                logqx = 0.0;
                logpx = 0.0;
                    
                if(lines_y->A[ii] > priors->LAmax) check = 1;
                if(lines_y->A[ii] < priors->LAmin) check = 1;
                if(lines_y->nu[ii] > numax) check = 1;
                if(lines_y->nu[ii] < numin) check = 1;
                if(lines_y->f[ii] > fhigh) check = 1;
                if(lines_y->f[ii] < flow) check = 1;
                    
                 newfreql = lines_y->f[ii];
                    
                }
                else
                {
                    check = 1; 
                }

           } // end add line
           else  // remove
           {
               typ = 3;
               lines_y->n = lines_x->n-1;
               
               if(lines_y->n > 0)
               {
                  // pick one to kill
                   k = (int)((double)(lines_x->n)*gsl_rng_uniform(r));
                   ki = k;
                   // reverse move would add line at this location
                   j = (int)((lines_x->f[k]-flow)*data->Tobs);

                   logqy = 0.0;
                   logpy = 0.0;
                   logqx = log(fprop[j]);
                   logpx = -log((double)(ncut)); 
                   
                   j = 0;
                   for(i=0; i< lines_x->n; i++)
                   {
                       if(i != k)
                       {
                       lines_y->f[j] = lines_x->f[i];
                       lines_y->A[j] = lines_x->A[i];
                       lines_y->nu[j] = lines_x->nu[i];
                       j++;
                       }
                   }
        
               }
               else
               {
                   check = 1; 
               }
           }
           
      }
      else  // regular MCMC update
      {
        lines_y->n = lines_x->n;
         
        if(lines_y->n > 0 && lines_y->n <= tmax)
        {
          typ=1;
            
          //pick a term to update
          jj=(int)(gsl_rng_uniform(r)*(double)(lines_x->n));
          //find label of who is geting updated
          ii = jj;

          alpha = gsl_rng_uniform(r);
            
          if(alpha > 0.8)
          {
              typ = 0;
              
              lbl = 4;
              
              // here we try and move an exisiting line to a totally new location
              
              lines_y->f[ii] = qdraw(fprop, pmax, flow, fhigh, ncut, data->Tobs, r);
              logqy = lprop(lines_y->f[ii], fprop, data);
              logpy = lprop(lines_x->f[ii], fprop, data);
              logpx = 0.0; logqx = 0.0;
              
              lines_y->nu[ii] = exp(lnumin+(lnumax-lnumin)*gsl_rng_uniform(r));
              lines_y->A[ii] = exp(lAmin+(lAmax-lAmin)*gsl_rng_uniform(r));
              
              
          }
          else
          {
            typ = 1;
            alpha = gsl_rng_uniform(r);
              
              lbl = 5;

            if     (alpha > 0.9) beta = 1.0e+1;
            else if(alpha > 0.6) beta = 1.0e+0;
            else if(alpha > 0.3) beta = 1.0e-1;
            else                 beta = 1.0e-2;
              


            e2 = beta*s2;
            e3 = beta*s3;
            e4 = beta*s4;
            
            x2 = gsl_ran_gaussian(r, e2);
            x3 = gsl_ran_gaussian(r, e3);
            x4 = gsl_ran_gaussian(r, e4);
       
            lines_y->A[ii] = lines_x->A[ii]*exp(x2);
            lines_y->nu[ii] = lines_x->nu[ii]+x3;
            lines_y->f[ii] = lines_x->f[ii]+x4;
             
            logpx = 0.0; logpy = 0.0;
            logqx = 0.0; logqy = 0.0;
          }
          newfreql = lines_y->f[ii];
         
          if(lines_y->A[ii] > priors->LAmax || lines_y->A[ii] < priors->LAmin)  check = 1;
          if(lines_y->f[ii] < flow || lines_y->f[ii] > fhigh)                   check = 1;
          if(lines_y->nu[ii] < numin || lines_y->nu[ii] > numax)  check = 1;
          }
          else check = 1;
      }

      }
    }
      
    //  printf("label %d type %d check %d\n", lbl, typ, check);

      
    //If line parameters satisfy priors, continue with MCMC update
    if(check == 0)
    {

      if(typ > 3)  // delta updates
      {
        if(nsy>=7)
        {
          if(!bayesline->constantLogLFlag)
          {
              imin = 0; imax = ncut;
              if(SplineFlag == 0)
              {
                //Interpolate in that region {spointsy, sdatay} ==> {freq,xint}
                CubicSplineGSL(imin,imax,nsy,spointsy,sdatay,freq,xint);
              }
              if(SplineFlag == 1)
              {
                 //Akima Spline
                 for(i=0; i<ncut; i++) { Sbase[i] = Sbasex[i]; Sn[i] = Snx[i];}
                 // if control point iu is updated, region between (iu-3) and (iu+3) is impacted. Care needs to be taken at boundary points
                 getrangeakima(iu,nsy,spointsy,data,ncut,&imin,&imax);

                 //Interpolate in that region {spointsy, sdatay} ==> {freq,xint}
                 AkimaSplineGSL(imin,imax,nsy,spointsy,sdatay,freq,xint);
                 
              }  

              for(i=imin;i<imax; i++) Sbase[i] = exp(xint[i]);

              full_spectrum_spline(Sline, data, lines_y);
              for(i=imin; i<imax; i++) 
              {
                  Sn[i] = Sbase[i] + Sline[i];
              }
              logLy = logLx + delta_loglike(power, Sn, Snx, imin, imax);
          }
          else logLy = 1.0; 
        }
        else logLy = -1e60;
      }

      if(typ == 1 || typ == 0)  // fixed dimension MCMC of line ii
      {
        if(!bayesline->constantLogLFlag)
        {
            full_spectrum_single(Sn, Snx, data, lines_x, lines_y, ii, &ilowx, &ihighx, &ilowy, &ihighy);
            logLy = logLx + loglike_single(power, Sn, Snx, ilowx, ihighx, ilowy, ihighy);
        }
        else logLy = 1.0;
      }

      if(typ == 2)  // add new line with label ii
      {
        if(!bayesline->constantLogLFlag)
        {
            full_spectrum_add_or_subtract(Sn, Snx, data, lines_y, ii, &ilowy, &ihighy,  1);
            logLy = logLx + loglike_pm(power, Sn, Snx, ilowy, ihighy);
        }
        else logLy = 1.0;
      }

      if(typ == 3)  // remove line with label ki
      {
        if(!bayesline->constantLogLFlag)
        {
             full_spectrum_add_or_subtract(Sn, Snx, data, lines_x, ki, &ilowx, &ihighx, -1);
             logLy = logLx + loglike_pm(power, Sn, Snx, ilowx, ihighx);
        }
        else logLy = 1.0;
      }

        
      cc[lbl]++;
        
   
      // prior on line number e(-ZETA * n).  (this is a prior, not a proposal)
      // PLEASE KEEP THIS ON (Important for Evidence)
      // effectively an SNR cut on lines
      logpy -= ZETA*(double)(lines_y->n);
      logpx -= ZETA*(double)(lines_x->n);

      //logPsy = logprior(priors->invsigma, priors->mean, Sn, 0, ncut);
      //if(priorFlag)logPsy = logprior(priors->sigma, priors->mean, Sn, 0, ncut);
      if(priorFlag==1)
      {
        logPsy = logprior(priors->lower, priors->upper, Sn, 0, ncut);
        //logPsy = logprior_gaussian_model(priors->mean, priors->sigma, Sn, spointsy, nsy, lines_y->f, lines_y->n, data);
      }
      if(priorFlag==2)
      {
        logPsy = logprior_gaussian(priors->mean, priors->sigma, Sn, 0, ncut);
      }
  
      // if(typ>3) printf("typ %d iu %d check %d log %f %f logp %f %f logq %f %f\n", typ, iu, check, logLx, logLy, logpx, logpy, logqx, logqy);
      logH  = (logLy - logLx)*heat +logpy-logqy-logpx+logqx;
      if(priorFlag!=0) logH += logPsy - logPsx;

      alpha = log(gsl_rng_uniform(r));
     
      if(logH > alpha)
      {
        ac[lbl]++;
        logLx = logLy;
        
        //if(priorFlag!=0)
        logPsx = logPsy;
        lines_x->n = lines_y->n;
        spline_x->n = nsy;
        for(i=0; i< ncut; i++)
        {
          Snx[i] = Sn[i];
          if(typ > 3) Sbasex[i] = Sbase[i];
        }
        for(i=0; i< lines_x->n; i++)
        {
          lines_x->A[i] = lines_y->A[i];
          lines_x->f[i] = lines_y->f[i];
          lines_x->nu[i] = lines_y->nu[i];
        }
        for(i=0; i<spline_x->n; i++)
        {
          sdatax[i] = sdatay[i];
          spointsx[i] = spointsy[i];
        }
      }

    }//end prior check
      
      
      if(mc%5000 ==0)
      {
          printf("%d %d %d %e ", mc, lines_x->n, spline_x->n, logLx );
          for(i=0; i< 6; i++)
          {
              printf("%f ", (double)(ac[i])/(double)(cc[i]));
          }
          printf("\n");
      }
     

  }//End MCMC loop
    
  //Interpolate {spointsx,sdatax} ==> {freq,xint}
  if(SplineFlag == 1) AkimaSplineGSL(0,ncut,spline_x->n,spointsx,sdatax,freq,xint);
  else if(SplineFlag == 0) CubicSplineGSL(0,ncut,spline_x->n,spointsx,sdatax,freq,xint);
  
  
  for(i=0; i< ncut; i++) Sbase[i] = exp(xint[i]);

  full_spectrum_spline(Sline, data, lines_x);
  for(i=0; i< ncut; i++)
  {
    Sn[i] = Sbase[i]+Sline[i];
      
    bayesline->Sbase[i] = Sbase[i];
    bayesline->Sline[i] = Sline[i];
  }

  // return updated spectral model
  for(i=0; i< ncut; i++) Snf[i] = Snx[i];

  // re-map the array to 0..mx ordering
  for(i=0; i< lines_x->n; i++)
  {
    k = i;
    lines_y->f[i] = lines_x->f[k];
    lines_y->A[i] = lines_x->A[k];
    lines_y->nu[i] = lines_x->nu[k];
    lines_y->larray[i] = i;
  }
  
  // return the last value of the chain
  for(i=0; i< lines_x->n; i++)
  {
    lines_x->f[i] = lines_y->f[i];
    lines_x->A[i] = lines_y->A[i];
    lines_x->nu[i] = lines_y->nu[i];
    lines_x->larray[i] = i;
  }
  
  // check for outliers
  pmax = -1.0;
  for(i=0; i< ncut; i++)
  {
    x = power[i]/Snx[i];
    if(x > pmax) pmax = x;
  }
  
  *dan = pmax;

  free(foc);
  free(xint);
  free(Snx);
  free(Sn);
  free(Sbase);
  free(Sbasex);
  free(Sline);
  free(sdatay);
  free(spointsy);
    
  destroy_lorentzianParams(lines_y);
  
}

void BayesLineFree(struct BayesLineParams *bptr)
{
  if(bptr==NULL) return;

  free(bptr->data);

  // storage for line model for a segment. These get updated and passed back from the fitting routine
  destroy_lorentzianParams(bptr->lines_x);

  destroy_splineParams(bptr->spline);
  destroy_splineParams(bptr->spline_x);

  free(bptr->fa);
  free(bptr->Sna);
  free(bptr->freq);
  free(bptr->power);
  free(bptr->Snf);
  free(bptr->Sbase);
  free(bptr->Sline);

  free(bptr->spow);
  free(bptr->sfreq);

  /* Priors are owned by the BayesWave wrapper and are freed there. */

  /* set up GSL random number generator */
  if(bptr->r!=NULL) gsl_rng_free(bptr->r);

  free(bptr);
}

void BayesLineSetup(struct BayesLineParams *bptr, double *freqData, double fmin, double fmax, double deltaT, double Tobs)
{
  int i, j;
  int n, N;
  int imin, imax, nspline;
  double reFreq,imFreq;
  int max_lines=bptr->maxBLLines;

  bptr->data = malloc(sizeof(dataParams));

  bptr->lines_x = malloc(sizeof(lorentzianParams));

  bptr->spline   = malloc(sizeof(splineParams));
  bptr->spline_x = malloc(sizeof(splineParams));

  bptr->TwoDeltaT = deltaT*2.0;
  
  /* set up GSL random number generator */
  const gsl_rng_type *T = gsl_rng_default;
  bptr->r = gsl_rng_alloc (T);
  gsl_rng_env_setup();

  imin = (int)floor(fmin*Tobs);
  imax = (int)floor(fmax*Tobs);
  N = (int)rint(Tobs/deltaT);
  if(imin < 0) imin = 0;
  if(imax > N/2) imax = N/2;
  if(imax <= imin)
  {
    fprintf(stderr,"BayesLineSetup: invalid analysis band [%g, %g) for Tobs=%g and deltaT=%g\n", fmin, fmax, Tobs, deltaT);
    exit(1);
  }
  n = imax-imin;

  bptr->freq  = malloc((size_t)(sizeof(double)*(n)));
  bptr->power = malloc((size_t)(sizeof(double)*(n)));
  bptr->Snf   = malloc((size_t)(sizeof(double)*(n)));
  bptr->Sbase = malloc((size_t)(sizeof(double)*(n)));
  bptr->Sline = malloc((size_t)(sizeof(double)*(n)));

  for(i=0; i<n; i++)
  {
    j = i+imin;
    bptr->freq[i] = (double)(j)/Tobs;

    reFreq = freqData[2*j];
    imFreq = freqData[2*j+1];

    bptr->power[i] = (reFreq*reFreq+imFreq*imFreq);
  }

  // storage for data meta parameters
  create_dataParams(bptr->data,bptr->freq,n,N,Tobs,max_lines);

  // storage for master line model (data->tmax = max_lines)
  create_lorentzianParams(bptr->lines_x,bptr->data->tmax);

  // start with a single line. This number gets updated and passed back
  bptr->lines_x->n = 1;

  bptr->fa  = malloc((size_t)(sizeof(double)*(n)));
  bptr->Sna = malloc((size_t)(sizeof(double)*(n)));

  //starting resolution for the spline
  nspline = (int)((bptr->data->fmax)/bptr->data->fgrid);
  //nspline = (int)(fmax*Tobs);//(int)((bptr->data->fmax-bptr->data->fmin)/bptr->data->fgrid)+2;
    
  create_splineParams(bptr->spline,nspline);

  create_splineParams(bptr->spline_x,nspline);


  // Re-set dataParams structure to use full dataset
  bptr->data->flow  = bptr->data->fmin;
  bptr->data->fhigh = bptr->data->fmax;
  imin  = bptr->data->nmin;
  imax  = imin + n;
  bptr->data->ncut  = imax-imin;
  bptr->data->imin  = imin;
  bptr->data->imax  = imax;

  bptr->spow  = malloc((size_t)(sizeof(double)*(bptr->data->ncut)));
  bptr->sfreq = malloc((size_t)(sizeof(double)*(bptr->data->ncut)));
    
    if(bptr->priors == NULL)
    {
      bptr->priors = malloc(sizeof(BayesLinePriors));
      bptr->priors->upper = malloc(N*sizeof(double));
      bptr->priors->lower = malloc(N*sizeof(double));
      bptr->priors->mean  = malloc(N*sizeof(double));
      bptr->priors->sigma = malloc(N*sizeof(double));
    }
    // Read the windowed-Lorentzian lookup table if it exists, or generate it.
    setup_lorentzian_lookup(bptr->lines_x, N, Tobs, bptr->data->imin);

}

void BayesLineInitialize(struct BayesLineParams *bayesline)
{

  int i;
  int imin,imax;
  int j;


  int jj = 0;

  /******************************************************************************/
  /*                                                                            */
  /*  Rapid non-Markovian fit over small bandwidth blocks of data               */
  /*                                                                            */
  /******************************************************************************/

  if(bayesline->lines_x->n < 1 ) bayesline->lines_x->n = 1;

  // Re-set dataParams structure to use full dataset
  bayesline->data->flow  = bayesline->data->fmin;
  imin  = bayesline->data->imin;
  imax  = bayesline->data->imax;
  bayesline->data->ncut  = imax-imin;

  /******************************************************************************/
  /*                                                                            */
  /*  Full-data spline fit (holding lines fixed from search phase)              */
  /*                                                                            */
  /******************************************************************************/

  int k;
  double mdn;
  double x,z;

  /* Make local pointers to BayesLineParams structure members */
  dataParams *data             = bayesline->data;
  splineParams *spline         = bayesline->spline;
  double *Sna                  = bayesline->Sna;
  double *fa                   = bayesline->fa;


  //initialize spline grid & PSD values
  k=0;
  for(j=0; j<bayesline->spline->n; j++)
  {
    x = data->fmin+(data->fhigh-data->fmin)*(double)(j)/(double)(bayesline->spline->n-1);

    if     (x <= fa[0])    z = Sna[0];
    else if(x >= fa[jj-1]) z = Sna[jj-1];
    else
    {
      i=k-10;
      mdn = 0.0;
      do
      {
        if(i>=0) mdn = fa[i];
        i++;
      } while(x > mdn);

      k = i;
      z = Sna[i];
    }

    spline->points[j] = x;
    spline->data[j]   = z;
  }

  for(j=0; j<bayesline->spline->n; j++)
  {
    bayesline->spline_x->points[j] = bayesline->spline->points[j];
    bayesline->spline_x->data[j]   = bayesline->spline->data[j];
  }

  bayesline->data->flow  = bayesline->data->fmin;

  imin  = bayesline->data->imin;

  for(i=0; i< bayesline->data->ncut; i++)
  {
    j = i + imin - bayesline->data->nmin;
    bayesline->spow[i] = bayesline->power[j];
    bayesline->sfreq[i] = bayesline->freq[j];
  }

}

void BayesLineRJMCMC(struct BayesLineParams *bayesline, double *freqData, double *psd, double *invpsd, double *splinePSD, int N, int cycle, double beta, int priorFlag, double *fprop, int SplineFlag)
{
  int i,j;
  int imin,imax;
  double reFreq,imFreq;
  double dan;

  //at frequencies below fmin output SAmax (killing lower frequencies in any future inner products)
  imax = bayesline->data->imax;
  imin = bayesline->data->imin;

  for(i=0; i< bayesline->data->ncut; i++)
  {
    j = i + imin;

    reFreq = freqData[2*j];
    imFreq = freqData[2*j+1];

    bayesline->spow[i] = (reFreq*reFreq+imFreq*imFreq);
  }
  
  BayesLineLorentzSplineMCMC(bayesline, beta, cycle, priorFlag, &dan, fprop, SplineFlag);

  /******************************************************************************/
  /*                                                                            */
  /*  Output PSD in format for BayesWave                                        */
  /*                                                                            */
  /******************************************************************************/

  for(i=0; i<N/2; i++)
  {
    if(i>=imin && i<imax)
    {
      psd[i] = bayesline->Snf[i-imin];
      splinePSD[i] = bayesline->Sbase[i-imin];
    }
    else
    {
      psd[i] = 1.0;
      splinePSD[i] = 1.0;
    }
    invpsd[i] = 1./psd[i];
  }
    

    
}

double rjdraw(double model, double sp, double prange, double pmin, gsl_rng *r)
{
    double x;
    double alpha;
        
    alpha = gsl_rng_uniform(r);
        
    if(alpha < ufrac)  // uniform draw
    {
        x = pmin + prange *gsl_rng_uniform(r);
    }
    else
    {
        x = model + sp*gsl_ran_gaussian(r,1.0);
    }
    return x;
}

double rjden(double model, double ref, double sp, double prange)
{
    double x, u;

    u = (ref-model)/sp;
    x = log( ufrac/prange + (1.0-ufrac)/(sp*sqrt(TPI))*exp(-u*u/2.0));
    return x;
}

void copy_bayesline_params(struct BayesLineParams *origin, struct BayesLineParams *copy)
{
  int i;
  int imax;
  int imin;

  copy->maxBLLines = origin->maxBLLines;
  copy->constantLogLFlag = origin->constantLogLFlag;
  copy->flatPriorFlag = origin->flatPriorFlag;
  copy->TwoDeltaT = origin->TwoDeltaT;
  copy->data->tmax = origin->data->tmax;

  for(i=0; i<origin->data->ncut; i++)
  {
    copy->spow[i]  = origin->spow[i];
    copy->sfreq[i] = origin->sfreq[i];
  }

  imax = origin->data->imax;
  imin = origin->data->imin;
  for(i=0; i<imax-imin; i++)
  {
    copy->Snf[i]   = origin->Snf[i];
    copy->spow[i]  = origin->spow[i];
    copy->sfreq[i] = origin->sfreq[i];
    copy->Sbase[i] = origin->Sbase[i];
    copy->Sline[i] = origin->Sline[i];
  }

  copy_splineParams(origin->spline, copy->spline);
  copy_splineParams(origin->spline_x, copy->spline_x);
  copy_lorentzianParams(origin->lines_x, copy->lines_x);
}

#include <gsl/gsl_statistics.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>

#define Qs 8.0          // Q used in the glitch cleaning for spectral estimation
#define fsp 4.0         // spacing of spline points in Hz
#define TPI 6.2831853071795862319959269370884
#define RTPI 1.772453850905516                    // sqrt(Pi)
#define sthresh 9.0
#define warm 5.0
#define LN2 0.6931471805599453                 // ln 2


#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif


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
static void phase_blind_time_shift(double *corr, double *corrf, double *data1, double *data2, int n)
{
  int nb2, i, l, k;
  
  nb2 = n / 2;
  
  corr[0] = 0.0;
  corrf[0] = 0.0;
  corr[nb2] = 0.0;
  corrf[nb2] = 0.0;
  
  for (i=1; i < nb2; i++)
  {
    l=i;
    k=n-i;
    
    corr[l]  = (data1[l]*data2[l] + data1[k]*data2[k]);
    corr[k]  = (data1[k]*data2[l] - data1[l]*data2[k]);
    corrf[l] = corr[k];
    corrf[k] = -corr[l];
  }
  
  gsl_fft_halfcomplex_radix2_inverse(corr, 1, n);
  gsl_fft_halfcomplex_radix2_inverse(corrf, 1, n);
  
  
}

static void SineGaussianC(double *hs, double *sigpar, double Tobs, int NMAX)
{
  double f0, Q, sf;
  double fmax, fmin, fac;
  double f;
  double tau, dt;
  int imin, imax;
  
  int i;
  
  // Torrence and Compo
  
  f0 = sigpar[1];
  Q = sigpar[2];
  
  tau = Q/(TPI*f0);
  
  dt = Tobs/(double)(NMAX);
  
  fmax = f0 + 3.0/tau;  // no point evaluating waveform past this time (many efolds down)
  fmin = f0 - 3.0/tau;  // no point evaluating waveform before this time (many efolds down)
  
  fac = sqrt(sqrt(2.0)*M_PI*tau/dt);
  
  i = (int)(f0*Tobs);
  imin = (int)(fmin*Tobs);
  imax = (int)(fmax*Tobs);
  if(imax - imin < 10)
  {
    imin = i-5;
    imax = i+5;
  }
  
  if(imin < 0) imin = 0;
  if(imax > NMAX/2) imax = NMAX/2;
  
  hs[0] = 0.0;
  hs[NMAX/2] = 0.0;
  
  for(i = 1; i < NMAX/2; i++)
  {
    hs[i] = 0.0;
    hs[NMAX-i] = 0.0;
    
    if(i > imin && i < imax)
    {
      f = (double)(i)/Tobs;
      sf = fac*exp(-M_PI*M_PI*tau*tau*(f-f0)*(f-f0));
      hs[i] = sf;
      hs[NMAX-i] = 0.0;
    }
    
  }
  
  
}

static double f_nwip(double *a, double *b, int n)
{
  int i, j, k;
  double arg, product;
  double ReA, ReB, ImA, ImB;
  
  arg = 0.0;
  for(i=1; i<n/2; i++)
  {
    j = i;
    k = n-1;
    ReA = a[j]; ImA = a[k];
    ReB = b[j]; ImB = b[k];
    product = ReA*ReB + ImA*ImB;
    arg += product;
  }
  
  return(arg);
  
}

static void TransformC(double *a, double *freqs, double **tf, double **tfR, double **tfI, double Q, double Tobs, int n, int m)
{
    int i, j;
    double fix;
    double *AC, *AF;
    double *b;
    double *params;
    double bmag;
    
    // [0] t0 [1] f0 [2] Q [3] Amp [4] phi
    
    fix = sqrt((double)(n/2));
    
    params= double_vector(5);
    AC=double_vector(n);  AF=double_vector(n);
    b = double_vector(n);
    
    params[0] = 0.0;
    params[2] = Q;
    params[3] = 1.0;
    params[4] = 0.0;
    
    for(j = 0; j < m; j++)
    {
        params[1] = freqs[j];
        
        SineGaussianC(b, params, Tobs, n);
        
        bmag = sqrt(f_nwip(b, b, n)/(double)n);
        
        bmag /= fix;
        
        phase_blind_time_shift(AC, AF, a, b, n);
        
        for(i = 0; i < n; i++)
        {
            tfR[j][i] = AC[i]/bmag;
            tfI[j][i] = AF[i]/bmag;
            tf[j][i] = tfR[j][i]*tfR[j][i]+tfI[j][i]*tfI[j][i];
        }
    
    }
    
    free_double_vector(AC);  free_double_vector(AF);
    free_double_vector(b);  free_double_vector(params);
    
}


static double Getscale(double *freqs, double Q, double Tobs, double fmax, int n, int m)
{
  double *data, *intime, *ref, **tfR, **tfI, **tf;
  double f, t0, delt, t, x, dt;
  double scale, sqf;
  int i, j;
  
  data   = double_vector(n-1);
  ref    = double_vector(n-1);
  intime = double_vector(n-1);
  
  tf  = double_matrix(m-1,n-1);
  tfR = double_matrix(m-1,n-1);
  tfI = double_matrix(m-1,n-1);
  
  f = fmax/4.0;
  t0 = Tobs/2.0;
  delt = Tobs/8.0;
  dt = Tobs/(double)(n);
  
  //out = fopen("packet.dat","w");
  for(i=0; i< n; i++)
  {
    t = (double)(i)*dt;
    x = (t-t0)/delt;
    x = x*x/2.0;
    data[i] = cos(TPI*t*f)*exp(-x);
    ref[i] = data[i];
    //fprintf(out,"%e %e\n", t, data[i]);
  }
  // fclose(out);
  
  
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
  // out = fopen("testtime.dat","w");
  for(i=0; i< n; i++)
  {
    // fprintf(out,"%e %e %e\n",times[i], intime[i], ref[i]);
    
    if(fabs(ref[i]) > 0.01)
    {
      j++;
      x += intime[i]/ref[i];
    }
  }
  //fclose(out);
  
  x /= sqrt((double)(2*n));
  
  scale = (double)j/x;
  
  // printf("scaling = %e %e\n", x/(double)j, (double)j/x);
  
  free_double_vector(data);
  free_double_vector(ref);
  free_double_vector(intime);
  free_double_matrix(tf, m-1);
  free_double_matrix(tfR,m-1);
  free_double_matrix(tfI,m-1);
  
  return scale;
  
}
static void whiten(double *data, double *Sn, int N)
{
  double x;
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
static void medspecspline(double *data, double *S, double *SN, double *SM, double df, int N)
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

static void clean(double *D, double *Draw, double *sqf, double *freqs, double *Sn, double *specD, double *sspecD, double df, double Q, double Tobs, double scale, int Nf, int N, int imin, int imax, double *SNR)
{
  
  int i, j, k;
  int flag;
  int ii, jj;
  double x;
  double S;
  double fwindow;

  // allocate some arrays
  double **tfDR, **tfDI;
  double **tfD;
  double **live;
  double **live2;
  
  double *Dtemp;
    
  FILE *out;
  
    live= double_matrix(Nf,N);
    live2= double_matrix(Nf,N);
    tfDR = double_matrix(Nf,N);
    tfDI = double_matrix(Nf,N);
    tfD = double_matrix(Nf,N);
  
  Dtemp = (double*)malloc(sizeof(double)*(N));
  
  for (i = 0; i < N; i++) Dtemp[i] = Draw[i];
  
  // D holds the previously cleaned data
  // Draw holds the raw data
  
  // D is used to compute the spectrum. A copy of Draw is then whitened using this spectrum and glitches are then identified
  // The glitches are then re-colored, subtracted from Draw to become the new D
  
  // FFT
  gsl_fft_real_radix2_transform(D, 1, N);
  gsl_fft_real_radix2_transform(Dtemp, 1, N);
  
  // Form spectral model for whitening data (lines plus a smooth component)
   medspecspline(D, Sn, specD, sspecD, df, N);
    
  // whiten data
  whiten(Dtemp, specD, N);
  
  // Wavelet transform
  TransformC(Dtemp, freqs, tfD, tfDR, tfDI, Q, Tobs, N, Nf);
    
    
   /*
    out = fopen("wcheck.dat","w");
    for(i = 1; i < N/2; i++) fprintf(out,"%e %e\n", (double)(i)/Tobs, Dtemp[i]*Dtemp[i]+Dtemp[N-i]*Dtemp[N-i]);
    fclose(out);
    out = fopen("pcheck.dat","w");
    for(i = 1; i < N/2; i++) fprintf(out,"%e %e %e\n", (double)(i)/Tobs, Sn[i], specD[i]);
    fclose(out);
    
        double f, t, dt;
        dt = Tobs/(double)(N);
        out = fopen("Qtransform.dat","w");
        for(j = 0; j < Nf; j++)
        {
            f = freqs[j];
            
            for(i = 0; i < N; i++)
            {
                t = (double)(i)*dt;
                if(t > 1.0 && t < Tobs-1.0) fprintf(out,"%e %e %e\n", t-Tobs/2.0, f, tfD[j][i]);
            }
            
            fprintf(out,"\n");
        }
        fclose(out);
    */
     
  
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
  
  
  /*out = fopen("wglitch.dat", "w");
   for(i=0; i< N; i++)
   {
   fprintf(out,"%d %e\n", i, Dtemp[i]);
   }
   fclose(out);*/
  
  
  // Compute the excess power (relative to the current spectral model
  S = 0.0;
  for (i = imin; i < imax; ++i) S += Dtemp[i]*Dtemp[i];
  S = sqrt(S);
  
  printf("   Excess SNR = %f\n", S);
  
  *SNR = S;
  
  
  //Unwhiten and subtract the excess power so we can compute a better spectral estimate
  // Back to frequency domain
  
  gsl_fft_real_radix2_transform(Dtemp, 1, N);
  
  
  // only use smooth spectrum in the un-whitening
  Dtemp[0] = 0.0;
  for(i=1; i< N/2; i++)
  {
    x = sqrt(sspecD[i]);
    Dtemp[i] *= x;
    Dtemp[N-i] *= x;
  }
  Dtemp[N/2] = 0.0;
  
  gsl_fft_halfcomplex_radix2_inverse(Dtemp, 1, N);
  
  
  x = sqrt((double)(2*N));
  
  for(i=0; i< N; i++)
  {
    D[i] = Draw[i]-Dtemp[i]/x;
  }
  
    /*
   out = fopen("glitch.dat", "w");
   for(i=0; i< N; i++)
   {
   fprintf(out,"%d %e\n", i, Dtemp[i]/x);
   }
   fclose(out);
  */
  
  free(Dtemp);
  free_double_matrix(live, Nf);
  free_double_matrix(live2,Nf);
  free_double_matrix(tfDR, Nf);
  free_double_matrix(tfDI, Nf);
  free_double_matrix(tfD,  Nf);
  
  
  return;
   
}
void gakima(int k, int Nknot, double *ffit, double Tobs, int *imin, int *imax)
{
    // The original Akima solines used a 5 point stencil. The GSL Akima splines use
    // a 7 point stencil (maybe to make the fit C^2 versus C^1?)
    
    int flag;
    
    flag = 0;
     
    
              if(k > 3)
              {
                  *imin = (int)(ffit[k-3]*Tobs);
              }
              else
              {
                  *imin = (int)(ffit[0]*Tobs);
                  flag = 1;
              }
              
              if(k < Nknot-4)
              {
                  *imax = (int)(ffit[k+3]*Tobs);
              }
              else
              {
                  *imax = (int)(ffit[Nknot-1]*Tobs);
                  flag = 2;
              }
    
    if(flag == 1) *imax = (int)(ffit[6]*Tobs);
    if(flag == 2) *imin = (int)(ffit[Nknot-7]*Tobs);
    
}


static int trim_spline(double *splinef, double *splineA, int Nknot, int trials, double *freqs, double *lSM, double *lSMx, int istart, int iend, double Tobs, gsl_rng *r)
{
    int i, j, k, kx, q, ii, NKmax, NK;
    int ist, ied;
    int flag;
    double x, y, z;
    double logLx, logLy, dLsq;
    double *sA, *sf, *lSMy;
    double fkeep;
    int *list;
    
    double f1;
    
    fkeep = 16.0; // keep spline points below this frequency
    kx = 0;
    do
    {
        kx++;
    }while(splinef[kx] < fkeep);
    
    dLsq = 0.2*0.2;  // aim for roughly 0.2 tolerance in log S.
    
    gsl_spline   *aspline;
    gsl_interp_accel *acc;
    
    // maximum value for iend is N/2-1
    
    lSMy = double_vector(iend+1);
    list = int_vector(iend+1); // list of active points
    
    for (i = istart; i <= iend; ++i) list[i] = 1;
    
    sf = double_vector(Nknot);
    sA = double_vector(Nknot);
    
    NK = Nknot;
    
         ii = 0;
           do
           {
               ii++;
      
             if(NK > 10)
             {
            
                // pick a knot to update
                do
                {
                    // we don't mess with the first kx knots or the last knot (Nknot-1)
                    j  = kx+(int)((double)(Nknot-kx-1)*gsl_rng_uniform(r));
                    q = (int)(splinef[j]*Tobs);
                    k = list[q];
                    //printf("%d %d\n", j, k);
                }while(k==0);
                 
                k = 0;
                for (i = 0; i < Nknot; ++i)
                {
                    if(list[(int)(splinef[i]*Tobs)] == 1) // make sure the point is active
                    {
                      if(i != j) // skip the point i = j
                      {
                        sA[k] = splineA[i];
                        sf[k] = splinef[i];
                        k++;
                      }
                    }
                 }
                 
                //printf("%d %d\n", k, NK);
                 
                aspline = gsl_spline_alloc(gsl_interp_akima, k);
                acc = gsl_interp_accel_alloc();
                
                gsl_spline_init(aspline,sf,sA,k);
                
                // this is the range spanned by the original spline points surrounding the deleted point
                gakima(j, Nknot, splinef, Tobs, &ist, &ied);
                 
                // printf("%d %d %d %d\n", ist, ied, istart, iend);

                
                for (i = ist; i <= ied; ++i) lSMy[i] = gsl_spline_eval(aspline,freqs[i],acc);
                 
                 gsl_spline_free (aspline);
                 gsl_interp_accel_free (acc);
                
                z = 0.0;
                logLx = 0.0;
                logLy = 0.0;
                for (i = ist; i <= ied; ++i)
                {
                    x = lSM[i]-lSMx[i];
                    x = x*x/dLsq;
                    logLx += x*x;
                    x = lSM[i]-lSMy[i];
                    if(fabs(x) > z) z = fabs(x);
                    x = x*x/dLsq;
                    logLy += x*x;
                }
            
                 
                x = (logLx-logLy);
                 
                
                if(x > -1.0 && z < 0.5)
                {
                    list[q] = 0; // point to be removed
                    NK = k;
                }
                 
               // printf("%d %d %e %e %e %e\n", ii, NK, x, z, logLx, logLy);
                
            }
            
           }while(ii < trials && NK > 50);
    
      k = 0;
      for (i = 0; i < Nknot; ++i)
      {
        if(list[(int)(splinef[i]*Tobs)] == 1) // make sure the point is active
        {
            sA[k] = splineA[i];
            sf[k] = splinef[i];
            k++;
         }
       }
    
      printf("check %d %d\n", k, NK);

    
        for (i = 0; i < NK; ++i)
        {
            splineA[i] = sA[i];
            splinef[i] = sf[i];
        }
        
        free_double_vector(sf);
        free_double_vector(sA);
        free_double_vector(lSMy);
        free(list);
    
    return NK;
        
}


static int splinestart(int N, int istart, int iend, double fstep, double *SM, double *freqs, double Tobs, double *splineA, double *splinef, gsl_rng *r)
{
    int i, j, k, ii, jj, kk, kkold, n, Nx, Np;
    int inc;
    double *lSM, *Sfit;
    double *scale;
    double y2, cnt, xtol;
    double chisqquad, chisqlin, chisqconst;
    double x, z;
   
    FILE *out;
    
    lSM = double_vector(N/2);
    Sfit = double_vector(N/2);
    
    // maximum value for iend is N/2-1
    
    for(i=istart; i< N/2; i++) lSM[i] = log(SM[i]);
    
    ii = (int)(Tobs*fstep);
    Np = iend-istart;
    kk = Np/ii;
    if((Np-kk*ii) != 0) kk++;
    
    //printf("%d\n", kk);
    
    for(i=0; i< kk; i++)
    {
        k = istart+i*ii;
        splinef[i] = freqs[k];
        splineA[i] = lSM[k];
    }
    splinef[kk] = freqs[iend];
    splineA[kk] = lSM[iend];
    
    Nx = kk+1;
    
    // smooth spectrum
    // Allocate spline
    gsl_spline   *aspline;
    gsl_interp_accel *acc;
    
    aspline = gsl_spline_alloc(gsl_interp_akima, Nx);
    acc = gsl_interp_accel_alloc();
    gsl_spline_init(aspline,splinef,splineA,Nx);
    Sfit[istart] = splineA[0];
    Sfit[iend] = splineA[Nx-1];
    for (i = istart+1; i < iend; ++i)
    {
        Sfit[i] = gsl_spline_eval(aspline,freqs[i],acc);
    }
    
    gsl_spline_free (aspline);
    gsl_interp_accel_free (acc);
    
    Nx = trim_spline(splinef, splineA, Nx, 2*Nx, freqs, lSM, Sfit, istart, iend, Tobs, r);
    
    aspline = gsl_spline_alloc(gsl_interp_akima, Nx);
    acc = gsl_interp_accel_alloc();
    gsl_spline_init(aspline,splinef,splineA,Nx);
    
    Sfit[istart] = splineA[0];
    Sfit[iend] = splineA[Nx-1];
    for (i = istart+1; i < iend; ++i)
    {
        Sfit[i] = gsl_spline_eval(aspline,freqs[i],acc);
    }
    
    gsl_spline_free (aspline);
    gsl_interp_accel_free (acc);
    
    free_double_vector(Sfit);
    free_double_vector(lSM);
    
    return Nx;
   
    
}

double cut_lorentz(lorentzianParams *restrict ll, int N, double *SM, double *SL, double *PG, double *lf, double *lnu, double *lS, double Tobs)
{
    int q, imin, imax, i, j, n, kx, flag;
    int k, ii, kk, jj;
    int NL;
    double *SLy, *SLx;
    double dlogL, x, y, Lx;
    double fx, nux, Sx;
    double fmax, numax, Smax, Lmax;
    double loglS, lognu;
    double logLx, logLy;
    
    SLx = double_vector(N/2);
    SLy = double_vector(N/2);

            k = llook(ll, lf[0], lnu[0], lS[0], Tobs);
            
            imin = k-ll->wdth/2;
            imax = k+ll->wdth/2;
            if(imin < 1) imin = 1;
            if(imax > N/2) imax = N/2;
    
            for (i = 0; i < N/2; ++i) SLx[i] = 0.0;
            for (i = 0; i < N/2; ++i) SLy[i] = 0.0;
           
            // update the line model
            for (i = 0; i < ll->wdth; ++i)
            {
                kk = k - ll->wdth/2+i;
                if(kk > 0 && kk < N/2) SLx[kk] = SL[kk]-ll->lt[i];
            }
            
                // likelihood without the line
                logLx = 0.0;
                for (i = imin; i < imax; ++i)
                {
                    if((SM[i]+SLx[i]) > 0.0) logLx += -(log(SM[i]+SLx[i]) + PG[i]/(SM[i]+SLx[i]));
                }
    
             // now try adding back in the line with some maximization
              lognu = log(lnu[0]);
              loglS = log(lS[0]);
              Lmax = -1.0e60;
              for (ii = -5; ii <= 5; ++ii)
              {
                  fx = lf[0] + 0.01*(double)(ii)/Tobs;
                  for (jj = -4; jj <= 4; ++jj)
                  {
                      nux = exp(lognu+(double)(jj)/8.0);
                      
                      if(nux >= ll->numin && nux <= ll->numax)
                      {
                          k = llook(ll, fx, nux, lS[0], Tobs);
                          for (kk = -4; kk <= 4; ++kk)
                          {
                              Sx = exp(loglS+(double)(kk)/8.0);
                              x = Sx/lS[0];
                              
                              for (i = 0; i < ll->wdth; ++i)
                              {
                                  n = k - ll->wdth/2+i;
                                  if(n > 0 && n < N/2) SLy[n] = SLx[n]+x*ll->lt[i];
                              }
                              logLy = 0.0;
                              for (i = imin; i < imax; ++i)
                              {
                                  // very rarely we get an invalid PSD need to investigate more
                                  if((SM[i]+SLy[i]) > 0.0) logLy += -(log(SM[i]+SLy[i]) + PG[i]/(SM[i]+SLy[i]));
                              }
                              if(logLy > Lmax)
                              {
                                  Lmax = logLy;
                                  fmax = fx;
                                  numax = nux;
                                  Smax = Sx;
                              }
                          }
                      }
                    }
                  
              }
    
             dlogL = Lmax-logLx;
            
        if(dlogL < 10.0)
        {
            // remove
            for (i = imin; i < imax; ++i) SL[i] = SLx[i];
        }
        else
        {
            // keep updated line
            k = llook(ll, fmax, numax, Smax, Tobs);
            for (i = 0; i < ll->wdth; ++i)
            {
                n = k - ll->wdth/2+i;
                if(n > 0 && n < N/2) SL[n] = SLx[n]+ll->lt[i];
            }
            lf[1] = fmax;
            lnu[1] = numax;
            lS[1] = Smax;
        }
         
        
    free(SLx);
    free(SLy);
   
    return dlogL;
    
}

static void lineget(int N, int istart, int iend, double *SM, double *PS, double *freqs, double *lf, int *Nlns)
{
    int Nlines, flag, is, ie, i, j, k, ii;
    double x, xm, xp, max, spread;
    double *lh;
    FILE *out;
    
    lh = double_vector(N/2);
    
    // make sure that we have points either side
    is = istart;
    if(is < 3) is = 3;
    ie = iend;
    if(ie > N/2-3) ie = N/2-3;
    
  // count the number of lines
  j = 0;
  for (i = is; i < ie; ++i)
  {
      
      x = PS[i]/SM[i];
      
      if(x > linemul)
      {
          if(PS[i] > PS[i-1] && PS[i] > PS[i+1]) // local peak
          {
                  lf[j] = freqs[i];
                  lh[j] = PS[i];
                  j++;
          }
          
      }
      
  }
    
    Nlines = j;
  
    printf("There are %d line candidates\n", Nlines);
    
    *Nlns = Nlines;
    
    free(lh);

}


static void lorentzfit(lorentzianParams *restrict ll, double *dprime, double *freqs, double Tobs, int N, int istart, int iend, int max_lines, int *Nln)
{
    double df, thresh, lgLx;
    double f, nu, Q, logL, nux, fx, f0, A;
    double Ax, scale;
    double *SM, *PG, *SN;
    double *SL, *PR;
    double *dhold;
    
    double x, y, z;
    int Nlines, Nltotal;
    int i, j, k, m, ii, jj, kk;
    double lnumin, lnumax;
    
    double *fa, *na, *Aa;
    
    double *fhold, *nuhold, *Ahold;
    
    double *linef, *lgL;
    double *lf, *lnu, *lA;
    double fwin;
    double hr, hi;
    double Lthresh, lnu1, lnu2, lnm;
    int w1;
    double wide;
    
    const gsl_rng_type * T;
    gsl_rng * r;
    
    FILE *out;
    
    gsl_rng_env_setup();
    
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    
    fa = double_vector(2);
    na = double_vector(2);
    Aa = double_vector(2);
    
    dhold = double_vector(N);
    SL = double_vector(N/2);
    SN = double_vector(N/2);
    SM = double_vector(N/2);
    PG = double_vector(N/2);
    PR = double_vector(N/2);
    
    linef = double_vector(max_lines);
    lgL = double_vector(max_lines);
    lf = double_vector(max_lines);
    lnu = double_vector(max_lines);
    lA = double_vector(max_lines);
    
    df = 1.0/Tobs;
    
    lnumin = ll->lnumin;
    lnumax = ll->lnumax;
    
    // restrict line widths used in the initial fit to avoid hiding other lines
    lnu1 = log(2.0e-4);
    if(lnu1 > lnumax) lnu1 = lnumax;
    lnu2 = log(2.0e-3);
    if(lnu2 > lnumax) lnu2 = lnumax;
    
    for (i = 1; i < N/2; ++i) SL[i] = 0.0;
    for (i = 1; i < N; ++i) dhold[i] = dprime[i];
    for (i = 1; i < N/2; ++i) PR[i] = 2.0*(dprime[i]*dprime[i]+dprime[N-i]*dprime[N-i]);
    
    Nltotal = 0;
    
    for (m = 0; m < 4; ++m)
    {
        
        if(m==0)
        {
            thresh = 100.0;
            Lthresh = 10.0;
            lnm = lnu1;
        }
        
        if(m==1)
        {
            thresh = 30.0;
            Lthresh = 10.0;
            lnm = lnu2;
        }
        
        if(m > 1)
        {
            thresh = 9.0;
            Lthresh = 10.0;
            lnm = lnumax;
        }
        
        medspecspline(dprime, PG, SN, SM, df, N);

        lineget(N, istart, iend, SM, PG, freqs, linef, &Nlines);
        
        jj = 0;
        for (kk = 0; kk < Nlines; ++kk)
        {
            
            lgLx = -1.0e40;
            for (i = -100; i <= 100; ++i)
            {
                f = linef[kk]+(double)(i)/(200.0*Tobs);
                for (j = 0; j <= 20; ++j)
                {
                    nu = exp(lnumin + (lnm-lnumin)*(double)(j)/20.0);
                    logL = Lpeak(ll, PG, SM, N, 2, f, nu, &A, Tobs);
                    if(logL > lgLx)
                    {
                        lgLx = logL;
                        fx = f;
                        nux = nu;
                        Ax = A;
                    }
                }
            }
            
            //printf("%d %e %e %e %e\n", kk, fx, nux, Ax, lgLx);
        
            if(lgLx > Lthresh)
            {
                lf[Nltotal+jj] = fx;
                lnu[Nltotal+jj] = nux;
                lA[Nltotal+jj] = Ax;
                lgL[Nltotal+jj] = lgLx;
                jj++;
             }
        }
        
        for (kk = 0; kk < jj; ++kk)
        {
            k = llook(ll, lf[Nltotal+kk], lnu[Nltotal+kk], lA[Nltotal+kk], Tobs);
            for (i = 0; i < ll->wdth; ++i)
            {
                j = k-ll->wdth/2+i;
                if(j > 0 && j < N/2) SL[j] += ll->lt[i];
            }
        }
        
        Nltotal += jj;
        
        // remove weak lines
        
        // sorting doesn't have much effect
        
        gsl_vector *v = gsl_vector_alloc (Nltotal);
        gsl_permutation *p = gsl_permutation_alloc (Nltotal);
        for (i = 0; i < Nltotal; ++i) gsl_vector_set (v, i, lgL[i]);
        gsl_sort_vector_index (p, v);
        
        fhold = double_vector(Nltotal);
        nuhold = double_vector(Nltotal);
        Ahold = double_vector(Nltotal);
        j = 0;
        for (i = Nltotal-1; i >= 0; --i)
        {
            fhold[j] = lf[p->data[i]];
            nuhold[j] = lnu[p->data[i]];
            Ahold[j] = lA[p->data[i]];
            //printf("%d %e\n", i, lgL[p->data[i]]);
            j++;
        }
        
        gsl_vector_free (v);
        gsl_permutation_free (p);
        
        ii = 0;
        for (kk = 0; kk < Nltotal; ++kk)
        {
            fa[0] = fhold[kk];
            na[0] = nuhold[kk];
            Aa[0] = Ahold[kk];
            x = cut_lorentz(ll, N, SM, SL, PR, fa, na, Aa, Tobs);
            if(x > 8.0) // keep
            {
                lf[ii] = fa[1];
                lnu[ii] = na[1];
                lA[ii] = Aa[1];
                ii++;
            }
        }
        
        Nltotal = ii;
        
        free(fhold);
        free(nuhold);
        free(Ahold);
        
        for (i = 1; i < N/2; ++i)
        {
            x = SL[i]/(SM[i]+SL[i]);
            hr = 0.0;
            hi = 0.0;
            if(x > 1.0e-2)
             {
                y = x*dhold[i];
                z = x*dhold[N-i];
                lmcmc(1000, SM[i], SL[i], dhold[i], dhold[N-i], y, z, &hr, &hi, r);
              }
            dprime[i] = dhold[i]-hr;
            dprime[N-i] = dhold[N-i]-hi;
        }
        
    }
    
    medspecspline(dprime, PG, SN, SM, df, N);
    
    // form up initial line model
    for (i = 1; i < N/2; ++i) SL[i] = 0.0;
    for (j = 0; j < Nltotal; ++j)
    {
        k  = llook(ll, lf[j], lnu[j], lA[j], Tobs);
        for (i = 0; i < ll->wdth; ++i)
        {
            jj = k-ll->wdth/2+i;
            if(jj > 0 && jj < N/2) SL[jj] += ll->lt[i];
        }
    }
    
    // copy the initial line model into the line structure
    ll->n = Nltotal;
    for (i = 0; i < Nltotal; ++i)
    {
        ll->f[i] = lf[i];
        ll->A[i] = lA[i];
        ll->nu[i] = lnu[i];
    }
    
    free(SL);
    free(PR);
    free(SM);
    free(SN);
    free(PG);
    
    free(Aa);
    free(na);
    free(fa);
    
    free(linef);
    free(lgL);
    free(lf);
    free(lnu);
    free(lA);
    
    *Nln = Nltotal;
    
}

void lmcmc(int M, double SM, double SL, double dr, double di, double xr, double xi, double *hr, double *hi, gsl_rng *r)
{
    int mc, cnt, ac;
    double hrx, hry, hix, hiy;
    double logLx, logLy;
    double logpx, logpy;
    double x, y, sigma, H;
    double alpha;
    
    hrx = xr;
    hix = xi;
    
    x = 0.25*SM*SL/(SM+SL);
    sigma = sqrt(x);
    
    x = dr-hrx;
    y = di-hix;
    logLx = -2.0*(x*x+y*y)/SM;
    
    cnt = 0;
    ac = 0;
    for (mc = 1; mc < M; ++mc)
    {
        hry = hrx + gsl_ran_gaussian(r,sigma);
        hiy = hix + gsl_ran_gaussian(r,sigma);
        x = dr-hry;
        y = di-hiy;
        logLy = -2.0*(x*x+y*y)/SM;
        
        H = logLy-logLx;
        
        alpha = log(gsl_rng_uniform(r));
        
        cnt++;

        if(H > alpha)
        {
            logLx = logLy;
            hrx = hry;
            hix = hiy;
            ac++;
        }
        
       // if(mc%80==0) printf("%d %e %e\n", mc, logLx, (double)(ac)/(double)(cnt));
        
    }
    
    *hr = hrx;
    *hi = hix;
    
    
}

void faststart(lorentzianParams *ll, double *data, int N, double *splinef, double *splineA, int *Nknt, int max_lines, double Tobs, double fmin, double fmax, double fstep, double fac)
{
    int i, j, k, mc, sc;
    int scount, sacc, hold, uc;
    int NC, NCC, NCH, Ns, MM, Nk;
    int imin, imax;
    int typ, sm, sma;
    int istart, iend;
    double *freqs, *dcopy;
    int Nlines, Nknot;
    double smsc, lnsc, f;
    double x, y, z, df;
    double *SM, *PG, *SN;

    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    clock_t start, end;
    double cpu_time_used;

    double itime, ftime, exec_time;

    FILE *out;

    Ns = (int)(Tobs*fmax);

    istart = (int)(fmin*Tobs);
    iend = (int)(fmax*Tobs);
    if(iend > Ns-1) iend = Ns-1;
    
    df = 1.0/Tobs;

    freqs = (double*)malloc(sizeof(double)*(Ns));
    for (i = 0; i < Ns; ++i)   freqs[i] = (double)(i)/Tobs;
    
    // need to apply the scaling used by BayesWave
    x = sqrt(fac);
    dcopy = double_vector(N);
    for (i = 0; i < N; ++i) dcopy[i] = data[i]*x;
    
    // these are used by the spline.
    gsl_spline   *aspline;
    gsl_interp_accel *acc;
    
    lorentzfit(ll, dcopy, freqs, Tobs, N, istart, iend, max_lines, &Nlines);
    printf("Total lines %d\n", Nlines);
    
    // dcopy comes back with lines removed
    SM = double_vector(N/2);
    PG = double_vector(N/2);
    SN = double_vector(N/2);
    medspecspline(dcopy, PG, SN, SM, df, N);
    
    *Nknt = splinestart(N, istart, iend, fstep, SM, freqs, Tobs, splineA, splinef, r);
    
    free(SM);
    free(PG);
    free(SN);
    free(freqs);
    free(dcopy);

}
void blstart(lorentzianParams *line, double *data, double *residual, int N, double dt, double fmin, double fstep, int *Nsp, double *dspline, double *pspline, int max_lines)
{
  int i, j, k, Nf,  ii;
  int Nlines, Nspline;
  int flag;
  int imin, imax;
  int istart, iend;
  double SNR, max;
  double Tobs, f, df, x, y, dx;
  double fmax, fmn, Q, fny, scale;
  double *freqs;
  double *Draw;
  double *D;
  double *Sn, *PS, *SM, *SN;
  double *specD, *sspecD;
  double *sqf;
  int subscale, octaves;
  double SNRold;
  double alpha;
  double s1, s2, fac;
  double mean, sd;
    
  int Nthread = 1;
  
  time_t rawtime;
  struct tm * timeinfo;
  
  char filename[1024];
  
  FILE *outFile;
  
  Q = Qs;  // Q of transform
  Tobs = (double)(N)*dt;  // duration
  
  // Tukey window parameter. Flat for (1-alpha) of data
  alpha = (2.0*t_rise/Tobs);
  
  df = 1.0/Tobs;  // frequency resolution
  fny = 1.0/(2.0*dt);  // Nyquist
  
  // Set the range of the spectrogram.
  fmax = fny;
  fmn = 1.0/Tobs; // want to clean below where the data is used to avoid edge effects
  
  D = (double*)malloc(sizeof(double)* (N));
  Draw = (double*)malloc(sizeof(double)* (N));
  
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
  subscale = 40;  // number of semi-tones per octave
  octaves = (int)(rint(log(fmax/fmn)/log(2.0))); // number of octaves
  Nf = subscale*octaves+1;
  freqs = (double*)malloc(sizeof(double)* (Nf));   // frequencies used in the analysis
  sqf = (double*)malloc(sizeof(double)* (Nf));
  dx = log(2.0)/(double)(subscale);
  x = log(fmn);
  for(i=0; i< Nf; i++)
  {
    freqs[i] = exp(x);
    sqf[i] = sqrt(freqs[i]);
    x += dx;
  }
  
  printf("frequency layers = %d\n", Nf);
  
  scale = Getscale(freqs, Q, Tobs, fmax, N, Nf);
  
  
  sspecD = (double*)malloc(sizeof(double)*(N/2));
  specD = (double*)malloc(sizeof(double)*(N/2));
  Sn = (double*)malloc(sizeof(double)*(N/2));
  PS = (double*)malloc(sizeof(double)*(N/2));
  SM = (double*)malloc(sizeof(double)*(N/2));
  SN = (double*)malloc(sizeof(double)*(N/2));
  
  SNRold = 0.0;
  clean(D, Draw, sqf, freqs, Sn, specD, sspecD, df, Q, Tobs, scale, Nf, N, imin, imax, &SNR);
  
  // if big glitches are present we need to rinse and repeat
  i = 0;
  while(i < 5 && (SNR-SNRold) > 10.0)
  {
    SNRold = SNR;
    clean(D, Draw, sqf, freqs, Sn, specD, sspecD, df, Q, Tobs, scale, Nf, N, imin, imax, &SNR);
    i++;
  }
    
  // re-compute the power spectrum using the cleaned data
  // tukey(D, alpha, N);
  
  gsl_fft_real_radix2_transform(D, 1, N);

  // Form spectral model for whitening data (lines plus a smooth component)
    medspecspline(D, Sn, specD, sspecD, df, N);
    
  
  fac = Tobs*Tobs/((double)(N)*(double)(N))/2.;
    
  for (i = 0; i < N/2; ++i) SM[i] = sspecD[i]*fac;
  for (i = 0; i < N/2; ++i) SN[i] = specD[i]*fac;
  for (i = 0; i < N/2; ++i) PS[i] = Sn[i]*fac;
  
  //copy output of clean (D) into residual for output
  residual[0] = 0.0;
  residual[1] = 0.0;
  for(i=1; i< N/2; i++)
  {
    residual[2*i]   = D[i]   * sqrt(fac);
    residual[2*i+1] = D[N-i] * sqrt(fac);
  }
    
   faststart(line, D, N, pspline, dspline, &Nspline, max_lines, Tobs, fmin, fmax, fstep, fac);
  
  *Nsp = Nspline;
  
  free(D);
  free(Draw);
  free(sspecD);
  free(specD);
  free(Sn);
  free(PS);
  free(SN);
  free(SM);
  free(freqs);
  free(sqf);

}
static void gaussian_kernel_smoothing(double *data, int N, double sigma)
{
  int i,j;
  
  /* calculate smoothing kernel */
  int NK = (int)sigma*6*2+1; //size of Kernel (6 sigma either side of 0)
  int imid = (NK-1)/2;       //width of one side of kernel
  double K[NK+1];            //kernel

  for(i=0; i<=NK; i++)
  {
    K[i] = exp( -0.5 * (double)(i-imid) * (double)(i-imid)/sigma/sigma )/sqrt(2.*M_PI*sigma*sigma);
  }
  
  
  
  /* setup array for smoothed data */
  double *data_smoothed = malloc(N*sizeof(double));
  for(i=0; i<N; i++) data_smoothed[i] = data[i];
  
  /* convolve kernel with data */
  for(i=imid; i < N-imid; i++)
  {
    data_smoothed[i]=0.0;
    for(j=-imid; j<=imid; j++)
    {
      data_smoothed[i] += K[j+imid]*data[i+j] ;
    }
  }
  
  /* copy smoothed data into original array */
  for(i=0; i<N; i++) data[i] = data_smoothed[i];
  
  free(data_smoothed);
}


void BayesLineBurnin(struct BayesLineParams *bayesline, double *timeData, double *freqData, char *ifo, double *fprop, int SplineFlag)
{
  //Initialize BayesLine data structures
  BayesLineInitialize(bayesline);

  /*********************************************************************************************
   The pspline array holds the locations of the control points (knots) used in the spline
   The dspline array holds the values of the log of smooth spectrum at the control points
   The linef array contains the central frequencies of the Lorentzians
   The lineh array contains the amplitudes of the Lorentzians
   The lineQ array contains the Q factors of the Lorentzians
   *********************************************************************************************/

  //shortcuts to members of BayesLine structure
  dataParams *data           = bayesline->data;
  lorentzianParams *lines    = bayesline->lines_x;
  splineParams *spline       = bayesline->spline_x;

  int i,j, k, kk;
  int max_lines=bayesline->maxBLLines;
  //number of samples in the data
  double Tobs = data->Tobs;
  double fmax = data->fmax;
  double dt   = 1./data->cadence;
  double x, y;

  int N = 2*data->n;
  
  blstart(lines, timeData, freqData, N, dt, data->fmin, data->fgrid, &spline->n, spline->data, spline->points, max_lines);
   
  //Work space for assembling PSD model
  double *f     = malloc(sizeof(double)*(N+1));
  double *logSn = malloc(sizeof(double)*(N+1));
  double *Sl    = malloc(sizeof(double)*(N+1));
  
  int imin = (int)(data->fmin*Tobs);
    
  for(i=0; i<N/2; i++)
  {
    f[i]     = (double)i/Tobs;
    logSn[i] = 1.0;
    Sl[i]    = 0.0;
    
    if(i<bayesline->data->ncut){
    bayesline->Sbase[i] = 1.0;
    bayesline->Sline[i] = 0.0;
    }
  }

  //interpolate spline points
  if(SplineFlag == 1) AkimaSplineGSL(0,bayesline->data->ncut,spline->n,spline->points,spline->data,f+imin,logSn+imin);
  else if(SplineFlag == 0) CubicSplineGSL(0,bayesline->data->ncut,spline->n,spline->points,spline->data,f+imin,logSn+imin);
  
  
  //initialize work space for spline model
  bayesline->data->flow  = bayesline->data->fmin;

  for(i=0; i< bayesline->data->ncut; i++)
  {
    j = i + imin;
    bayesline->power[i] = freqData[2*j]*freqData[2*j]+freqData[2*j+1]*freqData[2*j+1];
    bayesline->spow[i]  = bayesline->power[i];
    bayesline->sfreq[i] = bayesline->freq[i];
  }
    
    x = -1.0;
    y = 1.0e10;
    for (k = 0; k < lines->n; ++k)
    {
        kk = llook(lines, lines->f[k], lines->nu[k], lines->A[k], Tobs);
        for (i = 0; i < lines->wdth; ++i)
        {
            j = kk-lines->wdth/2+i;
            if(j > 0 && j < N/2) Sl[j] += lines->lt[i];
        }
        if(lines->A[k] > x) x = lines->A[k];
        if(lines->A[k] < y) y = lines->A[k];
    }
    
    bayesline->priors->LAmin = 0.1*y;
    bayesline->priors->LAmax = 10.0*x;
    

    FILE *outfile;
    
   
  //combine spline and line model
  for(i=0; i<bayesline->data->ncut; i++)
  {
    bayesline->Sbase[i] = exp(logSn[i+imin]);
    bayesline->Sline[i] = Sl[i+imin];
    bayesline->Snf[i] = bayesline->Sbase[i] + bayesline->Sline[i];
  }
    
 /*
  outfile = fopen("xx.dat","w");
  for(i=0; i< bayesline->data->ncut; i++)
  {
      fprintf(outfile,"%e %e %e %e\n", (double)(i+imin)/Tobs, bayesline->power[i], bayesline->Sbase[i], bayesline->Sline[i]);
  }
  fclose(outfile);
   */
    
    // set up the line proposal
    for(i=0; i< bayesline->data->ncut; i++)
    {
        fprop[i] = 1.0;
        // The periodogram does not have the factor of 2 for one-sided. Putting it in here
        x = 2.0*bayesline->power[i]/bayesline->Sbase[i];
        if(x > 10.0) fprop[i] = 100.0;
    }
    
    // normalize the line proposal
    x = 0.0;
    for(i=0; i<bayesline->data->ncut; i++) x += fprop[i];
    for(i=0; i<bayesline->data->ncut; i++) fprop[i] /= x;
  
   
  //Use initial estimate of PSD to set priors
  double PRIORSCALE = 10.0;
  /*
   set the lower bound of the frequency prior to be a factor of PRIORSCALE
   lower than ~max sensitivty in the "bucket"
   */
  double f_bucket = 200.0; //Hz
  int i_bucket = (int)floor(f_bucket*Tobs);
    char filen[1024];
     double fval;
    sprintf(filen, "%s_lower_upper_priorpsd.dat", ifo);
    outfile = fopen(filen,"w");
  for(i=0; i< bayesline->data->ncut; i++)
  {
    bayesline->priors->sigma[i] = bayesline->Snf[i];
    bayesline->priors->mean[i]  = bayesline->Snf[i];
    if(bayesline->flatPriorFlag) bayesline->priors->lower[i] = bayesline->Sbase[i_bucket-imin] / (PRIORSCALE*10.);
    else                         bayesline->priors->lower[i] = bayesline->Sbase[i]/PRIORSCALE;
    //bayesline->priors->upper[i] = bayesline->Snf[i]   * PRIORSCALE;
  
    bayesline->priors->upper[i] = (bayesline->Sbase[i] + bayesline->Sline[i]*100.0)*PRIORSCALE;
    
      
    fval = (double)(i+imin)*(1.0/Tobs); 
    fprintf(outfile,"%f %e %e\n", fval, log(bayesline->priors->lower[i]), log(bayesline->priors->lower[i]*100.0));
  }
  fclose(outfile);
  //Smooth PSD prior
  gaussian_kernel_smoothing(bayesline->priors->upper, bayesline->data->ncut, 1);

    
  free(f);
  free(logSn);
  free(Sl);
 
}

void lorentzgrid(int nf, int nnu, int wdth, double numin, double numax, double ***ltemplate, int N, double Tobs, double alpha)
{
    
    double f, f0, nu, S;
    int NX, K, J;
    double TX, dt;
    int i, j, k, m;
    int ii, jj, kk;
    int w4, w34;
    double *TWL, *tuk, *tukL;
    double *lorentz;
    double lnumin, lnumax, x, y;
    
    K = 128;
    
    TX = (double)(K)*Tobs;
    NX = K*N;
    
    J = wdth*K/2;
    TWL = double_vector(J);
    tukL = double_vector(NX);
    tuk = double_vector(N);
    
    // short Tukey window
    x = 1.0/(double)(N);
    for (i = 0; i < N; ++i) tuk[i] = x;
    tukey(tuk, alpha, N);
    // padded window
    for (i = 0; i < NX; ++i) tukL[i] = 0.0;
    for (i = 0; i < N; ++i) tukL[i+NX/2] = tuk[i];
    gsl_fft_real_radix2_transform(tukL, 1, NX);
    
    TWL[0] = (tukL[0]*tukL[0]);
    for (i = 1; i < J; ++i)
    {
        TWL[i] = (tukL[i]*tukL[i]+tukL[NX-i]*tukL[NX-i]);
    }
    
    free(tuk);
    free(tukL);
    
    lnumin = log(numin);
    lnumax = log(numax);
    
    S = 1.0;
    
    lorentz = double_vector(N);
    
    // pick 40 Hz offset so as not to hit edges
    m = (int)(40.0*Tobs);
    
    for (i = 0; i < N; ++i) lorentz[i] = 0.0;
    
    for (ii = 0; ii <= nf; ++ii)
    {
        f0 = 40.0+(double)(2*ii-nf)/(double)(nf)*0.5/Tobs;  // offset from a bin
        
        for (jj = 0; jj <= nnu; ++jj)
        {
            nu = exp(lnumin+(lnumax-lnumin)*(double)(jj)/(double)(nnu));
            
            wlorentz(f0, nu, S, Tobs, J, N, K, TWL, lorentz);
            // find max
            x = 0.0;
            for (i = -4; i <= 4; ++i)
            {
                if(lorentz[m+i] > x) x= lorentz[m+i];
            }
            
            for (kk = 0; kk < wdth; ++kk)
            {
                ltemplate[ii][jj][kk] = log(lorentz[m+kk-wdth/2]/x);
            }
            
         }
        
    }
    
    free(lorentz);
    free(TWL);

    
}

void wlorentz(double f0, double nu, double S, int Tobs, int J, int N, int K, double *TWL, double *line)
{
    int i, j1, j2, k, m, NX;
    double Tlong = (double)(K)*Tobs;
    double *freq, *pf;
    double x;
    
    NX = N*K;
    m = (int)(f0*Tlong);
    
    freq = double_vector(4*J);
    pf = double_vector(4*J);
    
    j1 = m-2*J;
    j2 = m+2*J;
    
    for (i = 0; i < 4*J; ++i) freq[i] = (double)(i+j1)/(Tlong);
    lorentzraw(f0, nu, S, J, freq, pf);
    
    for (i = m-J; i < m+J; ++i)
    {
        if(i%K == 0)
        {
            if(i > 0 && i < NX/2)
            {
                x = 0.0;
                for (k = j1; k < j2; ++k)
                {
                    if(abs(i-k) < J) x += pf[k-j1]*TWL[abs(i-k)];
                }
                
               line[i/K] = x;
            }
        }
    }
    
    free(freq);
    free(pf);
    
}

void lorentzraw(double f0, double nu, double S, int J, double *freqs, double *pf)
{
    int i;
    double om0, om;
    double om02, om2, n2;
    double x;
    
    n2 = nu*nu;
    om0 = 2.0*M_PI*f0;
    om02 = om0*om0;
    
    for (i = 0; i < 4*J; ++i)
    {
        om = 2.0*M_PI*freqs[i];
        om2 = om*om;
        x = om02-om2;
        pf[i] = S/(x*x+n2*om2);
    }
    
}

static int llook_sparse_setup(lorentzianParams *restrict ll, double f0, double nu, double Tobs, int *ii, int *jj, double *y, double *z)
{
    double x, f, dfx, dnux, lnu;
    int k;
    
    if(f0 < 0.0) printf("neg\n");
    
    k = (int)(rint(f0*Tobs));
    x = (f0-(double)(k)/Tobs);
    
    if(x*Tobs <= -0.5)
    {
        k -= 1;
        x = (f0-(double)(k)/Tobs);
    }
    
    if(x*Tobs >= 0.5)
    {
        k += 1;
        x = (f0-(double)(k)/Tobs);
    }
    
    dfx = 1.0/((double)(ll->nf)*Tobs);
    dnux = (ll->lnumax-ll->lnumin)/(double)(ll->nnu);
    
    *ii = (ll->nf + (int)(floor(2.0*x*Tobs*(double)(ll->nf))))/2;
    
    // edge case
    if(*ii == ll->nf)
    {
        *ii -= 1;
    }
    
    f = (double)(2*(*ii)-ll->nf)/(double)(ll->nf)*0.5/Tobs;
    *y = (x-f)/dfx;
    
    lnu = log(nu);
    if(lnu < ll->lnumin) lnu = ll->lnumin;
    if(lnu > ll->lnumax) lnu = ll->lnumax;
    
    x = (log(nu)-ll->lnumin)/dnux;
    *jj = (int)(x);
    *z = x-(double)(*jj);
    
    // edge case
    if(*jj == ll->nnu)
    {
        *jj -= 1;
        *z = x-(double)(*jj);
    }
    
    return k;
}

static double llook_sparse_value(lorentzianParams *restrict ll, int ii, int jj, double y, double z, double A, int i)
{
    return A*exp((1.0-z)*((1.0-y)*ll->ltemplate[ii][jj][i]+y*ll->ltemplate[ii+1][jj][i])+z*((1.0-y)*ll->ltemplate[ii][jj+1][i]+y*ll->ltemplate[ii+1][jj+1][i]));
}

// returns interpolated windowed Lorentzian and the bin index
// lt needs to have size ll->wdth
int llook(lorentzianParams *restrict ll, double f0, double nu, double A, double Tobs)
{
    double x, y, z, f, dfx, dnux, lnu;
    int i, j, k, ii, jj;
    
    if(f0 < 0.0) printf("neg\n");
    
    k = (int)(rint(f0*Tobs));
    x = (f0-(double)(k)/Tobs);
    
    if(x*Tobs <= -0.5)
    {
        k -= 1;
        x = (f0-(double)(k)/Tobs);
    }
    
    if(x*Tobs >= 0.5)
    {
        k += 1;
        x = (f0-(double)(k)/Tobs);
    }
    
    // x = (double)(2*ii-nf)/(double)(nf)*0.5/Tobs;
    // 2.0*x*Tobs = (2*ii-nf)/nf
    
    dfx = 1.0/((double)(ll->nf)*Tobs);
    dnux = (ll->lnumax-ll->lnumin)/(double)(ll->nnu);
    
    ii = (ll->nf + (int)(floor(2.0*x*Tobs*(double)(ll->nf))))/2;
    
    // edge case
    if(ii == ll->nf)
    {
        ii -= 1;
    }
    
    f = (double)(2*ii-ll->nf)/(double)(ll->nf)*0.5/Tobs;
    
    //printf("%d %e %e %e %e\n", ii, f, x, (x-f)/dfx, dfx);
    
    y = (x-f)/dfx;
    
    lnu = log(nu);
    if(lnu < ll->lnumin) lnu = ll->lnumin;
    if(lnu > ll->lnumax) lnu = ll->lnumax;
    
    x = (log(nu)-ll->lnumin)/dnux;
    jj = (int)(x);
    z = x-(double)(jj);
    
    // edge case
    if(jj == ll->nnu)
    {
        jj -= 1;
        z = x-(double)(jj);
    }
    
    //printf("%d %e %e %e\n", jj, lnu, z, dnux);
    
    
   //printf("%f %e %d %d\n", f0, nu, ii, jj);
    
    for (i = 0; i < ll->wdth; ++i)
    {
     ll->lt[i]  = A*exp((1.0-z)*((1.0-y)*ll->ltemplate[ii][jj][i]+y*ll->ltemplate[ii+1][jj][i])+z*((1.0-y)*ll->ltemplate[ii][jj+1][i]+y*ll->ltemplate[ii+1][jj+1][i]));
    }
    
    return k;
    
    
}
double Lpeak(lorentzianParams *restrict ll, double *PG, double *SM, int N, int spread, double f0, double nu, double *A, double Tobs)
{
    int i, j, k, ii, jj, ilo, ihi, idx;
    double x, y, z, logL, peak, lt;
    
    k = (int)(rint(f0*Tobs));
    
    
    x = (f0-(double)(k)/Tobs);
    y = x*Tobs;
    if(y < -0.5) k -= 1;
    if(y > 0.5) k += 1;
    x = (f0-(double)(k)/Tobs);
    y = x*Tobs;
    
    if(k > 1 && k < N/2-2)
    {
        if(y < 0.0)
        {
            peak = (-y)*(PG[k-1]-SM[k-1]) + (1.0+y)*(PG[k] - SM[k]);
        }
        else
        {
            peak = (y)*(PG[k+1]-SM[k+1]) + (1.0-y)*(PG[k] - SM[k]);
        }
    }
    else
    {
        peak = 0.0;
    }
    
    if(peak < 0.0) peak = 0.0;  // shouldn't happen
    *A = peak;
    
    k = llook_sparse_setup(ll, f0, nu, Tobs, &ii, &jj, &y, &z);
    
    logL = 0.0;
    
    j = k-ll->wdth/2;
    ilo = ll->wdth/2-spread;
    ihi = ll->wdth/2+spread;
    if(ilo < 0) ilo = 0;
    if(ihi >= ll->wdth) ihi = ll->wdth-1;
    
    // compute likelihood difference with/without line
    for (i = ilo; i <= ihi; ++i)
    {
        idx = i+j;
        if(idx > 0 && idx < N/2)
        {
            lt = llook_sparse_value(ll, ii, jj, y, z, peak, i);
            logL += -log(SM[idx]+lt) - PG[idx]/(SM[idx]+lt);
            // subtract likelihood with no line contribution
            logL += log(SM[idx]) + PG[idx]/SM[idx];
        }
        
    }

    return logL;
}

void Qscan(double *dataf, double *psd, double Q, double fmin, double fmax, double dt, int N)
{
    
    // Prepare to make spectogram
    
    int i, j, k;
    int subscale, octaves, Nf;
    double dx, f, t, x, Tobs;
    double *freqs, *sqf;
    double **tfDR, **tfDI;
    double **tfD;
    
    FILE *out;
    
    Tobs = dt*(double)(N);
    
    // logarithmic frequency spacing
    subscale = 40;  // number of semi-tones per octave
    octaves = (int)(rint(log(fmax/fmin)/log(2.0))); // number of octaves
    Nf = subscale*octaves+1;
    freqs = (double*)malloc(sizeof(double)* (Nf));   // frequencies used in the analysis
    sqf = (double*)malloc(sizeof(double)* (Nf));
    dx = log(2.0)/(double)(subscale);
    x = log(fmin);
    for(i=0; i< Nf; i++)
    {
      freqs[i] = exp(x);
      sqf[i] = sqrt(freqs[i]);
      x += dx;
    }
 
      tfDR = double_matrix(Nf,N);
      tfDI = double_matrix(Nf,N);
      tfD = double_matrix(Nf,N);
    
    // whiten data
    whiten(dataf, psd, N);
    
    // Wavelet transform
    TransformC(dataf, freqs, tfD, tfDR, tfDI, Q, Tobs, N, Nf);
      
    x = 0.0;
    k =0;
    out = fopen("Qtransform.dat","w");
    for(j = 0; j < Nf; j++)
    {
        f = freqs[j];
        
        for(i = 0; i < N; i++)
        {
            t = (double)(i)*dt;
            if(t > 1.0 && t < Tobs-1.0)
            {
                fprintf(out,"%e %e %e\n", t-Tobs+4.0, f, tfD[j][i]);
                x += tfD[j][i];
                k++;
            }
        }
        
        fprintf(out,"\n");
    }
    fclose(out);
    
    printf("mean Q power %f\n", x/(double)(k));
    
}

