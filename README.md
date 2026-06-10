# LIGOtools
Python and C tools for PSD estimation and wavelet denoising of LVK data

To get LIGO open data there is the code gwoscget.py that uses gwpy. For example,

python gwoscget.py L1 1420878137 1420878145

grabs the Livingston data for 8 seconds surrounding the trigger at GPS time 1420878141. I then copy this data.txt file to something more memorable:

cp data.txt frame_8_1420878141_1.dat 

The main code is BWtest.c, which is a stand-alone driver script to run BayesLine.c. The compile instructions are in the header. You can do a test run with

./BWtest frame_8_1420878141_1.dat

The main products are psd.dat and whitelsf.dat. The first is just the PSD. The second is the whitened data, with lines subtracted, in the Fourier domain. The codes us a lookup table for the Lorentizian profiles. If the lookup table does not exist in the directory, the code will generate it and store a copy for later use.

The Python version does much the same things. It can also read LIGO frame files directly. By default it does not produce whitelsf.dat, but it can with an optional flag

python BWtest.py --file frame_8_1420878141_1.dat --writewhite

The outputs of the C and Python versions are compatible, but slightly different since they use random number generators. One difference, BWtest.py calls the PSD BWpsd.dat.

The wdm_viafreq.c code performs a WDM wavelet transform of the whitened frequency domain data and makes a bunch of graphs. The run command looks like

./wdm_viafreq whitelsf.dat 4 1

The 4 says use wavelet pixels of width 4 Hz and the 1 tells the code it is to expect frequency domain data.

For this example, there is a loud signal in the data so it fails all of the tests of Gaussianity and stationarity.

The P-value plot shows the results of applying the Anderson-Darling test to blocks of pixels, by default 4 frequency bins by 8 time bins (set at line 720, 721). These can be changed. The particular variant of the Anderson-Darling test is the one for data with known mean and variance, in this case 0 and 1. The AD score gets converted to a P-value and the -log10 of this P-value is used to color the pixels. The code also makes a histogram of the P-values, which should be uniform and lying within the shaded 1-2-3 sigma bands.

There are several other plots made. I find the most useful plot after the P-values to be is tranS.png. This is the wavelet domain power, smoothed with a Gaussian blur (default sigma's of 3 pixels in each direction). The code subtracts 1 from this (the expected average value) and use this to make the color map. Above the color map is the power as a function of frequency (found by adding up the power at each time), along with 1-2-3 sigma shaded bands for where this should sit.

If you instead run on the file whitelsf_noglitch.dat

./wdm_viafreq whitelsf_noglitch.dat 4 1

the code ingests the glitch (or in this case signal) subtracted data and applies the same tests. Interestingly, even the glitch cleaned data has a bad P-value distribution in this case. But the other plots look much cleaner.

There is an additional Anderson_Darling tests code that works in the frequency domain:

python ADtest.py psd.dat frequency_data.dat 

For the python version of BWtest, replace psd.dat with BWpsd.dat in the above command line. This code computes the average power in chunks of data (default 8 Hz bins). It also computes the Anderson-darling P-value for each 8 second chunk, using the AD test where the mean and variance are estimated from the data. The file ADplot.pdf shows the results.

The code Denoise.py (and Denoise.c) performs the same low latency PSD estimation and wavelet denoising as the BWtest codes. But it also applies clustering to the glitches and extracts the significant glitches. The Python version makes a before and after animated "Qscan_blink.gif".
The Denoise code also produces the files features.dat and features_colored.dat containing the white and re-colored non-Gaussian features.

For example, if you run

python Denoise.py --file frame_8_1420878141_1.dat

then plot feature.dat between 3.5 and 4.5 seconds and you will see a nice BH merger reconstruction.
