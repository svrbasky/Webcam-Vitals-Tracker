#ifndef vita_hpp
#define vita_hpp

#include <stdio.h>
#include <math.h>
//#include <fftw3.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <chrono>
#include <complex>

#include <librealsense2/rs.hpp>			// Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>			// Include OpenCV API
//#include "RPPG.hpp"
//#include "opencvhelper.hpp"			// local opencv functions

#include <opencv2/imgcodecs.hpp>
//#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#define SHAPE_FILE_PATH "../data/shape_predictor_81_face_landmarks.dat"
#define LENGTH_PPG 64
#define MIN_TS 0.05
#define PI 3.1415926535897932384626433832795
#define LOW_BPM 42
#define HIGH_BPM 240
#define SEC_PER_MIN 60
const cv::Scalar ONE (1);
const std::complex<double> ii(0,1);

using namespace dlib;
using namespace std;
using namespace cv;

struct PPG_params
{
    std::chrono::high_resolution_clock::time_point timeVal;
    double avgcVal;
    double avgiVal;
    double avgrVal;
};

class vita {
    
private:

    double *ComputeLP( int FilterOrder )
    {
        double *NumCoeffs;
        int m;
        int i;

        NumCoeffs = (double *)calloc( FilterOrder+1, sizeof(double) );
        if( NumCoeffs == NULL ) return( NULL );

        NumCoeffs[0] = 1;
        NumCoeffs[1] = FilterOrder;
        m = FilterOrder/2;
        for( i=2; i <= m; ++i)
        {
            NumCoeffs[i] =(double) (FilterOrder-i+1)*NumCoeffs[i-1]/i;
            NumCoeffs[FilterOrder-i]= NumCoeffs[i];
        }
        NumCoeffs[FilterOrder-1] = FilterOrder;
        NumCoeffs[FilterOrder] = 1;

        return NumCoeffs;
    }

    double *ComputeHP( int FilterOrder )
    {
        double *NumCoeffs;
        int i;

        NumCoeffs = ComputeLP(FilterOrder);
        if(NumCoeffs == NULL ) return( NULL );

        for( i = 0; i <= FilterOrder; ++i)
            if( i % 2 ) NumCoeffs[i] = -NumCoeffs[i];

        return NumCoeffs;
    }

    double *TrinomialMultiply( int FilterOrder, double *b, double *c )
    {
        int i, j;
        double *RetVal;

        RetVal = (double *)calloc( 4 * FilterOrder, sizeof(double) );
        if( RetVal == NULL ) return( NULL );

        RetVal[2] = c[0];
        RetVal[3] = c[1];
        RetVal[0] = b[0];
        RetVal[1] = b[1];

        for( i = 1; i < FilterOrder; ++i )
        {
            RetVal[2*(2*i+1)]   += c[2*i] * RetVal[2*(2*i-1)]   - c[2*i+1] * RetVal[2*(2*i-1)+1];
            RetVal[2*(2*i+1)+1] += c[2*i] * RetVal[2*(2*i-1)+1] + c[2*i+1] * RetVal[2*(2*i-1)];

            for( j = 2*i; j > 1; --j )
            {
                RetVal[2*j]   += b[2*i] * RetVal[2*(j-1)]   - b[2*i+1] * RetVal[2*(j-1)+1] +
                    c[2*i] * RetVal[2*(j-2)]   - c[2*i+1] * RetVal[2*(j-2)+1];
                RetVal[2*j+1] += b[2*i] * RetVal[2*(j-1)+1] + b[2*i+1] * RetVal[2*(j-1)] +
                    c[2*i] * RetVal[2*(j-2)+1] + c[2*i+1] * RetVal[2*(j-2)];
            }

            RetVal[2] += b[2*i] * RetVal[0] - b[2*i+1] * RetVal[1] + c[2*i];
            RetVal[3] += b[2*i] * RetVal[1] + b[2*i+1] * RetVal[0] + c[2*i+1];
            RetVal[0] += b[2*i];
            RetVal[1] += b[2*i+1];
        }

        return RetVal;
    }

    double *ComputeNumCoeffs(int FilterOrder,double Lcutoff, double Ucutoff, double *DenC)
    {
        double *TCoeffs;
        double *NumCoeffs;
        std::complex<double> *NormalizedKernel;
        double Numbers[11]={0,1,2,3,4,5,6,7,8,9,10};
        int i;

        NumCoeffs = (double *)calloc( 2*FilterOrder+1, sizeof(double) );
        if( NumCoeffs == NULL ) return( NULL );

        NormalizedKernel = (std::complex<double> *)calloc( 2*FilterOrder+1, sizeof(std::complex<double>) );
        if( NormalizedKernel == NULL ) return( NULL );

        TCoeffs = ComputeHP(FilterOrder);
        if( TCoeffs == NULL ) return( NULL );
        
        for( i = 0; i < FilterOrder; ++i)
        {
            NumCoeffs[2*i] = TCoeffs[i];
            NumCoeffs[2*i+1] = 0.0;
        }
        NumCoeffs[2*FilterOrder] = TCoeffs[FilterOrder];
        
        double Wn, cp[2];
        
        cp[0] = 2 * 2.0 * std::tan(PI * Lcutoff / 2.0);
        cp[1] = 2 * 2.0 * std::tan(PI * Ucutoff / 2.0);
        
        //center frequency
        Wn = std::sqrt(cp[0] * cp[1]);
        Wn = 2 * std::atan2(Wn, 4);
        
        for(int k = 0; k<11; k++)
        {
            NormalizedKernel[k] = std::exp(-ii * (Wn * Numbers[k]));
        }
        double b=0;
        double den=0;
        for(int d = 0; d<11; d++)
        {
            b+=real(NormalizedKernel[d]*NumCoeffs[d]);
            den+=real(NormalizedKernel[d]*DenC[d]);
        }
        
        for(int c = 0; c<11; c++)
        {
            NumCoeffs[c]=(NumCoeffs[c]*den)/b;
        }
        

        //free(TCoeffs);
        return NumCoeffs;
    }

    double *ComputeDenCoeffs( int FilterOrder, double Lcutoff, double Ucutoff )
    {
        int k;            // loop variables
        double theta;     // PI * (Ucutoff - Lcutoff) / 2.0
        double cp;        // cosine of phi
        double st;        // sine of theta
        double ct;        // cosine of theta
        double s2t;       // sine of 2*theta
        double c2t;       // cosine 0f 2*theta
        double *RCoeffs;     // z^-2 coefficients
        double *TCoeffs;     // z^-1 coefficients
        double *DenomCoeffs;     // dk coefficients
        double PoleAngle;      // pole angle
        double SinPoleAngle;     // sine of pole angle
        double CosPoleAngle;     // cosine of pole angle
        double a;         // workspace variables

        cp = cos(PI * (Ucutoff + Lcutoff) / 2.0);
        theta = PI * (Ucutoff - Lcutoff) / 2.0;
        st = sin(theta);
        ct = cos(theta);
        s2t = 2.0*st*ct;        // sine of 2*theta
        c2t = 2.0*ct*ct - 1.0;  // cosine of 2*theta

        RCoeffs = (double *)calloc( 2 * FilterOrder, sizeof(double) );
        TCoeffs = (double *)calloc( 2 * FilterOrder, sizeof(double) );

        for( k = 0; k < FilterOrder; ++k )
        {
            PoleAngle = PI * (double)(2*k+1)/(double)(2*FilterOrder);
            SinPoleAngle = sin(PoleAngle);
            CosPoleAngle = cos(PoleAngle);
            a = 1.0 + s2t*SinPoleAngle;
            RCoeffs[2*k] = c2t/a;
            RCoeffs[2*k+1] = s2t*CosPoleAngle/a;
            TCoeffs[2*k] = -2.0*cp*(ct+st*SinPoleAngle)/a;
            TCoeffs[2*k+1] = -2.0*cp*st*CosPoleAngle/a;
        }

        DenomCoeffs = TrinomialMultiply(FilterOrder, TCoeffs, RCoeffs );
        //free(TCoeffs);
        //free(RCoeffs);

        DenomCoeffs[1] = DenomCoeffs[0];
        DenomCoeffs[0] = 1.0;
        for( k = 3; k <= 2*FilterOrder; ++k )
            DenomCoeffs[k] = DenomCoeffs[2*k-2];


        return DenomCoeffs;
    }

    void filter(int ord, double *a, double *b, int np, double *x, double *y)
    {
        int i,j;
        y[0]=b[0] * x[0];
        for (i=1;i<ord+1;i++)
        {
            y[i]=0.0;
            for (j=0;j<i+1;j++)
                y[i]=y[i]+b[j]*x[i-j];
            for (j=0;j<i;j++)
                y[i]=y[i]-a[j+1]*y[i-j-1];
        }
        for (i=ord+1;i<np+1;i++)
        {
            y[i]=0.0;
            for (j=0;j<ord+1;j++)
                y[i]=y[i]+b[j]*x[i-j];
            for (j=0;j<ord;j++)
                y[i]=y[i]-a[j+1]*y[i-j-1];
        }
    }

        
    void detrend(double *y, double *out)
    {
        double xmean, ymean;
        int i;
        double temp;
        double Sxy;
        double Sxx;

        double grad;
        double yint;
        double x[LENGTH_PPG];

        for (i = 0; i < LENGTH_PPG; i++)
            x[i] = i;

        xmean = 0;
        ymean = 0;
        for (i = 0; i < LENGTH_PPG; i++)
        {
            xmean += x[i];
            ymean += y[i];
        }
        xmean /= LENGTH_PPG;
        ymean /= LENGTH_PPG;

        temp = 0;
        for (i = 0; i < LENGTH_PPG; i++)
            temp += x[i] * y[i];
        Sxy = temp / LENGTH_PPG - xmean * ymean;

        temp = 0;
        for (i = 0; i < LENGTH_PPG; i++)
            temp += x[i] * x[i];
        Sxx = temp / LENGTH_PPG - xmean * xmean;

        grad = Sxy / Sxx;
        yint = -grad * xmean + ymean;

        for (i = 0; i < LENGTH_PPG; i++)
            out[i] = y[i] - (grad * i + yint);

    }


    // Moving average filter (low pass equivalent)
    void movingAverage(InputArray _a, OutputArray _b, int n)
    {
        int s = (floor(vita::fps/6) > 2) ? floor(vita::fps/6) : 2;

        CV_Assert(s > 0);

        _a.getMat().copyTo(_b);
        Mat b = _b.getMat();
        for (size_t i = 0; i < n; i++)
        {
            cv::blur(b, b, Size(s, s));
        }
    }
    
    void normalization(InputArray _a, OutputArray _b)
    {
           _a.getMat().copyTo(_b);
           Mat b = _b.getMat();
           Scalar mean, stdDev;
           for (int i = 0; i < b.cols; i++)
           {
               meanStdDev(b.col(i), mean, stdDev);
               b.col(i) = (b.col(i) - mean[0]) / stdDev[0];
           }
    }
    
    void timeToFrequency(InputArray _a, OutputArray _b, bool magnitude)
    {

        // Prepare planes
        Mat a = _a.getMat();
        Mat planes[] = {cv::Mat_<float>(a), cv::Mat::zeros(a.size(), CV_32F)};
        Mat powerSpectrum;
        merge(planes, 2, powerSpectrum);

        // Fourier transform
        dft(powerSpectrum, powerSpectrum, DFT_COMPLEX_OUTPUT);

        if (magnitude)
        {
            split(powerSpectrum, planes);
            cv::magnitude(planes[0], planes[1], planes[0]);
            planes[0].copyTo(_b);
        }
        else
        {
            powerSpectrum.copyTo(_b);
        }
    }
    
    void pcaComponent(cv::InputArray _a, cv::OutputArray _b, cv::OutputArray _pc, int low, int high)
    {

        Mat a = _a.getMat();
        CV_Assert(a.type() == CV_64F);

        // Perform PCA
        cv::PCA pca(a, cv::Mat(), PCA::DATA_AS_ROW);

        // Calculate PCA components
        cv::Mat pc = a * pca.eigenvectors.t();

        // Band mask
        const int total = a.rows;
        Mat bandMask = Mat::zeros(a.rows, 1, CV_8U);
        bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);

        // Identify most distinct
        std::vector<double> vals;
        for (int i = 0; i < pc.cols; i++)
        {
            cv::Mat magnitude = Mat(pc.rows, 1, CV_32F);
            // Calculate spectral magnitudes
            timeToFrequency(pc.col(i), magnitude, true);
            // Normalize
            //printMat<float>("magnitude1", magnitude);
            normalize(magnitude, magnitude, 1, 0, NORM_L1, -1, bandMask);
            //printMat<float>("magnitude2", magnitude);
            // Grab index of max
            double min, max;
            Point pmin, pmax;
            cv::minMaxLoc(magnitude, &min, &max, &pmin, &pmax, bandMask);
            vals.push_back(max);

        }

        // Select most distinct
        int idx[2];
        cv::minMaxIdx(vals, 0, 0, 0, &idx[0]);
        if (idx[0] == -1)
        {
            pc.col(1).copyTo(_b);
        }
        else
        {
            //pc.col(1).copyTo(_b);
            pc.col(idx[1]).copyTo(_b);
        }

        pc.copyTo(_pc);
    }
    
    void linspace(double start, double end, double num, double *linspaced)
    {
        int i=0;
        double delta = (end - start) / (num - 1);

        for(i=0; i < num-1; ++i)
        {
            linspaced[i] = (start + delta * i);
        }
        linspaced[i] = end;
    }
    
    double interpolate( double *xData, double *yData, double x )
    {
        int i = 0;
        
        if ( x >= xData[LENGTH_PPG - 2] )
        {
            i = LENGTH_PPG - 2;
        }
        else
        {
            while ( x > xData[i+1] ) i++;
        }
        
        double xL = xData[i], yL = yData[i], xR = xData[i+1], yR = yData[i+1];

        double dydx = ( yR - yL ) / ( xR - xL );

        return yL + dydx * ( x - xL );
    }
    
    void split_array(complex<double> *inputs, int N)
    {
        complex<double>* even = new complex<double>[N/2];
        complex<double>* odd = new complex<double>[N/2];
        int ei = 0;
        int oi = 0;

        for (int j=0; j < N; j++)
        {
            if ((j % 2) == 0)
                even[ei++] = inputs[j];
            else
                odd[oi++] = inputs[j];
        }

        int size = N/2;
        memcpy(inputs, even, sizeof(complex<double>)*size);
        memcpy(inputs+size, odd, sizeof(complex<double>)*size);

        delete[] even;
        delete[] odd;
    }
    
    void fast_fourier(complex<double> *x, double N)
    {
        /* base of recursion */
        if (N == 1)
            return;
        
        /* no rounding needed if N is base 2 */
        int n = N/2;
        
        
        /* set primitive root of unity */
        complex<double> wn = exp((2*PI*ii)/N);
        complex<double> w = 1;

        /* move odd and evened indexed to each half
           of array x */
        split_array(x, 2*n);
        
        /* even and odd */
        fast_fourier(x, n);
        /* pass pointer starting
        at the n/2th element */
        fast_fourier(x+n, n);

        complex<double> even(0,0);
        complex<double> odd(0,0);

        for (int k = 0; k < n; k++)
        {
            /* code */
            even = x[k];
            odd = x[k+n]; /* k + N/2 */

            x[k] = even + w*odd;
            x[k+n] = even - w*odd;

            /* multiply same as k+1
               in exponent */
            w = w*wn;
        }
    }
    
    void extract_vitals(double *xs, double *avgc, double *avgi)
    {
        double xL[LENGTH_PPG], yc[LENGTH_PPG], yi[LENGTH_PPG];
        for (int i = 0; i< LENGTH_PPG; i++)
        {
            xL[i] = 0;
            yc[i] = 0;
            yi[i] = 0;
        }
        
        vita::linspace(xs[0], xs[LENGTH_PPG-1], (double)LENGTH_PPG, xL);
        
        double multiplier = 1;
        double ycmean = 0;
        double yimean = 0;
        for (int i = 0; i< LENGTH_PPG; i++)
        {
            multiplier = 0.54 - (0.46 * cos( 2 * PI * (i / ((LENGTH_PPG - 1) * 1.0))));
            yc[i] = multiplier * interpolate( xs, avgc, xL[i]);
            yi[i] = multiplier * interpolate( xs, avgi, xL[i]);
            ycmean += yc[i];
            yimean += yi[i];
        }
        ycmean /= LENGTH_PPG;
        yimean /= LENGTH_PPG;
        
        for (int i = 0; i< LENGTH_PPG; i++)
        {
            yc[i] -= ycmean;
            yi[i] -= yimean;
        }
        
        complex<double> sC[LENGTH_PPG], sI[LENGTH_PPG];
        
        for(int i=0; i<LENGTH_PPG; i++)
        {
            sC[i].real(yc[i]);
            sC[i].imag(0);
            
            sI[i].real(yi[i]);
            sI[i].imag(0);
           
        }
        
        fast_fourier(sC, LENGTH_PPG);
        fast_fourier(sI, LENGTH_PPG);
        
        double prunI[LENGTH_PPG], prunC[LENGTH_PPG], pfreq[LENGTH_PPG], pfreqC[LENGTH_PPG];
        
        for (int i = 0; i < LENGTH_PPG; ++i)
        {
            prunC[i] = 2*sqrt(pow(sC[i].real(),2) + pow(sC[i].imag(),2))/LENGTH_PPG;
            
            prunI[i] = 2*sqrt(pow(sI[i].real(),2) + pow(sI[i].imag(),2))/LENGTH_PPG;
        }
        
        double freqs[LENGTH_PPG];
        
        for (int i = 0; i< LENGTH_PPG/2+1; i++)
        {
            freqs[i] = ((double)SEC_PER_MIN * vita::fps / (double)LENGTH_PPG) * i;
        }
        
        
        for (int i=0; i<LENGTH_PPG/2+1; i++)
        {
            if(freqs[i] > 60 && freqs[i] < 120)
            {
                pfreq[i] = freqs[i];
            }
            else
            {
                    prunC[i] = -5000;
                    prunI[i] = -5000;
                    pfreq[i] = -5000;
            }
        }
        
        int idxC = 0;
        int idxI = 0;
        double maxC = -5000;
        double minC = 5000;
        double minI = 5000;
        double maxI = -5000;
        for (int i=0; i<LENGTH_PPG/2+1; i++)
        {
            if(prunC[i] != -5000 && prunC[i] > maxC)
            {
                maxC = prunC[i];
                idxC = i;
            }
            if(prunI[i] != -5000 && prunI[i] > maxI)
            {
                maxI = prunI[i];
                idxI = i;
            }
            if(prunC[i] != -5000 && prunC[i] < minC)
            {
                minC = prunC[i];
            }
            if(prunI[i] != -5000 && prunI[i] < minI)
            {
                minI = prunI[i];
            }
        }
        double HR_C = pfreq[idxC];
        double HR_I = pfreq[idxI];
        
        vita::HR = (HR_C + HR_I)/2;
        
        double RC = maxC/minC;
        double RI = maxI/minI;
        double Ratio = RC/RI;
        vita::SPO2 = 125 - (26 * Ratio);
        
        for (int i = 0; i < LENGTH_PPG; ++i)
        {
            prunC[i] = sqrt(sC[i].real() * sC[i].real() + sC[i].imag() * sC[i].imag());
            
            prunI[i] = sqrt(sI[i].real() * sI[i].real() + sI[i].imag() * sI[i].imag());
        }
        
        for (int i=0; i<LENGTH_PPG/2+1; i++)
        {
            if(freqs[i] > 10 && freqs[i] < 60)
            {
                pfreq[i] = freqs[i];
            }
            else
            {
                    prunC[i] = -5000;
                    prunI[i] = -5000;
                    pfreq[i] = -5000;
            }
        }
        
        idxC = 0;
        idxI = 0;
        maxC = -5000;
        maxI = -5000;
        for (int i=0; i<LENGTH_PPG/2+1; i++)
        {
            if(prunC[i] != -5000 && prunC[i] > maxC)
            {
                maxC = prunC[i];
                idxC = i;
            }
            if(prunI[i] != -5000 && prunI[i] > maxI)
            {
                maxI = prunI[i];
                idxI = i;
            }
        }
        double RR_C = pfreq[idxC];
        double RR_I = pfreq[idxI];
        
        vita::RR = (RR_C + RR_I)/2;
        
        double max_yi = yi[0];
        double max_yc = yc[0];
        double min_yc = yc[0];
        double min_yi = yi[0];
        double mean_yi = 0;
        double mean_yc = 0;
        for (int i=0; i<LENGTH_PPG; i++)
        {
            mean_yi += yi[i];
            mean_yc += yc[i];
            if(yi[i]>max_yi) max_yi = yi[i];
            if(yc[i]>max_yc) max_yc = yc[i];
            if(yi[i]<min_yi) min_yi = yi[i];
            if(yc[i]<min_yi) min_yc = yc[i];
        }
        
        mean_yi /= LENGTH_PPG;
        mean_yc /= LENGTH_PPG;
        
        
        double mNPV_i = abs(max_yi - min_yi) / abs(mean_yi);
        double mNPV_c = abs(max_yc - min_yc) / abs(mean_yc);

        double SBP_i = exp(0.379 * log(HR_I) + 0.015 * log(mNPV_i) + 2.515);
        double DBP_i = exp(0.389 * log(HR_I) - 0.002 * log(mNPV_i) + 2.719);
        
        double SBP_c = exp(0.379 * log(HR_C) + 0.015 * log(mNPV_c) + 2.515);
        double DBP_c = exp(0.389 * log(HR_C) - 0.002 * log(mNPV_c) + 2.719);
        
        vita::SBP = (SBP_i + SBP_c)/2;
        vita::DBP = (DBP_i + DBP_c)/2;
    }
    
public:
    pthread_mutex_t mutex_img = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mutex_add_array = PTHREAD_MUTEX_INITIALIZER;
    
    int count;
    PPG_params ppgArray[LENGTH_PPG];
    cv::VideoCapture cam;
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    //rs2::pipeline pipe;
	//rs2::config config;
	int w = 640;
	int h = 480;
    //cv::Mat frame;
    
	cv::Mat imgI;
	cv::Mat imgC;
	cv::Mat image;
    cv::Mat bottom_data;
    cv::Mat bottom_ones;
    cv::Mat top_ones;
    cv::Mat spd_image;
    
    Rect lrect;
    Rect hrect;
    Rect rrect;
    
	//rs2::frameset data;
	//rs2::frame depth;
	//rs2::frame color;
	std::chrono::high_resolution_clock::time_point timeStart;
    
	dlib::frontal_face_detector detector;
	dlib::shape_predictor sp;
    dlib::full_object_detection shapes;
    std::vector<dlib::rectangle> faces;
    cv::Rect faces1;

	double fps;
    double HR;
    double RR;
    double SPO2;
    double SBP;
    double DBP;
    int start_flg = 1;
    bool flg = false;
    
    
    static void *start_process_thread(void *ptr) { return ((vita *)ptr)->detectFace(); }
    
    static void *start_extract_thread(void *ptr) { return ((vita *)ptr)->extractSignal(); }
    
	void capture_frame()
	{
		//vita::data = vita::pipe.wait_for_frames();
		//vita::depth = vita::data.get_infrared_frame();
		//vita::color = vita::data.get_color_frame();
        
		//vita::w = vita::depth.as<rs2::video_frame>().get_width();
		//vita::h = vita::depth.as<rs2::video_frame>().get_height();

		// Create OpenCV matrix of size (w,h) from the colorized depth data
		//cv::Mat img(cv::Size(vita::w, vita::h), CV_8UC1, (void*)vita::depth.get_data(), cv::Mat::AUTO_STEP);
        
        pthread_mutex_lock( &mutex_img );
        vita::cam.read(vita::imgC);
        
		//vita::imgI = img;
		//cvtColor(vita::imgI,vita::imgI,COLOR_GRAY2RGB);
		//cv::Mat img1(cv::Size(vita::w, vita::h), CV_8UC3, (void*)vita::color.get_data(), cv::Mat::AUTO_STEP);
		//vita::imgC = img1;
        
        cv::Mat bgr[3];
        cv::split(vita::imgC,bgr);
        cvtColor(bgr[1], vita::imgI, cv::COLOR_GRAY2RGB);
        pthread_mutex_unlock( &mutex_img );
        pthread_mutex_lock( &mutex_img );
        if(vita::faces.size() > 0 && vita::start_flg == 0)
        {
            cv::rectangle(vita::imgI, vita::faces1, Scalar( 255, 0, 0 ));
            cv::rectangle(vita::imgC, vita::faces1, Scalar( 255, 0, 0 ));
            //cv::rectangle(vita::imgC, vita::lrect, Scalar( 0, 0, 255 ));
            //cv::rectangle(vita::imgC, vita::hrect, Scalar( 0, 0, 255 ));
            //cv::rectangle(vita::imgC, vita::rrect, Scalar( 0, 0, 255 ));
        }
        
        pthread_mutex_unlock( &mutex_img );
        pthread_mutex_lock( &mutex_img );
		cv::hconcat(vita::imgI, vita::imgC, vita::image);
        top_ones = cv::Mat(cv::Size(vita::spd_image.cols, 50), CV_8UC3, cv::Scalar(0,0,0));
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now()-vita::timeStart).count();
        std::ostringstream strS1;
        strS1 << "T.elap: "<<elapsed_seconds<<" s";
        std::string cStr = strS1.str();
        cv::putText(vita::top_ones, cStr, cv::Point(vita::top_ones.cols - 250, vita::top_ones.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,127,255), 1);
        cv::vconcat(vita::top_ones, vita::image, vita::image);
		cv::vconcat(vita::spd_image, vita::image, vita::image);
        
        cv::vconcat(vita::image, vita::bottom_data, vita::image);
		cv::vconcat(vita::image, vita::bottom_ones, vita::image);
        pthread_mutex_unlock( &mutex_img );
        
	}

	vita()
	{
		//vita::config.enable_stream(RS2_STREAM_INFRARED,1,640,480,RS2_FORMAT_Y8,60);
		//vita::config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,60);
		//vita::pipe.start(vita::config);

        vita::cam.open(deviceID, apiID);
        vita::cam.set(cv::CAP_PROP_FRAME_WIDTH, vita::w);
        vita::cam.set(cv::CAP_PROP_FRAME_HEIGHT, vita::h);
        if (!vita::cam.isOpened()) {
            cerr << "ERROR! Unable to open camera\n";
            //return -1;
        }
        
		vita::spd_image = cv::imread("../data/spd.png");
		cv::Mat zerosM = cv::Mat(cv::Size(640,vita::spd_image.rows),CV_8UC3,cv::Scalar(0,0,0));
		cv::hconcat(vita::spd_image, zerosM, vita::spd_image);
		cv::Mat imgBuf = cv::Mat(cv::Size(40, vita::spd_image.rows), CV_8UC3, cv::Scalar(0,0,0));
		cv::hconcat(imgBuf, vita::spd_image, vita::spd_image);
        
        vita::bottom_data = cv::Mat(cv::Size(vita::spd_image.cols,200), CV_8UC3, cv::Scalar(0,0,0));
        vita::bottom_ones = cv::Mat(cv::Size(vita::spd_image.cols,100), CV_8UC3, cv::Scalar(0,0,0));
        
        std::ostringstream strS1;
        strS1 << "Press <SPACE> to start or pause. Press <ESC> to close";
        std::string cStr = strS1.str();
        cv::putText(vita::bottom_ones, cStr, cv::Point(50, vita::bottom_ones.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,255,255), 1);
        
		cv::Mat onesM = cv::Mat(cv::Size(vita::spd_image.cols, 100), CV_8UC3, cv::Scalar(0,0,0));
		cv::vconcat(onesM, vita::spd_image, vita::spd_image);
        
		cv::putText(vita::spd_image, "Remote PPG Demo", cv::Point(400, vita::spd_image.rows/2 + 20), cv::FONT_HERSHEY_DUPLEX, 2, CV_RGB(255,255,255), 2);
        
        cv::Mat onesM1 = cv::Mat(cv::Size(640,480), CV_8UC3, cv::Scalar(0,0,0));
        
        pthread_mutex_lock( &mutex_img );
        cv::hconcat(onesM1, onesM1, vita::image);
        
        top_ones = cv::Mat(cv::Size(vita::spd_image.cols, 50), CV_8UC3, cv::Scalar(0,0,0));
        cv::vconcat(vita::top_ones, vita::image, vita::image);
        
        cv::vconcat(vita::spd_image, vita::image, vita::image);
        cv::vconcat(vita::image, vita::bottom_data, vita::image);
        cv::vconcat(vita::image, vita::bottom_ones, vita::image);
        pthread_mutex_unlock( &mutex_img );
		vita::detector = dlib::get_frontal_face_detector();
		dlib::deserialize(SHAPE_FILE_PATH) >> vita::sp;
        
        pthread_mutex_lock( &mutex_add_array );
        PPG_params _ppgVal = PPG_params{std::chrono::high_resolution_clock::now(),-1,-1,-1};
        for(int i=0; i<LENGTH_PPG; i++)
        {
            ppgArray[i] = _ppgVal;
        }
        vita::count = 0;
        pthread_mutex_unlock( &mutex_add_array );
        vita::lrect = Rect(0, 0, 1, 1);
        vita::hrect = Rect(0, 0, 1, 1);
        vita::rrect = Rect(0, 0, 1, 1);
        vita::timeStart = std::chrono::high_resolution_clock::now();
        
	}
    
	~vita(){}
    
    void* detectFace()
    {
        while(1)
        {
            if(vita::flg)
                break;
            
            if(vita::start_flg == 1)
            {
                pthread_mutex_lock( &mutex_img );
                vita::bottom_data = cv::Mat(cv::Size(vita::spd_image.cols,200), CV_8UC3, cv::Scalar(0,0,0));
                vita::bottom_ones = cv::Mat(cv::Size(vita::spd_image.cols,100), CV_8UC3, cv::Scalar(0,0,0));
                std::ostringstream strS1, strS2;
                strS1 << "Press <SPACE> to start or pause. Press <ESC> to close";
                std::string cStr = strS1.str();
                cv::putText(vita::bottom_ones, cStr, cv::Point(50, vita::bottom_ones.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,255,255), 1);
                strS2 << "Paused";
                std::string cStr2 = strS2.str();
                cv::putText(vita::bottom_ones, cStr2, cv::Point(vita::bottom_ones.cols - 200, vita::bottom_ones.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255,0,0), 1);
                pthread_mutex_unlock( &mutex_img );
                pthread_mutex_lock( &mutex_add_array );
                PPG_params _ppgVal = PPG_params{std::chrono::high_resolution_clock::now(),-1,-1,-1};
                for(int i=0; i<LENGTH_PPG; i++)
                {
                    ppgArray[i] = _ppgVal;
                }
                vita::count = 0;
                pthread_mutex_unlock( &mutex_add_array );
                //break;
            }
            else
            {
                pthread_mutex_lock( &mutex_img );
                vita::bottom_ones = cv::Mat(cv::Size(vita::spd_image.cols,100), CV_8UC3, cv::Scalar(0,0,0));
                std::ostringstream strS1, strS2;
                strS1 << "Press <SPACE> to start or pause. Press <ESC> to close";
                std::string cStr = strS1.str();
                cv::putText(vita::bottom_ones, cStr, cv::Point(50, vita::bottom_ones.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,255,255), 1);
                strS2 << "Running";
                std::string cStr2 = strS2.str();
                cv::putText(vita::bottom_ones, cStr2, cv::Point(vita::bottom_ones.cols - 200, vita::bottom_ones.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0,255,0), 1);
                pthread_mutex_unlock( &mutex_img );
                
                pthread_mutex_lock( &mutex_img );
                cv::Mat bgr[3];
                cv::split(vita::imgC,bgr);
                pthread_mutex_unlock( &mutex_img );
                cv_image<unsigned char> cimg(bgr[1]);
                pthread_mutex_lock( &mutex_img );
                vita::faces = vita::detector(cimg);
                pthread_mutex_unlock( &mutex_img );
                cout << "Number of faces detected: " << vita::faces.size() << endl;
                if(vita::faces.size() > 0)
                {
                    vita::shapes = vita::sp(cimg, faces[0]);
                    pthread_mutex_lock( &mutex_img );
                    vita::faces1 = cv::Rect(cv::Point2i(faces[0].left(), faces[0].top()), cv::Point2i(faces[0].right() + 1, faces[0].bottom() + 1));

                    vita::lrect = Rect((int)shapes.part(17).x(),(int)shapes.part(28).y(), (int)abs(shapes.part(17).x() - shapes.part(31).x()), (int)abs(shapes.part(28).y() - shapes.part(48).y()));
                    vita::hrect = Rect((int)shapes.part(18).x(),(int)shapes.part(69).y(), (int)abs(shapes.part(18).x() - shapes.part(25).x()), (int)abs(shapes.part(69).y() - (shapes.part(25).y() - shapes.part(18).y())/2));
                    vita::rrect = Rect((int)shapes.part(35).x(),(int)shapes.part(28).y(), (int)abs(shapes.part(35).x() - shapes.part(26).x()), (int)abs(shapes.part(28).y() - shapes.part(54).y()));
                    pthread_mutex_unlock( &mutex_img );
                    pthread_mutex_lock( &mutex_img );
                    
                    double lr = cv::mean(imgI(lrect))[0];
                    double hr = cv::mean(imgI(hrect))[0];
                    double rr = cv::mean(imgI(rrect))[0];
                    double _avgi = lr + hr + rr;
            
                    double lc = cv::mean(imgC(lrect))[0];
                    double hc = cv::mean(imgC(hrect))[0];
                    double rc = cv::mean(imgC(rrect))[0];
                    double _avgc = lc + hc + rc;
                    pthread_mutex_unlock( &mutex_img );
                    double _avgr = _avgi/_avgc;

                    if(vita::count < LENGTH_PPG)
                    {
                        if((isnan(_avgc) || _avgc <= 0) || (isnan(_avgi) || _avgi <= 0) || (isnan(_avgr) || _avgr <= 0))
                        {
                            pthread_mutex_lock( &mutex_add_array );
                            pthread_mutex_unlock( &mutex_add_array );
                        }
                        else
                        {
                            pthread_mutex_lock( &mutex_add_array );
                            PPG_params _ppgVal = PPG_params{ std::chrono::high_resolution_clock::now() ,_avgi,_avgc,_avgr};
                            ppgArray[count] = _ppgVal;
                            vita::count++;
                            pthread_mutex_unlock( &mutex_add_array );
                        }
                    }
                    else
                    {
                        //double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(ppgArray[vita::count-1].timeVal-ppgArray[0].timeVal).count();
                        //cout<<"fps: " << double(LENGTH_PPG)/elapsed_seconds << " elp. s: "<< elapsed_seconds << endl;
                        //break;
                    }
                }
            }
        }
        return (void*)-1;
    }
    
    void* extractSignal()
    {
        while(1)
        {
            if(vita::flg)
                break;
            
            if(vita::start_flg == 1)
            {
                //break;
            }
            else
            {
                if(vita::count == LENGTH_PPG)
                {
                    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(ppgArray[vita::count-1].timeVal - ppgArray[0].timeVal).count();
                    vita::fps = double(LENGTH_PPG)/elapsed_seconds;
            
                    double nyq = vita::fps/2.0;
                    double FrequencyBands[2] = {0.5/nyq, 3.0/nyq};
            
                    int FiltOrd = 5;

                    double *DenC = 0;
                    double *NumC = 0;
            
                    DenC = vita::ComputeDenCoeffs(FiltOrd, FrequencyBands[0], FrequencyBands[1]);
                    NumC = vita::ComputeNumCoeffs(FiltOrd, FrequencyBands[0], FrequencyBands[1], DenC);
                    
                    double avgi[LENGTH_PPG], avgc[LENGTH_PPG], avgr[LENGTH_PPG], xs[LENGTH_PPG], avgiBPF[LENGTH_PPG], avgcBPF[LENGTH_PPG], avgrBPF[LENGTH_PPG], avgiDT[LENGTH_PPG], avgcDT[LENGTH_PPG], avgrDT[LENGTH_PPG], avgiICA[LENGTH_PPG], avgcICA[LENGTH_PPG], avgrICA[LENGTH_PPG];
                    
                    int ct = 0;
                    for (int i=0; i<LENGTH_PPG; i++)
                    {
                        if((isnan(ppgArray[i].avgiVal) || (ppgArray[i].avgiVal < 0)) || (isnan(ppgArray[i].avgcVal) || (ppgArray[i].avgcVal < 0)))
                        {
                            ct++;
                            //vita::start_flg = 1;
                            //break;
                        }
                        else
                        {
                            xs[i] = std::chrono::duration_cast<std::chrono::duration<double>>(ppgArray[i].timeVal - ppgArray[0].timeVal).count();
                            avgi[i] = ppgArray[i].avgiVal;
                            avgc[i] = ppgArray[i].avgcVal;
                            avgr[i] = ppgArray[i].avgrVal;
                            avgiBPF[i] = 0;
                            avgcBPF[i] = 0;
                            avgrBPF[i] = 0;
                            avgiDT[i] = 0;
                            avgcDT[i] = 0;
                            avgrDT[i] = 0;
                            avgiICA[i] = 0;
                            avgcICA[i] = 0;
                            avgrICA[i] = 0;
                        }
                    }
                    
                    if(ct == 0)
                    {
                        vita::filter(FiltOrd, DenC, NumC, 5, avgi, avgiBPF);
                        vita::filter(FiltOrd, DenC, NumC, 5, avgc, avgcBPF);
                        vita::filter(FiltOrd, DenC, NumC, 5, avgr, avgrBPF);
                    
                        vita::detrend(avgiBPF, avgiDT);
                        vita::detrend(avgcBPF, avgcDT);
                        vita::detrend(avgrBPF, avgrDT);
                        
                        Mat1d s_avgi = Mat(LENGTH_PPG, 1, CV_64FC1, avgiDT).clone();
                        Mat1d s_avgc = Mat(LENGTH_PPG, 1, CV_64FC1, avgcDT).clone();

                        Mat s_avgi_pca = Mat(s_avgi.rows, 1, CV_64FC1);
                        Mat pc_avgi = Mat(s_avgi.rows, s_avgi.cols, CV_64FC1);
                        Mat s_avgc_pca = Mat(s_avgc.rows, 1, CV_64FC1);
                        Mat pc_avgc = Mat(s_avgc.rows, s_avgc.cols, CV_64FC1);
            
                        int low = (int)(s_avgi.rows * LOW_BPM / SEC_PER_MIN / vita::fps);
                        int high = (int)(s_avgi.rows * HIGH_BPM / SEC_PER_MIN / vita::fps) + 1;
            
                        vita::pcaComponent(s_avgi, s_avgi_pca, pc_avgi, low, high);
                        vita::pcaComponent(s_avgc, s_avgc_pca, pc_avgc, low, high);

                        Mat s_avgi_mav = Mat(s_avgi.rows, 1, CV_64FC1);
                        Mat s_avgc_mav = Mat(s_avgc.rows, 1, CV_64FC1);
            
                        vita::movingAverage(s_avgi_pca, s_avgi_mav, 3);
                        vita::movingAverage(s_avgc_pca, s_avgc_mav, 3);
                
                        for (int i = 0; i < LENGTH_PPG; i++)
                        {
                            avgiICA[i] = s_avgi_mav.at<double>(i);
                            avgcICA[i] = s_avgc_mav.at<double>(i);
                        }
                    
                        vita::extract_vitals(xs, avgcICA, avgiICA);
                        cout<<"- FPS:"<<vita::fps<<" - HR:" << vita::HR <<" - RR:" << vita::RR <<" - SPO2:"<< vita::SPO2 << " - SBP:"<< vita::SBP <<" - DBP:"<<vita::DBP<<" -"<<endl;
                        pthread_mutex_lock( &mutex_img );
                        vita::bottom_data = cv::Mat(cv::Size(vita::spd_image.cols,200), CV_8UC3, cv::Scalar(0,0,0));
                        std::ostringstream strStream1, strStream2;
                        strStream1 << "- HR:"<<vita::HR<<" (bpm) - RR:"<<vita::RR<<" - SPO2:"<<vita::SPO2<<" (%)";
                        strStream2 << "- BP:"<<vita::SBP<<"/"<<vita::DBP<<" (mm Hg)";
                        std::string cStr1 = strStream1.str();
                        std::string cStr2 = strStream2.str();
                        cv::putText(vita::bottom_data, cStr1, cv::Point(50, vita::bottom_data.rows/2 - 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255,255,255), 1);
                        cv::putText(vita::bottom_data, cStr2, cv::Point(50, vita::bottom_data.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255,255,255), 1);
                        //vita::flg = true;
                        pthread_mutex_unlock( &mutex_img );
                        //break;
                    }
                    else
                    {
                        pthread_mutex_lock( &mutex_add_array );
                        PPG_params _ppgVal = PPG_params{std::chrono::high_resolution_clock::now(),-1,-1,-1};
                        for(int i=0; i<LENGTH_PPG; i++)
                        {
                            ppgArray[i] = _ppgVal;
                        }
                        vita::count = 0;
                        pthread_mutex_unlock( &mutex_add_array );
                    }
                }
            }
        }
        
        return (void*)-1;
    }
    
    
    
};

#endif /* vita_hpp */
