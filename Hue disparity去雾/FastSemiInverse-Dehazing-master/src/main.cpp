/*
*   A fast semi-inverse approach to detect and
*   remove the haze from a single image
*/
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <vector>
#include <queue>
#include <limits>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<time.h>


//#define DEBUG
//#define DEBUG_DIFFERENCE
//#define DEBUG_ALE

using namespace std;
using namespace cv;

//start of utilities
//minMaxLoc was acting the magot, kept giving assertion errors
void maxLocs(const Mat& src, queue<Point>& dst, size_t size)
{
    uchar maxValue = numeric_limits<uchar>::min();
    uchar* srcData = reinterpret_cast<uchar*>(src.data);

    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            //cout<<i<<","<<j<<endl;
			//此处有问题若srcData[0,0]==255，则dst里面始终只有一个255
            if(srcData[i*src.rows + j] > maxValue)
            {
                maxValue = srcData[i*src.rows + j];

                dst.push(Point(j, i));

                // pop the smaller one off the end if we reach the size threshold.
                if(dst.size() > size)
                {
                    dst.pop();
                }
            }
        }
    }
}

double* getRange(cv::Mat& input)
{
    /*
        Max and min value of matrix
        More robust
    */
    double *vals = new double[2];
    cv::Mat gray=input.clone();
    GaussianBlur(gray,gray,Size(5,5),1,1,0);
    minMaxLoc(gray,&vals[1],&vals[2]);
    return vals;
}

cv::Point getMax(cv::Mat& input,cv::Mat mask)
{
    /*
        Max and min value of matrix
        More robust with added gaussian blurring
        only search the area of the mask (area of haze)
    */
    mask.convertTo(mask,CV_8U);
    cv::Mat gray;
    cvtColor(input,gray,CV_BGR2GRAY);
    GaussianBlur(gray,gray,Size(11,11),1,1,0);
    double *vals = new double[2];
    cv::Point min,max;
    minMaxLoc(gray,&vals[1],&vals[2],&min,&max,mask);
    return max;
}



template <class ForwardIterator,class Generator>
void generate(ForwardIterator first,ForwardIterator last,Generator g)
{
    while(first!=last)
    {
        *first++ = g();
    }
}

struct Generator
{
    Generator():ci(0.0){}
    Generator(float c_init):c(c_init){}
    float operator()(){ci+=c;return ci;}
    float operator()(float c_init){c=c_init;return c;}
    float ci=0.0;
    float c=0.0;
};

//end of Utilities
cv::Mat semiInverse(cv::Mat& image)
{
    /*
    *   Create the semi inverse of the input image
    */
    image.convertTo(image,CV_32F);
    cv::Mat output = cv::Mat(image.rows,image.cols,image.type());
    std::vector<cv::Mat> channels;
    split(image,channels);
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            float b1=channels[0].at<float>(i,j),
                    g1=channels[1].at<float>(i,j),
                    r1=channels[2].at<float>(i,j);
            float b2=saturate_cast<float>(255-channels[0].at<float>(i,j)),
                    g2=saturate_cast<float>(255-channels[1].at<float>(i,j)),
                    r2=saturate_cast<float>(255-channels[2].at<float>(i,j));
            output.at<Vec3f>(i,j)[0] = max(b1,b2);
            output.at<Vec3f>(i,j)[1] = max(g1,g2);
            output.at<Vec3f>(i,j)[2] = max(r1,r2);
        }
    }
    cv::normalize(output, output, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
    cv::normalize(image, image, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
    return output;
}

cv::Mat semiIncrease(cv::Mat inverse,float xi)
{
    /*
    *   Get the semi inverse of the image, only add xi
    *   to the semi inverse component, chnages the mask area
    */
    inverse.convertTo(inverse,CV_32F);
    cv::Mat output = cv::Mat(inverse.rows,inverse.cols,inverse.type());
    for(int i=0;i<inverse.rows;i++)
    {
        for(int j=0;j<inverse.cols;j++)
        {
            output.at<Vec3f>(i,j)[0] = max((float)inverse.at<Vec3f>(i,j)[0],xi*(float)(255.0-inverse.at<Vec3f>(i,j)[0]));
            output.at<Vec3f>(i,j)[1] = max((float)inverse.at<Vec3f>(i,j)[1],xi*(float)(255.0-inverse.at<Vec3f>(i,j)[1]));
            output.at<Vec3f>(i,j)[2] = max((float)inverse.at<Vec3f>(i,j)[2],xi*(float)(255.0-inverse.at<Vec3f>(i,j)[2]));
        }
    }
    cv::normalize(output, output, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
    cv::normalize(inverse, inverse, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
    return output;
}

cv::Mat haze_difference(cv::Mat& image,cv::Mat& inverse, int rho)
{
    /**
    *   Create a binary mask of the haze areas
    **/
    cv::Mat hsv1,hsv2;
    vector<cv::Mat> channels(3),channels2(3);

    cvtColor(image,hsv1,CV_BGR2HSV);
    cvtColor(inverse,hsv2,CV_BGR2HSV);

    split(hsv1,channels);
    split(hsv2,channels2);

    cv::Mat h1=Mat::zeros(image.rows,image.cols,image.type());
    cv::Mat h2=Mat::zeros(image.rows,image.cols,image.type());
    cv::Mat diff=Mat::zeros(image.rows,image.cols,image.type());
    
    #ifdef DEBUG_DIFFERENCE
        //cv::normalize(channels[0], channels[0], 0, 255, NORM_MINMAX, CV_32FC3);
        //cv::normalize(channels2[0], channels2[0], 0, 255, NORM_MINMAX, CV_32FC3);
        imwrite("h1.jpg",channels[0]);
        imwrite("h2.jpg",channels2[0]);
    #endif

    //subtract(h2,h1,diff,noArray(),CV_32FC1);
    absdiff(channels2[0],channels[0],diff);//get the difference between the H channels
    threshold( diff, diff, rho, 255, 1);//use the inverse binary threshold, the low value is anything below rho
    return diff;
}

float detect_haze(cv::Mat difference,int rho)
{
    /*
        return percentage haze in image, uses the mask
    */
    int count=0;
    for(int i=0;i<difference.rows;i++)
    {
        for(int j=0;j<difference.cols;j++)
        {
            if(difference.at<float>(i,j) < rho)
            {
                count++;
            }
        }
    }

    return (float)count/(difference.rows*difference.cols)*100;
}

cv::Point airlight_estimation(cv::Mat& input,cv::Mat& mask)
{
    /*
        get the brightest pixel value and return it
    */
    cv::Point max = getMax(input,mask);

    #ifdef DEBUG_ALE
        Vec3f ale = input.at<cv::Vec3f>(max);
        cout<<max<<endl;
        cout<<"ale:"<<ale<<endl;
        cv::Mat dst=Mat::ones(input.rows,input.cols,input.type());
        cv::Mat temp=input.clone();
        cv::normalize(temp, temp, 0, 255, NORM_MINMAX, CV_32FC3);
        circle(temp,max,25,(255,0,0),2);
        for(int i=0;i<dst.rows;i++)
        {
            for(int j=0;j<dst.cols;j++){dst.at<cv::Vec3f>(i,j)=ale;}
        }
        imwrite("ale.jpg",dst);
        imwrite("ale_location.jpg",temp);
    #endif
    return max;
}

cv::Mat transmissionMap(cv::Mat& input,cv::Point max)
{
    double w=0.75;
    float ale= input.at<float>(max);
    Mat transmission=Mat::zeros(input.rows,input.cols,input.type());
    for(int i=0;i<transmission.rows;i++)
    {
        for(int j=0;j<transmission.cols;j++)
        {
            transmission.at<Vec3f>(i,j)=(1-w*input.at<float>(i,j)/ale)*255.0;
        }
    }
    cv::normalize(transmission, transmission, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
    return transmission;
}

void AlphaBlend(const Mat& imgFore, Mat& imgDst, const Mat& alpha)
{
    vector<Mat> vAlpha;
    Mat imgAlpha3;
    for(int i = 0; i < 3; i++) {vAlpha.push_back(alpha);}
    merge(vAlpha,imgAlpha3);

    Mat blend = imgFore.mul(imgAlpha3,1.0/255) +
                imgDst.mul(Scalar::all(255)-imgAlpha3,1.0/255);
    blend.copyTo(imgDst);
}

cv::Mat dehaze(cv::Mat& image,cv::Mat &inverse,cv::Mat& difference,cv::Point ale,int k,int rho,double xi)
{
     cv::Mat output = cv::Mat(image.rows,image.cols,image.type());
     float c = 1.0/k;
     vector<float> ci(k);
     std::generate(ci.begin(),ci.end(),Generator(c));

     vector<cv::Mat> layers;
     vector<cv::Mat> mask_layers;
     vector<cv::Mat> diff_layers;
     Vec3f ale_temp= image.at<cv::Vec3f>(ale);
     //std::cout<<ci.size()<<std::endl;
     for(int i=0;i<ci.size();i++)
     {
         cv::Mat layer=image.clone();
         
         cv::Mat inv;
         std::vector<cv::Mat> channels;
         inv = semiIncrease(layer,xi);
         split(inv,channels);
         //layer-=(ci[i]*ale_temp);
         channels[0] -= (ci[i]*ale_temp);
         channels[1] -= (ci[i]*ale_temp);
         channels[2] -= (ci[i]*ale_temp);
         merge(channels,inv);
             
         xi+=0.10;
         cv::Mat diff = haze_difference(layer,inv,rho);
         ale = airlight_estimation(layer,diff);
         ale_temp= layer.at<float>(ale);
         

         #ifdef DEBUG
             cv::Mat in,d;
             cv::normalize(inv, in, 0, 255, NORM_MINMAX, CV_32FC3);
             cv::normalize(diff, d, 0, 255, NORM_MINMAX, CV_32FC3);
             //cv::normalize(mask, m, 0, 255, NORM_MINMAX, CV_32FC3);
             imwrite("layer_inv_"+to_string(i)+".jpg",in);
             imwrite("difference_"+to_string(i)+".jpg",d);
         #endif
         layers.push_back(inv);
         mask_layers.push_back(diff);
     }
     for(int count=layers.size()-1,opp=0;count>-1;count--,opp++)
     {
        float weight=(float)count/layers.size();
        
        addWeighted(layers[count],ci[count],output,ci[opp],0.0,output);
        AlphaBlend(layers[count],output,mask_layers[count]);
     }
     mask_layers[0] = 255 - mask_layers[0];
     addWeighted(image,ci[layers.size()-1],output,ci[0],0.0,output);
     AlphaBlend(image,output,mask_layers[0]);
     return output;
}

//int main(int argc,char *argv[])
int main()
{
    //inputs
    //cv::Mat input = imread(argv[1]);
	cv::Mat input = imread("C:\\Users\\USER\\Desktop\\dehaze_test\\42.png");
    double rho, k, xi;
	clock_t start, end;
	start = clock();
	rho = 10;
	xi = 0.3;
	k = 5;
    /*if(argc<4)
    {
        rho=10;
        xi=0.3;
        k=5;
    }
    else
    {
        rho = atof(argv[2]);
        xi = atof(argv[3]);
        k = atof(argv[4]);
    }*/
    cout<<rho<<","<<xi<<","<<k<<endl;

    cv::Mat inverse = semiInverse(input);
    
    cv::Mat difference = haze_difference(input,inverse,rho);
    
    float haze = detect_haze(difference,rho);
    
    cv::Point ale = airlight_estimation(input,difference);
    cv::Mat dehazed = dehaze(input,inverse,difference,ale,k,rho,xi);

    cout<<"Haze@:"<<haze<<"%"<<endl;
    
    imwrite("reg_42.jpg",input);
    imwrite("dehazed_42.jpg",dehazed);
	end = clock();
	cout << "The time is " << float(end - start) << endl;
    return 0;
}
