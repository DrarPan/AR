#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <artoolkit/AR/param.h>
#include <artoolkit/AR/ar.h>
#include <artoolkit/AR/video.h>
#include <cv_bridge/cv_bridge.h>
#include <string>

using namespace std;
using namespace cv;

void XY2uv(Mat P, double marker_trans[3][4],double XX,double YY, double& u,double & v);
void uv2XY(Mat P, double marker_trans[3][4],double u,double v,double& XX, double& YY);

int bmin(vector<int> uv,int minvalue){
    int minv=1000;
    for(int i=0;i<uv.size();i++)
       if(uv[i]<minv)
           minv=uv[i];
    if(minv<minvalue)
        minv=minvalue;
    return minv;
}

int bmax(vector<int> uv,int maxvalue){
    int maxv=0;
    for(int i=0;i<uv.size();i++)
       if(uv[i]>maxv)
           maxv=uv[i];
    if(maxv>maxvalue)
        maxv=maxvalue;
    return maxv;
}

class ARsingle{
private:
    //AR detection
    bool use_history_;
    int threshold_;
    double cf_threshold_;
    ARUint8* dataPtr_;
    ARMarkerInfo *marker_info_;
    int marker_num_;
    double marker_trans_[3][4];
    //AR initialization
    int imgwidth_,imgheight_;
    Mat P_;
    ARParam cam_param_;

    int patt_id_;
    string pattern_filename_;

    //marker information
    double marker_center_[2];
    double marker_width_;

public:
    ARsingle(){
        use_history_=true;
        threshold_=120;
        cf_threshold_=0.2;
        imgwidth_=640,imgheight_=480;
        P_=(Mat_<double>(3,4)<<538.907715,0.0,320.346012,0.0,
                               0.0,539.350342,238.023699,0.0,
                               0.0,0.0,1.0,0.0);
        cam_param_.xsize=imgwidth_;
        cam_param_.ysize=imgheight_;
        cam_param_.mat[0][0]=P_.at<double>(0,0);
        cam_param_.mat[0][1]=P_.at<double>(0,1);
        cam_param_.mat[0][2]=P_.at<double>(0,2);
        cam_param_.mat[0][3]=P_.at<double>(0,3);
        cam_param_.mat[1][0]=P_.at<double>(1,0);
        cam_param_.mat[1][1]=P_.at<double>(1,1);
        cam_param_.mat[1][2]=P_.at<double>(1,2);
        cam_param_.mat[1][3]=P_.at<double>(1,3);
        cam_param_.mat[2][0]=P_.at<double>(2,0);
        cam_param_.mat[2][1]=P_.at<double>(2,1);
        cam_param_.mat[2][2]=P_.at<double>(2,2);
        cam_param_.mat[2][3]=P_.at<double>(2,3);

        cam_param_.dist_factor[0]=320.225447;
        cam_param_.dist_factor[1]=238.434859;
        cam_param_.dist_factor[2]=0.0;
        cam_param_.dist_factor[3]=1.0;

        arInitCparam(&cam_param_);
        arParamDisp(&cam_param_);

        pattern_filename_="/home/pan/imageData/4x4_9.patt";
        patt_id_=arLoadPatt(pattern_filename_.c_str());

        marker_width_=80.0;
        marker_center_[0]=0.0;
        marker_center_[1]=0.0;

        marker_num_=0;

        for(int i=0;i<3;i++)
            for(int j=0;j<4;j++){
                marker_trans_[i][j]=0;
            }
    }
    bool DetectBestTransform(Mat img, double trans[3][4]){
        dataPtr_ = (ARUint8 *) ((IplImage) img).imageData;
        arDetectMarker(dataPtr_,threshold_,&marker_info_,&marker_num_);
        if(marker_num_>0){
            int bestindex=-1;
            double bestcf=0;

            for(int i=0;i<marker_num_;i++){
                if(marker_info_[i].cf>bestcf){
                    bestcf=marker_info_[i].cf;
                    bestindex=i;
                }
            }
            if(bestcf>cf_threshold_){
                arGetTransMat(&marker_info_[bestindex],marker_center_,marker_width_,marker_trans_);
                for(int i=0;i<3;i++)
                    for(int j=0;j<4;j++)
                        trans[i][j]=marker_trans_[i][j];
                return 1;
            }else return 0;
        }else return 0;
    }

    void XY2uv(double XX,double YY, double& u,double & v){
        u=int((marker_trans_[0][0]*XX+marker_trans_[0][1]*YY+marker_trans_[0][3])/(marker_trans_[2][0]*XX+marker_trans_[2][1]*YY+marker_trans_[2][3])*P_.at<double>(0,0)+P_.at<double>(0,2));
        v=int((marker_trans_[1][0]*XX+marker_trans_[1][1]*YY+marker_trans_[1][3])/(marker_trans_[2][0]*XX+marker_trans_[2][1]*YY+marker_trans_[2][3])*P_.at<double>(1,1)+P_.at<double>(1,2));
    }

    void uv2XY(double u,double v,double& XX, double& YY){
        double uu=(u-P_.at<double>(0,2))/P_.at<double>(0,0);
        double vv=(v-P_.at<double>(1,2))/P_.at<double>(1,1);

        double a=uu*marker_trans_[1][0]-vv*marker_trans_[0][0];
        double b=vv*marker_trans_[0][1]-uu*marker_trans_[1][1];
        double c=vv*marker_trans_[0][3]-uu*marker_trans_[1][3];

        YY=((marker_trans_[2][0]*c+marker_trans_[2][3]*a)*uu-marker_trans_[0][0]*c-marker_trans_[0][3]*a)/(marker_trans_[0][0]*b+marker_trans_[0][1]*a-uu*(marker_trans_[2][0]*b+marker_trans_[2][1]*a));
        XX=(b*YY+c)/a;
    }

    bool attachPattern(Mat& markerimg,Mat& attachedimg, double patwidth, double patheight){
        double xtopleft=-patwidth/2,ytopleft=patheight/2;
        double xbottomleft=-patwidth/2,ybottomleft=-patheight/2;
        double xbottomright=patwidth/2,ybottomright=-patheight/2;
        double xtopright=patwidth/2,ytopright=patheight/2;

        double utopleft,vtopleft;
        double ubottomleft,vbottomleft;
        double ubottomright,vbottomright;
        double utopright,vtopright;

        double scale=patwidth/attachedimg.size[0];

        XY2uv(xtopleft,ytopleft,utopleft,vtopleft);
        XY2uv(xbottomleft,ybottomleft,ubottomleft,vbottomleft);
        XY2uv(xbottomright,ybottomright,ubottomright,vbottomright);
        XY2uv(xtopright,ytopright,utopright,vtopright);

        vector<int> vecu;
        vector<int> vecv;

        vecu.push_back(int(utopleft));
        vecu.push_back(int(ubottomleft));
        vecu.push_back(int(ubottomright));
        vecu.push_back(int(utopright));

        vecv.push_back(int(vtopleft));
        vecv.push_back(int(vbottomleft));
        vecv.push_back(int(vbottomright));
        vecv.push_back(int(vtopright));

        int umax=bmax(vecu,imgwidth_);
        int umin=bmin(vecu,0);
        int vmax=bmax(vecv,imgheight_);
        int vmin=bmin(vecv,0);
        for(int i=umin;i<umax;i++){
            for(int j=vmin;j<vmax;j++){
               double x_,y_;
               uv2XY(i,j,x_,y_);
               //cout<<"x="<<x_<<"y="<<y_<<endl;
               if(x_>-patwidth/2 && x_<patwidth/2 && y_>-patheight/2 && y_<patheight/2 && attachedimg.at<uchar>(int((patheight/2-y_)/scale),int((x_+patwidth/2)/scale)*3)<200){
                   markerimg.at<uchar>(j,3*i)=attachedimg.at<uchar>(int((patheight/2-y_)/scale),int((x_+patwidth/2)/scale)*3);
                   markerimg.at<uchar>(j,3*i+1)=attachedimg.at<uchar>(int((patheight/2-y_)/scale),int((x_+patwidth/2)/scale)*3+1);
                   markerimg.at<uchar>(j,3*i+2)=attachedimg.at<uchar>(int((patheight/2-y_)/scale),int((x_+patwidth/2)/scale)*3+2);
               } 
            }
        }

        return 1;
    }

};

Mat rotateFu(Mat img,int angle){
    Mat fuRotateMat;
    Mat rotateImgFu;
    Point2f fuRotateCenter(img.size().width/2,img.size().height/2);

    fuRotateMat=getRotationMatrix2D(fuRotateCenter,angle,1);
    warpAffine(img,rotateImgFu,fuRotateMat,img.size(),INTER_LINEAR,BORDER_CONSTANT,Scalar(255,255,255));
    return rotateImgFu;
}

int main(){
    ARsingle arSingle;
    Mat imgARMarker;
    Mat imgFu=imread("/home/pan/imageData/Fu5.png");
    Mat rotateImgFu;
    double trans[3][4];
    VideoWriter writer("/home/pan/blessVideo.avi",CV_FOURCC('M','J','P','G'),20.0,Size(640,480));
    for(int i=45;i<225;i++){
        stringstream ss;
        ss<<"/home/pan/captureData/recordimg"<<i<<".png";
        rotateImgFu=rotateFu(imgFu,(i-5)*3);
        imgARMarker=imread(ss.str());
        bool gotTrans = arSingle.DetectBestTransform(imgARMarker,trans);
        arSingle.attachPattern(imgARMarker,rotateImgFu,250,250);
        if(gotTrans){
            //resize(imgARMarker,imgARMarker,Size(320,240),0,0,INTER_CUBIC);
            imshow("image",imgARMarker);
            writer<<imgARMarker;
            cvWaitKey(40);
        }
    }
    writer.release();

}

