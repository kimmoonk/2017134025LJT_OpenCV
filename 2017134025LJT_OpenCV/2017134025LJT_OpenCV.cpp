﻿#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "opencv/highgui.h"
#include "opencv/cv.h"


#define PI 3.141592 

using namespace std;
using namespace cv;

void hough_line();
void affine_transform();
Mat src;
Point2d srcQuad[4], dstQuad[4];
void hough_circles();

Point g_best_pos = { 0,0 };
double g_best_fitness = 0;
RNG rng(getTickCount());

class Particle
{
public:
    Point pos;
    Point vel;
    Point p_best_pos;

    double W;
    double C1;
    double C2;
    double fitness;
    double p_best_fitness;

    int numberOfDimension;
    Scalar color;


public:
    Particle(Point pos = { 5,5 }, Point vel = { 0,0 }, Point p_best_pos = { 0,0 },
        double w = 1.0, double c1 = 1.4, double c2 = 2.0,
        double fitness = 0, double p_best_fitness = 0, int NOD = 2,
        Scalar color = (0,0,255))
        :
        pos(pos),
        vel(vel),
        p_best_pos(p_best_pos),
        W(w),
        C1(c1),
        C2(c2),
        fitness(fitness),
        p_best_fitness(p_best_fitness),
        numberOfDimension(NOD),
        color(color)
    {}
    auto set_pos(Point pos) { this->pos = pos; };

    auto set_vel(Point vel) { this->vel = vel; };

    auto set_p_best_pos(Point p_best_pos) { this->p_best_pos = p_best_pos; };

    auto set_W(double w) { this->W = w; };

    auto set_C1(double c1) { this->C1 = c1; };

    auto set_C2(double c2) { this->C2 = c2; };

    auto set_fitness(double fitness) { this->fitness = fitness; };

    auto set_p_best_fitness(double p_best_fitness) { this->p_best_fitness = p_best_fitness; };

    auto set_numberOfDimension(int NOD) { this->numberOfDimension = NOD; };

    void update_velocity(Point g_best_pos);
    void update_position();
    void evaluate_fitness();

};

void Particle::update_velocity(Point g_best_pos)
{
    double distance = 10.0;
    Point2d d_personal_vel = { 0,0 };
    Point2d d_social_vel = { 0,0 };

    Point personal_vel = (this->p_best_pos - this->pos);
    d_personal_vel = Point2d(personal_vel);

        
    // 이동벡터량이 10보다 클 경우 이동벡터량을 10으로 고정 
    if (distance < sqrt(pow(personal_vel.x, 2) + pow(personal_vel.y, 2)))
    {
        if (personal_vel.x != 0 || personal_vel.y != 0)
        {
            double p_angle = atan2(personal_vel.y, personal_vel.x);
            d_personal_vel = { distance * cos(p_angle), distance * sin(p_angle) };
        }
    }


    /*if(personal_vel.x != 0 && personal_vel.y != 0){
        personal_vel.x = C1 * rng.uniform(0.0f,1.0f) * double(personal_vel.x) / (sqrt(pow(personal_vel.x, 2) + pow(personal_vel.y, 2)));
        personal_vel.y = C1 * rng.uniform(0.0f, 1.0f) *double(personal_vel.y) / (sqrt(pow(personal_vel.x, 2) + pow(personal_vel.y, 2)));
    }*/

    Point social_vel = (g_best_pos - this->pos);
    d_social_vel = Point2d(social_vel);

    // 이동벡터량이 10보다 클 경우 이동벡터량을 10으로 고정 
    if (distance < sqrt(pow(social_vel.x, 2) + pow(social_vel.y, 2)))
    {
        if (social_vel.x != 0 || social_vel.y != 0) 
        {
            double s_angle = atan2(social_vel.y, social_vel.x);
            d_social_vel = { distance * cos(s_angle), distance * sin(s_angle) };
        }
    }

    //if(social_vel.x != 0 && social_vel.y != 0)
    //{
    //    social_vel.x = C2 * rng.uniform(0.0f, 1.0f) * double(social_vel.x) / (sqrt(pow(social_vel.x, 2) + pow(social_vel.y, 2)));
    //    social_vel.y = C2 * rng.uniform(0.0f, 1.0f) * double(social_vel.y) / (sqrt(pow(social_vel.x, 2) + pow(social_vel.y, 2)));
    //}

    Point2d d_interia = { 0,0 };
    Point interia = vel; 
    d_interia = Point2d(vel);

    if (distance < sqrt(pow(interia.x, 2) + pow(interia.y, 2)))
    {
        if (interia.x != 0 || interia.y != 0) {
            double i_angle = atan2(interia.y, interia.x);
            d_interia = { distance * cos(i_angle), distance * sin(i_angle) };
        }
    }


    vel = W * d_interia + C1 * rng.uniform(0.0f,1.0f) * d_personal_vel + C2 * rng.uniform(0.0f, 1.0f) * d_social_vel;
    //printf("vel : %f %f \n", vel.x , vel.y);

}

void Particle::update_position()
{
    printf("X : %d Y : %d  + X : %d Y : %d\n", pos.x,pos.y, vel.x,vel.y);

    pos = pos + vel;

    if (pos.x > 360) { pos.x = 360; }
    if (pos.x < 0) { pos.x = 0; }
    if (pos.y > 300) { pos.y = 300; }
    if (pos.x < 0) { pos.y = 0; }


    // 맵밖으로 나가는거 잡기
}

void Particle::evaluate_fitness()
{
    double sigma = 10;

    // 뼈대만들고
    double bone = (1 / (sqrt(2 * PI) * sigma)) * exp(-1 / (2 * pow(sigma, 2)));

    // x,y,r 주면 원 = r * exp( pow(x-70,2) + pow(y-70,2) )

    // for 객체수
    // fitness += r * exp( pow(pos.x-x,2) + pow(pos.y-y,2) )
    // fitness = fitness * 뼈대 하면 총 fitness가 나온다 이말이야. 오케이


    //(30,30)이 최고점 
    fitness = 100 * (double)(1 / (sqrt(2 * PI) * sigma) * exp(-(pow(pos.x - 30, 2) + pow(pos.y - 30, 2)) / (2 * pow(sigma, 2))))
        + 150 * (double)(1 / (sqrt(2 * PI) * sigma) * exp(-(pow(pos.x - 70, 2) + pow(pos.y - 70, 2)) / (2 * pow(sigma, 2))));

    printf("fitness is : %f\n", fitness);

    if (fitness > p_best_fitness)
    {
        p_best_pos = pos;
        p_best_fitness = fitness;
    }

}

int main(void)
{
    Mat img;
    img = imread("C:/opencv/images/coins.png", IMREAD_GRAYSCALE);
    //img = imread("C:/opencv/images/circles.png", IMREAD_GRAYSCALE);


    if (img.empty()) {

    }

    Mat blurred;
    blur(img, blurred, Size(3, 3));

    vector<Vec3f> circles;
    HoughCircles(blurred, circles, CV_HOUGH_GRADIENT, 1, 50, 150, 30);

    Mat dst;
    Mat sim;
    cvtColor(img, dst, COLOR_GRAY2BGR);
    cvtColor(img, sim, COLOR_GRAY2BGR);


    for (Vec3f c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        circle(dst, center, radius, Scalar(0, 0, 255), 2);

        printf("X : %d , Y : %d , R : %d \n", cvRound(c[0]), cvRound(c[1]), cvRound(c[2]));
    }

    // Fitness 계산
    double sigma = 10.0;
    double bone = (1 / (sqrt(2 * PI) * sigma));

    int numberofParticle = 100;
    Particle swarm[100];



    for (int i = 0; i < 100; i++)
    {
        swarm[i].pos = { rng.uniform(0,300) , rng.uniform(0,300) };
        swarm[i].p_best_pos = swarm[i].pos;
        swarm[i].color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        printf("x: %d , y: %d\n", swarm[i].pos.x, swarm[i].pos.y);

        circle(sim, swarm[i].pos, 2, Scalar(0, 0, 255), 2);
    }
    //waitKey(0);

    for (int j = 0; j < 100; j++)
    {
        //각 particle check
        for (int i = 0; i < 100; i++)
        {
            double hagong = 0;

            // 해공간 만들기 
            for (Vec3f c : circles) 
            {
                Point center(cvRound(c[0]), cvRound(c[1]));
                int radius = cvRound(c[2]);
                //해공간에서 fitness 측정  
                hagong += (double)1000.0 * radius * exp( 
                    -(
                       ( pow(swarm[i].pos.x - center.x, 2) + pow(swarm[i].pos.y - center.y, 2) ) / (2 * pow(sigma,2))
                      ) 
                );
            }
            
            swarm[i].fitness = (double)( bone * hagong );
            printf("swarm[%d] 의 fitness : %f\n", i, swarm[i].fitness);

            if (swarm[i].fitness > swarm[i].p_best_fitness)
            {
                swarm[i].p_best_pos = swarm[i].pos;
                swarm[i].p_best_fitness = swarm[i].fitness;
            }
            
            if (swarm[i].p_best_fitness > g_best_fitness)
            {
                g_best_fitness = swarm[i].p_best_fitness;
                g_best_pos = swarm[i].p_best_pos;
            }
        }

        printf("global best update %f \n", g_best_fitness);
        printf("global pos = X:%d, Y:%d \n", g_best_pos.x,g_best_pos.y);

        cvtColor(img, sim, COLOR_GRAY2BGR);

        for (int i = 0; i < 100; i++)
        {
            swarm[i].update_velocity(g_best_pos);
            swarm[i].update_position();

            circle(sim, swarm[i].pos, 2, swarm[i].color, 2);

            printf("swarm[%d] 의 X : %d Y : %d \n", i, swarm[i].pos.x, swarm[i].pos.y);

            if (swarm[i].W > 0.4) {
                swarm[i].W -= 0.01;
            }
        }

        imshow("sim", sim);
        waitKey(10);

        printf("%d\n", j);
    }

    circle(dst, g_best_pos, 5, Scalar(0, 0, 255), 2);


    printf("global best update %f \n", g_best_fitness);
    printf("global pos = X:%d, Y:%d \n", g_best_pos.x, g_best_pos.y);

    imshow("image", img);
    imshow("dst", dst);


    waitKey(0);

    return 0;
}


// Mat M = (Mat_<double>(2,3)<< 1,0,150,0,1,100);
//int main(void)
//{
//    Mat myImage;
//    namedWindow("Video Player", WINDOW_AUTOSIZE);
//    VideoCapture cap(0);
//
//    Mat image;
//    Vec3b intensity;
//    Mat gray_image;
//
//    int frame_width = cvRound(cap.get(CV_CAP_PROP_FRAME_WIDTH));
//    int frame_height = cvRound(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
//    double fps = 30;
//    int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
//    VideoWriter video("outputVideo.avi", fourcc, fps, Size(frame_width, frame_height));
//
//    int i = 0;
//    while (cap.isOpened()) {
//        cout << "frame # = " << i << endl;
//        cap >> myImage;
//        //cap >> image;
//        /*gray_image = Mat::zeros(image.rows, image.cols, CV_8UC1);
//
//        for (int i = 0; i < image.rows; i++) {
//            for (int j = 0; j < image.cols; j++) {
//                intensity = image.at<Vec3b>(i, j);
//                int data = (intensity.val[0] + intensity.val[1] + intensity.val[2]) / 3;
//                gray_image.at<uchar>(i, j) = (char)data;
//            }
//        }*/
//
//        video.write(myImage);
//        flip(myImage, myImage, +1);
//        imshow("Video Player", myImage);
//        i = i + 1;
//
//        char c = (char)waitKey(25);
//
//        if (c == 27) {
//            break;
//        }
//
//        else if (c == 99)
//        {
//            string name = "saved_image_" + to_string(i) + ".jpg";
//            imwrite(name, myImage);
//        }
//
//        else if (i > 1000)
//        {
//            break;
//        }
//
//    }
//
//    cap.release();
//    video.release();
//    destroyAllWindows();
//
//    return 0;
//}


void affine_transform()
{
    Mat src;
    src = imread("C:/opencv/image/tekapo.bmp");

    Point2f srcPts[3], dstPts[3];
    srcPts[0] = Point2f(0, 0);
    srcPts[1] = Point2f(src.cols - 1, 0);
    srcPts[2] = Point2f(src.cols - 1, src.rows - 1);

    dstPts[0] = Point2f(50, 50);
    dstPts[1] = Point2f(src.cols - 100, 100);
    dstPts[2] = Point2f(src.cols - 50, src.rows - 50);

    Mat M = getAffineTransform(srcPts,dstPts); 
    Mat M2 = (Mat_<double>(2, 3) << 1, 0, 150, 0, 1, 100); //이동변환

    double mx = 0.3;
    Mat M3 = (Mat_<double>(2, 3) << 1, mx, 0, 0, 1, 0);

    Mat dst;
    //warpAffine(src, dst, M, Size());
    warpAffine(src, dst, M3, Size(cvRound(src.cols + src.rows * mx), src.rows));

    Mat dst1, dst2, dst3, dst4;
    resize(src, dst1, Size(), 4, 4, INTER_NEAREST);
    resize(src, dst2, Size(1920,1280));
    resize(src, dst3, Size(1920, 1280), 0, 0, INTER_CUBIC);
    resize(src, dst4, Size(1920, 1280), 0, 0, INTER_LANCZOS4);


    imshow("src", src);
    imshow("dst1", dst1(Rect(400,500,400,400)));
    imshow("dst2", dst2(Rect(400, 500, 400, 400)));
    imshow("dst3", dst3(Rect(400, 500, 400, 400)));
    imshow("dst4", dst4(Rect(400, 500, 400, 400)));

    waitKey(0);

}
void hough_line()
{
    Mat img;
    img = imread("C:/opencv/image/building.jpg",IMREAD_GRAYSCALE);

    if (img.empty()) {

    }

    Mat edge;
    Canny(img, edge, 50, 150);

    vector<Vec2f> lines;
    HoughLines(edge, lines,1, CV_PI/180,250);

    Mat dst;
    cvtColor(edge, dst, COLOR_GRAY2BGR);



    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        float cos_t = cos(theta), sin_t = sin(theta);
        float x0 = rho * cos_t, y0 = rho * sin_t;
        float alpha = 1000;

        Point pt1(cvRound(x0 - alpha * sin_t), cvRound(y0 + alpha * cos_t));
        Point pt2(cvRound(x0 + alpha * sin_t), cvRound(y0 - alpha * cos_t));
        line(dst, pt1, pt2, Scalar(0, 0, 255), 2);
    }

    imshow("image", img);
    imshow("dst", dst);
   

    waitKey(0);

}

void hough_circles()
{
    Mat img;
    img = imread("C:/opencv/image/coins.png", IMREAD_GRAYSCALE);

    if (img.empty()) {

    }

    Mat blurred;
    blur(img, blurred, Size(3,3));

    vector<Vec3f> circles;
    HoughCircles(blurred, circles, CV_HOUGH_GRADIENT,1,50,150,30);

    Mat dst;
    cvtColor(img, dst, COLOR_GRAY2BGR);

    for (Vec3f c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        circle(dst, center, radius, Scalar(0, 0, 255), 2);

        printf("X : %d , Y : %d , R : %d \n", cvRound(c[0]), cvRound(c[1]), cvRound(c[2]));
    }

    imshow("image", img);
    imshow("dst", dst);


    waitKey(0);

}