#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;

static void help()
{
	printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
		"Using the SURF desriptor:\n"
		"\n"
		"Usage:\n matcher_simple <image1> <image2>\n");
}

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		help();
		return -1;
	}

	Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	if(img1.empty() || img2.empty())
	{
		printf("Can't read one of the images\n");
		return -1;
	}

	// detecting keypoints
	SurfFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);

	// computing descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// drawing the results
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
	imshow("original matches", img_matches);
	waitKey();



	// 分配空间
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	// 把Keypoint转换为Mat
	Point2f pt;
	for (int i=0; i<ptCount; i++)
	{
		pt = keypoints1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	cout<<p1<<endl<<p2;

	// 用RANSAC方法计算F
	Mat m_Fundamental;
	// 上面这个变量是基本矩阵
	vector<uchar> m_RANSACStatus;
	// 上面这个变量已经定义过，用于存储RANSAC后每个点的状态
	m_Fundamental =  findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

	for (int i=0; i<m_RANSACStatus.size(); i++)
	{
		cout<<m_RANSACStatus[i]<<endl;
	}


	// 计算野点个数
	int OutlinerCount = 0;
	for (int i=0; i<ptCount; i++)
	{
		if (m_RANSACStatus[i] == 0) // 状态为0表示野点
		{
			OutlinerCount++;
		}
	}

	// 计算内点
	vector<Point2f> m_LeftInlier;
	vector<Point2f> m_RightInlier;
	vector<DMatch> m_InlierMatches;
	// 上面三个变量用于保存内点和匹配关系
	int InlinerCount = ptCount - OutlinerCount;
	m_InlierMatches.resize(InlinerCount);
	m_LeftInlier.resize(InlinerCount);
	m_RightInlier.resize(InlinerCount);
	InlinerCount = 0;
	for (int i=0; i<ptCount; i++)
	{
		if (m_RANSACStatus[i] != 0)
		{
			m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
			m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
			m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
			m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
			m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
			m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
			InlinerCount++;
		}
	}

	// 把内点转换为drawMatches可以使用的格式
	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(m_LeftInlier, key1);
	KeyPoint::convert(m_RightInlier, key2);

	// 显示计算F过后的内点匹配
	/*Mat m_matLeftImage;
	Mat m_matRightImage;*/
	
	// drawing the results
	//namedWindow("ransac", 2);
	Mat OutImage;
	drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
	cout<<OutImage.size();
	imshow("updated matches", OutImage);
	waitKey();
	
	return 0;
}
