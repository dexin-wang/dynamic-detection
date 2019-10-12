#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "iostream"

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace xfeatures2d;

class Func
{
public:
	Func() {};
	~Func() {};
	//ʹ��˵��
	void attention();
	//����Ƶ�ļ�
	void readVideo(String &videoFile);
	//��frame
	void readFrame();
	//SURF���
	void surfDetect();
	//Knnƥ��
	void BfMatch();
	//����Ӧ����˳�
	void filtForePts();
	//���ñ�������б�������
	void makeBGD();
	//ǰ����ȡ��֡�
	void getFGD();	
	//���ƶ�̬Ŀ������
	void drawContours();	

	//ʶ��̬Ŀ��, ����ֵΪ�����ǩ
	vector<String> recogMov(const Mat &img);


	//��ʾ�����ݸ���
	void showAndUpdate();
	//�������
	int readKey();

	//��ʾ��굱ǰλ��
	void onMouse(int event, int x, int y, int flags, void* ustc);

	//��С���˷����������
	Mat myGetAffineTransform(vector<Point2f> src, vector<Point2f> dst);
	//�������в�
	void calcPtsErrs();
	//��ȡ�߽Ǵ�������
	void getCornerPts(double dx, double dy);
	//����߽ǵ��׼��
	double calPtsS();
	//С�����˳�
	void smallTargetFilte(Mat &img, int thresh);
	//Ŀ����ɢ����鲢
	void combinTarget(Mat &img_rgb, Mat &img_abs);
	//BGRתHSI
	void BGR2HSI(vector<double> &bgr, vector<double> &hsi);
	//������������
	double pts2fDist(Point2f &pts1, Point2f &pts2);
	//�Ƚ���ɫ��������
	double colorDist(vector<double> &hsiMsg1, vector<double> &hsiMsg2);
	//�����pts1��pts2���ƶ�����
	double tgtMovDirec(Point2f &pts1, Point2f &pts2);
	//���۲�ͬ������ƶ�һ����
	double getmovOffset(vector<double> targetsMovMsg1, vector<double> targetsMovMsg2);

private:
	RNG rng;

	const float minRatio = 0.5f;						//BBF��������
	const int L = 20;									//����в�ȼ�
	//const int m = 5;									//����в�ȼ���ֵ

	VideoCapture capture;

	int width;			//��Ƶ֡��
	int height;			//��Ƶ֡��

	Ptr<SURF> surfDetector;// = SURF::create(1000);

	Mat prev_frame, prev_gray;							//ǰһ֡��ͼ���Ҷ�ͼ
	Mat prev_frame_tranf, prev_gray_tranf;				//ǰһ֡����������Ĳ�ͼ���Ҷ�ͼ
	Mat cur_frame, cur_gray;							//��ǰ֡
	Mat differ;											//֡����
	Mat differ_dst;										//֡����
	Mat frame_contours;									//cur_frame�ĸ��ƣ����ڻ�������
	Mat frame_features;									//cur_frame�ĸ��ƣ����ڻ���������

	vector<KeyPoint> Pts[2];							//surf����������
	Mat desc[2];										//surf������

	FlannBasedMatcher matcher;							//ƥ��ģ��
	vector<vector<DMatch>> knnMatches[2];				//��ʼƥ����

	vector<DMatch> BBFmatches;							//BBF�������match
	vector<Point> BBFpts[2];							//BBF����ƥ���ԣ�ǰһ֡-����ǰ֡��
	vector<Point> BBFpts_rev[2];						//BBF����ƥ���ԣ���ǰ֡-��ǰһ֡��
	vector<Point2f> pts_opti[2];						//�Գ�Լ��֮��ĵ��
	vector<Point2f> pts_corner[2];						//�Ľǵĵ��
	vector<Point2f> pts_BGD[2];							//����Ӧ����˳���ı�����
	vector<Point2f> pts_FGD[2];							//����Ӧ����˳����ǰ����

	//char* mouseLoc;

	Mat H;												//�������������

	vector<float> ptsErrs;								//����в�
	vector<int>   ptsErrs_num;							//����в�������� n
	vector<float> ptsErrs_prob;							//����в���ȼ�����
	float PtsBGD_prob;									//�����㼯���ʺ�
	float PtsFGD_prob;									//ǰ���㼯���ʺ�
	float PtsBGD_aver;									//�����㼯��ֵ
	float PtsFGD_aver;									//ǰ���㼯��ֵ
	int th;												//����в���ֵ

	Mat kernal3 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));		//��̬ѧ����
	Mat kernal5 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));		//��̬ѧ����

	vector<vector<Point>> movContours;					//��̬Ŀ������

	const vector<String> labels = { "background", "aeroplane", "bicycle",					//�����ǩ
						"bird", "boat", "bottle", "bus", "car",
						"cat", "chair", "cow", "diningtable",
						"dog", "horse", "motorbike", "person",
						"pottedplant", "sheep", "sofa", "train",
						"tvmonitor" };	
	const String SSD_model_file = "D:/develop_software/opencv_3.3.0/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.caffemodel";	//SSD����ģ��
	const String SSD_txt_file = "D:/develop_software/opencv_3.3.0/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.prototxt";		//SSD�����ļ�
	//Net net = readNetFromCaffe(SSD_txt_file, SSD_model_file);
	const float meanVal = 127.5;
	const float scaleFactor = 0.00783f;
	int bolbWidth = 300;
	int bolbHeight = 300;
	const float confidence_threshold = 0.4;					//SSD������ֵ

	/*********** Ŀ������Ԥ���� ************/
	Mat masks;					//������
	int border[5];				//���浱ǰ���ص�� �� �� ���� �� ���� �������ֵ

	/**************** ��ɢ����鲢 ***************/
	vector<Point2f> contour_center[2];		//��¼������֡ÿ�����������
	vector<vector<double>> hsiMsg[2];		//��¼������֡��������HSI��ɫ��Ϣ
	Mat prev_con;		//��¼ǰһ֡��ͼ
};