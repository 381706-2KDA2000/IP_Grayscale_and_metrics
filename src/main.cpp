#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;


float GetIntensity(Mat src)
{
  float res = 0;
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      res += (bgr[0] + bgr[1] + bgr[2]);
    }
  res = res / (src.rows * src.cols);
  return res;
}

float GetContrast(Mat src)
{
  float res = 0;
  float M = GetIntensity(src);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      res += pow((bgr[0] + bgr[1] + bgr[2]) - M, 2);
    }
  res = sqrt(res / (src.rows * src.cols));
  return res;
}

float GetCov(Mat src1, Mat src2)
{
  if (src1.size != src2.size)
    throw 1;
  float res = 0;
  float M1 = GetIntensity(src1);
  float M2 = GetIntensity(src2);
  for (int y = 0; y < src1.rows; y++)
    for (int x = 0; x < src1.cols; x++)
    {
      Vec3b bgr1 = src1.at<Vec3b>(y, x);
      Vec3b bgr2 = src2.at<Vec3b>(y, x);
      res += (bgr1[0] + bgr1[1] + bgr1[2] - M1)*(bgr2[0] + bgr2[1] + bgr2[2] - M2);
    }
  return res / (src1.rows * src1.cols);
}

float SSIMMetric(Mat src1, Mat src2)
{
  return (2.f * GetIntensity(src1) * GetIntensity(src2))*(2 * GetCov(src1, src2))/((GetIntensity(src1) * GetIntensity(src1) + GetIntensity(src2) * GetIntensity(src2)) * (GetContrast(src1) * GetContrast(src1) + GetContrast(src2) * GetContrast(src2)));
}

Mat AverageGray(Mat src)
{
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = (bgr[0] + bgr[1] + bgr[2]) / 3;
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}

Mat LightnessGray(Mat src)
{
  //Mat res(src.size(), CV_8UC1);
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; ++y)
    for (int x = 0; x < src.cols; ++x)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = (std::max(bgr[0], std::max(bgr[1], bgr[2])) + std::min(bgr[0], std::min(bgr[1], bgr[2]))) / 2;
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}

Mat PhotoshopGray(Mat src)
{
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = bgr[0] * 0.11f + bgr[1] * 0.59f + bgr[2] * 0.3f;
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}

Mat LuminosityGray(Mat src)
{
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = bgr[0] * 0.07f + bgr[1] * 0.72f + bgr[2] * 0.21f;
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}

Mat  ITU_RGray(Mat src)
{
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = bgr[0] * 0.0722f + bgr[1] * 0.7152f + bgr[2] * 0.2126f;
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}

Mat MaxGray(Mat src)
{
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = (std::max(bgr[0], std::max(bgr[1], bgr[2])));
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}

Mat MinGray(Mat src)
{
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = (std::min(bgr[0], std::min(bgr[1], bgr[2])));
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}

Mat NamelessGray(Mat src)
{
  Mat res;
  src.copyTo(res);
  for (int y = 0; y < src.rows; y++)
    for (int x = 0; x < src.cols; x++)
    {
      Vec3b bgr = src.at<Vec3b>(y, x);
      float gray = bgr[0] * 0.148f + bgr[1] * 0.5547f + bgr[2] * 0.2952f;
      res.at<Vec3b>(y, x)[0] = gray;
      res.at<Vec3b>(y, x)[1] = gray;
      res.at<Vec3b>(y, x)[2] = gray;
    }
  return res;
}



int main(int argc, char** argv)
{
  char string[32] = "";
  char string1[32] = "";
  std::vector<Mat> imgs;
  std::vector<char*> names {"original", "average", "lightness", "luminosity", "photoshop", "ITU_R", "max", "min", "nameless" };
  Mat img = imread("C:\\Users\\dimen\\Pictures\\sarcasm.jpg");
  imgs.push_back(img);
  Mat average = AverageGray(img);
  imgs.push_back(average);
  Mat lightness = LightnessGray(img);
  imgs.push_back(lightness);
  Mat luminosity = LuminosityGray(img);
  imgs.push_back(luminosity);
  Mat photoshop = PhotoshopGray(img);
  imgs.push_back(photoshop);
  Mat ITU_R = ITU_RGray(img);
  imgs.push_back(ITU_R);
  Mat max = MaxGray(img);
  imgs.push_back(max);
  Mat min = MinGray(img);
  imgs.push_back(min);
  Mat nameless = NamelessGray(img);
  imgs.push_back(nameless);
  for (int i = 0; i < imgs.size(); i++)
  {
    putText(imgs[i], itoa(GetIntensity(imgs[i]), string, 10), Point(10, 15), FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BGR2RGB);
    putText(imgs[i], itoa(GetContrast(imgs[i]), string1, 10), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BGR2RGB);
    imshow(names[i], imgs[i]);
  }
  waitKey();
  for (int i = 0; i < imgs.size(); i++)
  {
    for (int j = i + 1; j < imgs.size(); j++)
    {
      std::cout << "SSIM for " << names[i] << " and " << names[j] << " = " <<SSIMMetric(imgs[i], imgs[j]) << std::endl;
    }
  }
  waitKey();
  return 0;
}