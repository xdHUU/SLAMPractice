#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
   double aa = 1.0, ba = 2.0, ca = 3.0;//真实参数值
   double ae = 2.0, be = -1.0, ce = 5.0;//估计参数值
   int N = 100;  //数据点
   double w_sigma = 1.0;
   double inv_sigma = 1.0 / w_sigma; //set 1/Sigma = W^{-1}
   cv::RNG rng;
   
   //prepare data
   vector<double> x, y;
   for(int i = 0; i < N; i++){
	double temp = i / 100.0;
   	x.push_back(temp);
   	y.push_back(exp(aa * temp * temp + ba * temp + ca) + rng.gaussian(w_sigma));
   }
   
   //Gauss-Newton
   int iteration = 100; //time
   double cost = 0, lastCost = 0; // 本次迭代的cost和上一次迭代的cost
   
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   for(int i = 0; i < iteration; i++) {
   	Matrix3d H = Matrix3d::Zero(); // H = J^T * W^{-1} * J
   	Vector3d g = Vector3d::Zero(); // g = -J^T * W^{-1} * w
   	cost = 0;
   
   	for(int j = 0; j < N; j++){
		double xi = x[j], yi = y[j];
		double error = yi - exp(ae * xi * xi + be * xi + ce);
		Vector3d J;
      		J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
      		J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
      		J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc
      		
      		H += inv_sigma * J * J.transpose();
     		g += -inv_sigma * error * J;
     		
     		cost += error * error;
     	}
     	
     	Vector3d dx = H.ldlt().solve(g);
     	if(isnan(dx[0])) {
     		cout<<"result is nan!"<<endl;
     		break;
     	}
     	
     	if(i > 0 && cost >= lastCost){
     		cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
      		break;
    	}//notice that i > 0
     	
     	ae += dx[0];
     	be += dx[1];
     	ce += dx[2];
     	
     	lastCost = cost;
     	
     	cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
  }
  
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
  return 0;
}
   
   
   
   
   
   
   
   
   
   
   
   
   
