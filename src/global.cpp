/*
 * global.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#include"global.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <errno.h>
#include <string.h>
float alpha = 0.03;
float * weights, * biases;
float * gradWeights, * gradBiases;
float * Wcoef_AdaGrad, * Bcoef_AdaGrad;
float eta;
int num_weights, num_biases;
int num_train, num_CV, num_test;

int num_label = 104;

//pthread_t threads[NUM_THREADS];
//pthread_mutex_t mutex_param;


void RandomInitParam(){
	int NUM_W= 1000, NUM_B = 1000;

	weights = new float[NUM_W];
	biases = new float[NUM_B];

	float bound = 1;
	float bound2 = bound * 2;

	for(int i = 0; i < NUM_W; i++){
		weights[i] = i%10 * bound2 - bound;
	}
	for(int i = 0; i < NUM_B; i++){
		biases[i] = i % 10 * bound2 - bound;
	}
	cout << "INFO: Weights random initialized." << endl;
}


void ReadParam(const char * filepath){
	FILE *infile = fopen(filepath, "rb");
	cout<< "\nLoad parameters from "<< filepath<<flush;

	if( infile == NULL){
		cout << "ERR: Can't load parameters; file name : " << filepath << endl;
		exit(1);
	}



	fread((void *) &num_weights, sizeof(int), 1, infile);
	fread((void *) &num_biases, sizeof(int), 1, infile);
	cout <<"\n" <<strerror(errno)<<"\n";cout << flush;
	cout << "num weights "<<num_weights << " num biases " << num_biases << endl<<flush;

	weights = (float *) malloc( sizeof(float) * num_weights);
	biases  = (float *) malloc( sizeof(float) * num_biases);

	gradWeights = (float *) malloc( sizeof(float) * num_weights);
	gradBiases = (float *) malloc( sizeof(float) * num_biases);

	Wcoef_AdaGrad = (float *) malloc( sizeof(float) * num_weights);
	Bcoef_AdaGrad = (float *) malloc( sizeof(float) * num_biases);


	memset( (void *) gradWeights, 0, sizeof(float) * num_weights);
	memset( (void *) gradBiases,  0, sizeof(float) * num_biases);

	fread((void *)weights, sizeof(float), num_weights, infile);
	fread((void *)biases,  sizeof(float), num_biases, infile);
	//for(int i = 0;i<150;i++)
	//cout<<biases[5]<<endl;
//	cout << weights[0] << " " << weights[1] << " " << weights[num_weights-1] << endl;
//	cout << biases[0]  << " " << biases[1]  << " " << biases[num_biases-1]  << endl;
	return;
}
void ReadParamTBCNN(const char * filepath){
	FILE *infile = fopen(filepath, "rb");

	if( infile == NULL){
		cout << "ERR: Can't load parameters; file name : " << filepath << endl;
		exit(1);
	}



	fread((void *) &num_weights, sizeof(int), 1, infile);
	fread((void *) &num_biases, sizeof(int), 1, infile);

	cout << num_weights << " " << num_biases << endl;

	//weights = (float *) malloc( sizeof(float) * num_weights);
	//biases  = (float *) malloc( sizeof(float) * num_biases);

	fread((void *)weights, sizeof(float), num_weights, infile);
	fread((void *)biases,  sizeof(float), num_biases, infile);
	//for(int i = 0;i<150;i++)
	//cout<<biases[5]<<endl;
//	cout << weights[0] << " " << weights[1] << " " << weights[num_weights-1] << endl;
//	cout << biases[0]  << " " << biases[1]  << " " << biases[num_biases-1]  << endl;
	return;
}
void SaveParam(const char * filepath){
	FILE * outfile = fopen(filepath, "wb");
	if (outfile == NULL){
		cout << "ERR: Can't save parameters; file name: " << filepath << endl;
	}

	fwrite( &num_weights, sizeof(int), 1, outfile);
	fwrite( &num_biases,  sizeof(int), 1, outfile);

	cout<< num_weights << " " << num_biases << endl;
	fwrite( weights, sizeof(float), num_weights, outfile);
	fwrite( biases,  sizeof(float), num_biases,  outfile);
	fflush( outfile );
	return;
}

void saveTBCNNParam2(int epoch, float alpha, bool n_miniGDchange)
{
	cout<<"save param epoch = "<<epoch;
	char s[10];

	string strw = "/home/seke/lyx/TBCNN_Csmallnet/TBCNNParam/AllParam_";
	sprintf(s, "%d", epoch);
	string tmp;
	tmp = s;
	strw = strw+s+"_";
	int a;
	a = (int)(alpha*1000);
	sprintf(s,"%d",a);
	strw = strw+s;
	if(n_miniGDchange)strw = strw+"_alchange";	
	strw = strw+".txt";
	cout<<" in  "<<strw<<endl;
	const char * fweights = strw.c_str();
	SaveParam(fweights);

}

void readTBCNNParam2(int epoch,float alpha,bool n_miniGDchange)
{
	cout<<"save param epoch = "<<epoch;
	char s[10];

	string strw = "/home/seke/lyx/TBCNN_Csmallnet/TBCNNParam/AllParam_";
	sprintf(s, "%d", epoch);
	string tmp;
	tmp = s;
	strw = strw+s+"_";
	int a;
	a = (int)(alpha*1000);
	sprintf(s,"%d",a);
	strw = strw+s;
	if(n_miniGDchange)strw = strw+"_alchange";	
	strw = strw+".txt";
	cout<<" in  "<<strw<<endl;	
	const char * fweights = strw.c_str();
	ReadParamTBCNN(fweights);

}
