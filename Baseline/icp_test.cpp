/* 
 * Testbench for icp
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "icp.h"

int main()
{
	int2 viewImageSize = {640, 480};
	float4 viewIntrinsics = {573.71, 574.394, 346.471, 249.031};

	Mat44_cl approxInvPose_val = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
	Mat44_cl scenePose_val = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
	const Mat44_cl* approxInvPose = &approxInvPose_val;
	const Mat44_cl* scenePose = &scenePose_val;

	int2 sceneImageSize = {40, 30};
	float4 sceneIntrinsics = {35.8569, 35.8996, 21.6544, 15.5644};
	float distThresh = 0.01;
	unsigned char shortIteration = 1;
	unsigned char rotationOnly = 1;

	float *depth = (float*) malloc(sizeof(float) * MAX_IMG_X * MAX_IMG_Y);
	float4 *pointsMap = (float4*) malloc(sizeof(float4) * MAX_IMG_X * MAX_IMG_Y);
	float4 *normalsMap = (float4*) malloc(sizeof(float4) * MAX_IMG_X * MAX_IMG_Y);

	printf("Depth\n");
	FILE *fp;
	fp = fopen("depth.txt","r");
	for (int i = 0; i < 307200; i++) {
		float temp;
		fscanf(fp, "%f", &depth[i]);
	}
	fclose(fp);
	printf("\n");

	fp = fopen("C:normalsMap.txt","r");
	for (int i = 0; i < 307200; i++) {
		float4 temp;
		fscanf(fp, "%f, %f, $f, %f", &temp.x, &temp.y, &temp.z, &temp.w);
		normalsMap[i] = temp;
		//printf("normalsMap at %d is %f %f %f %f\n", i, normalsMap[i].x, normalsMap[i].y, normalsMap[i].z, normalsMap[i].w);
	}
	fclose(fp);
	printf("\n");


	printf("pointsMap \n");
	fp = fopen("pointsMap.txt","r");
	for (int i = 0; i < 307200; i++) {
		float4 temp;
		fscanf(fp, "%f, %f, $f, %f", &temp.x, &temp.y, &temp.z, &temp.w);
		pointsMap[i] = temp;
		//printf("pointsMap at %d is %f %f %f %f\n", i, pointsMap[i].x, pointsMap[i].y, pointsMap[i].z, pointsMap[i].w);
	}
	fclose(fp);
	printf("\n");


	// Outputs
	float sumHessian_out[HESSIAN_SIZE];
	float sumNabla_out[6];
	int* noValidPoints_out;
	float* sumF_out;

	// Golden Outputs
	float sumHessian_gold[HESSIAN_SIZE] = {123.474, -33.9865, 36.8201, 18.1851, -9.43707, 12.9704, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	float sumNabla_gold[6] = {2.02879, -0.569651, 0.283277, 1, 0, 0};
	int noValidPoints_gold = 498;
	float sum_F_gold = 0.0572447;


	// dut
	icp (depth, viewImageSize, viewIntrinsics, approxInvPose, scenePose, sceneImageSize, sceneIntrinsics, pointsMap,
			normalsMap,
			 distThresh,
			shortIteration,
			rotationOnly,
			sumHessian_out,
			sumNabla_out,
			noValidPoints_out,
			sumF_out);

	/*
	// result
	printf("sumHessian \n");
	for (int i = 0; i < HESSIAN_SIZE; i++) {
		printf("%f", sumHessian_out[i]);
	}
	printf("\n");

	printf("sumNabla \n");
	for (int i = 0; i < 6; i++) {
		printf("%f", sumNabla_out[i]);
	}
	printf("\n");

	printf("noValidPoints: %d", *noValidPoints_out);
	printf("sum_F: %f", *sumF_out);
	*/

	free(depth);
	free(pointsMap);
	free(normalsMap);
}
