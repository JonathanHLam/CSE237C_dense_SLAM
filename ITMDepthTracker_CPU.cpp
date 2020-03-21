// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMDepthTracker_CPU.h"
#include "../Shared/ITMDepthTracker_Shared.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>

using namespace ITMLib;

ITMDepthTracker_CPU::ITMDepthTracker_CPU(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels,
	float terminationThreshold, float failureDetectorThreshold, const ITMLowLevelEngine *lowLevelEngine)
 : ITMDepthTracker(imgSize, trackingRegime, noHierarchyLevels, terminationThreshold,  failureDetectorThreshold, lowLevelEngine, MEMORYDEVICE_CPU)
{ }

ITMDepthTracker_CPU::~ITMDepthTracker_CPU(void) { }

int ITMDepthTracker_CPU::ComputeGandH(float &f, float *nabla, float *hessian, Matrix4f approxInvPose)
{
	std::cout << "\n ICP called \n";
	Vector4f *pointsMap = sceneHierarchyLevel->pointsMap->GetData(MEMORYDEVICE_CPU);

	Vector4f *normalsMap = sceneHierarchyLevel->normalsMap->GetData(MEMORYDEVICE_CPU);

	Vector4f sceneIntrinsics = sceneHierarchyLevel->intrinsics;
	std::cout << "sceneIntrinsics " << sceneIntrinsics << "\n";

	Vector2i sceneImageSize = sceneHierarchyLevel->pointsMap->noDims;
	std::cout << "sceneImageSize " << sceneImageSize << "\n";

	std::ofstream fs;
	fs.open ("/Users/ShawnXu/Desktop/CSE237C_dense_SLAM/pointsMap.txt");
	for (int i = 0; i < sceneImageSize.x  * sceneImageSize.y; i++) {
		fs << pointsMap[i] << "\n";
	}
	fs << "\n";
	fs.close();

	fs.open ("/Users/ShawnXu/Desktop/CSE237C_dense_SLAM/normalsMap.txt");
	for (int i = 0; i < sceneImageSize.x  * sceneImageSize.y; i++) {
		fs << normalsMap[i] << "\n" ;
	}
	fs << "\n";
	fs.close();

	float *depth = viewHierarchyLevel->data->GetData(MEMORYDEVICE_CPU);

	Vector4f viewIntrinsics = viewHierarchyLevel->intrinsics;
	std::cout << "viewIntrinsics " << viewIntrinsics << "\n";

	Vector2i viewImageSize = viewHierarchyLevel->data->noDims;
	std::cout << "viewImageSize " << viewImageSize << "\n";
	std::cout << "\n";


	fs.open ("/Users/ShawnXu/Desktop/CSE237C_dense_SLAM/depth.txt");
	for (int i = 0; i < sceneImageSize.x  * sceneImageSize.y; i++) {
		fs << depth[i] << "\n";
	}
	fs << "\n";
	fs.close();


	if (iterationType == TRACKER_ITERATION_NONE) return 0;

	bool shortIteration = (iterationType == TRACKER_ITERATION_ROTATION) || (iterationType == TRACKER_ITERATION_TRANSLATION);

	float sumHessian[6 * 6], sumNabla[6], sumF; int noValidPoints;
	int noPara = shortIteration ? 3 : 6, noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;

	noValidPoints = 0; sumF = 0.0f;
	memset(sumHessian, 0, sizeof(float) * noParaSQ);
	memset(sumNabla, 0, sizeof(float) * noPara);

	std::cout << "approxInvPose " << approxInvPose << "\n";
	std::cout << "scenePose " << scenePose << "\n";
  std::cout << "levelId " << levelId << "\n";

	std::cout << "distThresh \n";
	for (int i = 0; i < 5; i++) {
		std::cout << distThresh[i] << ", ";
	}
	std::cout << " \n";

	std::cout << "ShortIteration " << shortIteration << "\n";
	std::cout << "Rotation Only " << (iterationType == TRACKER_ITERATION_ROTATION) << "\n";

	for (int y = 0; y < viewImageSize.y; y++) for (int x = 0; x < viewImageSize.x; x++)
	{
		float localHessian[6 + 5 + 4 + 3 + 2 + 1], localNabla[6], localF = 0;

		for (int i = 0; i < noPara; i++) localNabla[i] = 0.0f;
		for (int i = 0; i < noParaSQ; i++) localHessian[i] = 0.0f;

		bool isValidPoint;

		switch (iterationType)
		{
		case TRACKER_ITERATION_ROTATION:
			isValidPoint = computePerPointGH_Depth<true, true>(localNabla, localHessian, localF, x, y, depth[x + y * viewImageSize.x], viewImageSize,
				viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh[levelId]);
			break;
		case TRACKER_ITERATION_TRANSLATION:
			isValidPoint = computePerPointGH_Depth<true, false>(localNabla, localHessian, localF, x, y, depth[x + y * viewImageSize.x], viewImageSize,
				viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh[levelId]);
			break;
		case TRACKER_ITERATION_BOTH:
			isValidPoint = computePerPointGH_Depth<false, false>(localNabla, localHessian, localF, x, y, depth[x + y * viewImageSize.x], viewImageSize,
				viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh[levelId]);
			break;
		default:
			isValidPoint = false;
			break;
		}

		if (isValidPoint)
		{
			noValidPoints++; sumF += localF;
			for (int i = 0; i < noPara; i++) sumNabla[i] += localNabla[i];
			for (int i = 0; i < noParaSQ; i++) sumHessian[i] += localHessian[i];
		}
	}

	for (int r = 0, counter = 0; r < noPara; r++) for (int c = 0; c <= r; c++, counter++) hessian[r + c * 6] = sumHessian[counter];
	for (int r = 0; r < noPara; ++r) for (int c = r + 1; c < noPara; c++) hessian[r + c * 6] = hessian[c + r * 6];

	memcpy(nabla, sumNabla, noPara * sizeof(float));
	f = (noValidPoints > 100) ? sumF / noValidPoints : 1e5f;

	std::cout << "Outputs................................\n";

	std::cout << "sumHessian ";
	for (int i = 0; i < 36; i++) {
		std::cout << sumHessian[i] << ", ";
	}
	std::cout << " \n";

	std::cout << "sumNabla ";
	for (int i = 0; i < 6; i++) {
		std::cout << sumNabla[i] << ", ";
	}
	std::cout << " \n";

	std::cout << "noValidPoints " << noValidPoints << "\n";
	std::cout << "sumF " << sumF << "\n";

	exit(9);
	return noValidPoints;
}
