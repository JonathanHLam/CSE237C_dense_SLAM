#include <math.h>

#ifndef ICP_H_
#define ICP_H_

struct short2 {
    int x;
    int y;
};

struct int2 {
    int x;
    int y;
};

struct float2 {
    float x;
    float y;
};

struct float3 {
    float x;
    float y;
    float z;
};

struct float4 {
    float x;
    float y;
    float z;
    float w;
};

struct float8 {
    float s0;
    float s1;
    float s2;
    float s3;
    float s4;
    float s5;
    float s6;
    float s7;
};

struct float16 {
    float s0;
    float s1;
    float s2;
    float s3;
    float s4;
    float s5;
    float s6;
    float s7;
    float s8;
    float s9;
    float sa;
    float sb;
    float sc;
    float sd;
    float se;
    float sf;
};

struct Mat44_cl {
	float4 data0;
	float4 data1;
	float4 data2;
	float4 data3;

};

#define KNOB4_INTERPOLATE_POINT      1 // 0: do not interpolate | 1: bilinear interpolation
#define KNOB4_INTERPOLATE_NORMAL     1 // 0: do not interpolate | 1: bilinear interpolation
#define KNOB4_UNROLL_NABLA           1 // 0: no unroll | 1: full unroll
#define KNOB4_UNROLL_HESSIAN         0 // 0: no unroll | 1: full unroll
#define KNOB4_SHORT_ITERATION_BRANCH 1 // 0: no branches, always long iteration | 1: have short iteration branches
#define KNOB4_SUM_NABLA_TYPE         2 // 0: loop | 1: unrolled loop | 2: shift register
#define KNOB4_SUM_HESSIAN_TYPE       2 // 0: loop | 1: unrolled loop | 2: shift register
#define KNOB4_SHIFT_REG_SIZE         2
#define KNOB4_FLATTEN_LOOP           0 // 0: loop x and y | 1: combined loop
#define KNOB4_USE_ND_RANGE           0 // 0: single WI and loop | 1: Multiple WI with second kernel
#define KNOB4_PIPE_DEPTH             2
#define KNOB4_COMPUTE_UNITS          1


#define HESSIAN_SIZE        21 // (6 + 5 + 4 + 3 + 2 + 1)
#define SHORT_HESSIAN_SIZE  6  //(3 + 2 + 1)

//Set this according to image size
#define MAX_IMG_X			640
#define MAX_IMG_Y			480


//#define matVecMul3_2(m, vec) ((float4) (dot((m).data0, (vec)), dot((m).data1, (vec)), dot((m).data2, (vec)), 1.f))

#define dot(a,b) ((float) ((a.x*b.x) + (a.y*b.y) + (a.z*b.z) + (a.w*b.w)))

#define getValueAtPoint(source, position, sizeX) ( source[(int)round(position.x) + (int)round(position.y) * sizeX] )

float4 matVecMul3_2(
		Mat44_cl m,
		float4 vec
);

float4 interpolateBilinear_withHoles_OpenCL(
		const float4* source,
		float2 position,
		int2   imgSize
);

void icp (
		const float depth[MAX_IMG_X * MAX_IMG_Y],
		int2   viewImageSize,
		float4 viewIntrinsics,
		const Mat44_cl* approxInvPose,
		const Mat44_cl* scenePose,
		int2   sceneImageSize,
		float4 sceneIntrinsics,
		const float4 pointsMap[MAX_IMG_X * MAX_IMG_Y],
		const float4 normalsMap[MAX_IMG_X * MAX_IMG_Y],
		float distThresh,
		unsigned char shortIteration,
		unsigned char rotationOnly,
		float sumHessian[HESSIAN_SIZE],
		float sumNabla[6],
		int*   noValidPoints,
		float* sumF
);




#endif
