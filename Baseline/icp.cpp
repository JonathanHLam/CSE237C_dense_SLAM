#include "icp.h"
#include <stdio.h>

//#define matVecMul3_2(m, vec) ((float4) (dot((m).data0, (vec)), dot((m).data1, (vec)), dot((m).data2, (vec)), 1.f))


float4 matVecMul3_2(Mat44_cl m, float4 vec) {
	float4 out;
	out.x = dot((m).data0, (vec));
	out.y = dot((m).data1, (vec));
	out.z = dot((m).data2, (vec));
	out.x = 1.f;

	return out;
}

#if (KNOB4_INTERPOLATE_NORMAL != 0 || KNOB4_INTERPOLATE_POINT != 0)
float4 interpolateBilinear_withHoles_OpenCL(
		const float4* source,
		float2 position,
		int2   imgSize)
{
	float4 a, b, c, d;
	float4 result;
	short2 p;
	float2 delta;

	p.x = (short)floor(position.x);
	p.y = (short)floor(position.y);
	delta.x = position.x - (float)p.x;
	delta.y = position.y - (float)p.y;

	a = source[p.x + p.y * imgSize.x];
	b = source[(p.x + 1) + p.y * imgSize.x];
	c = source[p.x + (p.y + 1) * imgSize.x];
	d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	if (a.w < 0 || b.w < 0 || c.w < 0 || d.w < 0)
	{
		result.x = 0; result.y = 0; result.z = 0; result.w = -1.0f;
	}
	else {
	result.x = ((float)a.x * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.x * delta.x * (1.0f - delta.y) +
		(float)c.x * (1.0f - delta.x) * delta.y + (float)d.x * delta.x * delta.y);
	result.y = ((float)a.y * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.y * delta.x * (1.0f - delta.y) +
		(float)c.y * (1.0f - delta.x) * delta.y + (float)d.y * delta.x * delta.y);
	result.z = ((float)a.z * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.z * delta.x * (1.0f - delta.y) +
		(float)c.z * (1.0f - delta.x) * delta.y + (float)d.z * delta.x * delta.y);
	result.w = ((float)a.w * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.w * delta.x * (1.0f - delta.y) +
		(float)c.w * (1.0f - delta.x) * delta.y + (float)d.w * delta.x * delta.y);
	}
	return result;
}
#endif


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
#if KNOB4_USE_ND_RANGE == 1
		,
		write_only pipe unsigned char __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) valid_pipe,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) sumF_pipe,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) nabla_pipe0,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) nabla_pipe1,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) nabla_pipe2,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) nabla_pipe3,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) nabla_pipe4,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) nabla_pipe5,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe0,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe1,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe2,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe3,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe4,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe5,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe6,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe7,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe8,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe9,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe10,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe11,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe12,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe13,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe14,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe15,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe16,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe17,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe18,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe19,
		write_only pipe float __attribute__((blocking))
		                      __attribute__((depth(KNOB4_PIPE_DEPTH))) hessian_pipe20
#endif
)
{





#if KNOB4_SUM_NABLA_TYPE != 2 || KNOB4_UNROLL_NABLA == 0 || KNOB4_UNROLL_HESSIAN == 0
#if KNOB4_SHORT_ITERATION_BRANCH != 0
	const int noPara = shortIteration ? 3 : 6;
#else
	const int noPara = 6;
#endif
#endif

#if KNOB4_SUM_HESSIAN_TYPE != 2
#if KNOB4_SHORT_ITERATION_BRANCH != 0
	const int noParaSQ = shortIteration ? SHORT_HESSIAN_SIZE : HESSIAN_SIZE;
#else
	const int noParaSQ = HESSIAN_SIZE;
#endif
#endif


	int noValidPoints_local = 0;

#if KNOB4_SUM_NABLA_TYPE != 2 && KNOB4_SUM_HESSIAN_TYPE != 2
	float sumF_local        = 0.f;
#endif


	// Initialize Nabla buffer
#if KNOB4_SUM_NABLA_TYPE == 2
	float8 nabla_buffer;
	nabla_buffer.s0 = 0;
	nabla_buffer.s1 = 0;
	nabla_buffer.s2 = 0;
	nabla_buffer.s3 = 0;
	nabla_buffer.s4 = 0;
	nabla_buffer.s5 = 0;
	nabla_buffer.s6 = 0;
	nabla_buffer.s7 = 0;

	float8 nabla_shiftreg[KNOB4_SHIFT_REG_SIZE];
	for(unsigned int i = 0; i < KNOB4_SHIFT_REG_SIZE; i++)
	{
		nabla_shiftreg[i] = nabla_buffer;
	}
#else
	float sumNabla_local[6];
	for (int i = 0; i < 6; i++) sumNabla_local[i] = 0.0f;
#endif


	// Initialize Hessian buffer
#if KNOB4_SUM_HESSIAN_TYPE == 2
	float16 hessian_buffer1;
	float8  hessian_buffer2;

	hessian_buffer1.s0 = 0;
	hessian_buffer1.s1 = 0;
	hessian_buffer1.s2 = 0;
	hessian_buffer1.s3 = 0;
	hessian_buffer1.s4 = 0;
	hessian_buffer1.s5 = 0;
	hessian_buffer1.s6 = 0;
	hessian_buffer1.s7 = 0;
	hessian_buffer1.s8 = 0;
	hessian_buffer1.s9 = 0;
	hessian_buffer1.sa = 0;
	hessian_buffer1.sb = 0;
	hessian_buffer1.sc = 0;
	hessian_buffer1.sd = 0;
	hessian_buffer1.se = 0;
	hessian_buffer1.sf = 0;

	hessian_buffer2.s0 = 0;
	hessian_buffer2.s1 = 0;
	hessian_buffer2.s2 = 0;
	hessian_buffer2.s3 = 0;
	hessian_buffer2.s4 = 0;
	hessian_buffer2.s5 = 0;
	hessian_buffer2.s6 = 0;
	hessian_buffer2.s7 = 0;


	float16 hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE];
	float8  hessian_shiftreg2[KNOB4_SHIFT_REG_SIZE];
	for(unsigned int i = 0; i < KNOB4_SHIFT_REG_SIZE; i++)
	{
		hessian_shiftreg1[i] = hessian_buffer1;
		hessian_shiftreg2[i] = hessian_buffer2;
	}
#else
	float sumHessian_local[HESSIAN_SIZE];
	for (int i = 0; i < HESSIAN_SIZE; i++) sumHessian_local[i] = 0.0f;
#endif

	// Loop over X and Y
#if KNOB4_USE_ND_RANGE == 0
#if KNOB4_FLATTEN_LOOP != 0
	int x = -1, y = 0;
	for (int xy = 0; xy < viewImageSize.x * viewImageSize.y; xy++)
#else
	for (int y = 0; y < viewImageSize.y; y++) {
	//for (int y = 0; y < MAX_IMG_Y; y++)	{

#pragma HLS unroll factor=2
		for (int x = 0; x < viewImageSize.x; x++)
		//for (int x = 0; x < MAX_IMG_X; x++)

#endif
#else
	uchar isValid = 0;
	float localF = 0.f;
	float localHessian[HESSIAN_SIZE] = {0};
	float localNabla[6] = {0};
	int x = get_global_id(0);
	int y = get_global_id(1);
#endif
		{
#pragma HLS pipeline
#if KNOB4_USE_ND_RANGE == 0
			float localHessian[HESSIAN_SIZE];
			float localNabla[6];
#endif

			float A[6];
			float b;

#if KNOB4_FLATTEN_LOOP != 0
			x++;
			if(x >= viewImageSize.x){ x = 0; y++; }

			float d = depth[xy];
#else

			float d = depth[x + y * viewImageSize.x];
			//printf("%d X = %d, Y = %d Depth = %f\n",x+y*viewImageSize.x, x, y, d);
#endif

			if (d > 1e-8f) //check if valid -- != 0.0f
			{
				//printf("2 X = %d, Y = %d d = %f\n", x, y, d);
				float4 tmp3Dpoint, tmp3Dpoint_reproj;
				float3 ptDiff;
				float4 curr3Dpoint, corr3Dnormal;
				float2 tmp2Dpoint;

				tmp3Dpoint.x = d * (((float)x - viewIntrinsics.z) * viewIntrinsics.x); // viewIntrinsics.x is 1/fx
				tmp3Dpoint.y = d * (((float)y - viewIntrinsics.w) * viewIntrinsics.y); // viewIntrinsics.y is 1/fy
				tmp3Dpoint.z = d;
				tmp3Dpoint.w = 1.0f;

				// transform to previous frame coordinates
				tmp3Dpoint = matVecMul3_2(*approxInvPose, tmp3Dpoint);
	//			tmp3Dpoint.w = 1.0f;

				// project into previous rendered image
				tmp3Dpoint_reproj = matVecMul3_2(*scenePose, tmp3Dpoint);

				if (tmp3Dpoint_reproj.z > 0.0f)
				{
					tmp2Dpoint.x = sceneIntrinsics.x * tmp3Dpoint_reproj.x / tmp3Dpoint_reproj.z + sceneIntrinsics.z;
					tmp2Dpoint.y = sceneIntrinsics.y * tmp3Dpoint_reproj.y / tmp3Dpoint_reproj.z + sceneIntrinsics.w;

					if ((tmp2Dpoint.x >= 0.0f) && (tmp2Dpoint.x <= sceneImageSize.x - 2) && (tmp2Dpoint.y >= 0.0f) && (tmp2Dpoint.y <= sceneImageSize.y - 2))
					{

			#if KNOB4_INTERPOLATE_POINT != 0
						curr3Dpoint = interpolateBilinear_withHoles_OpenCL(pointsMap, tmp2Dpoint, sceneImageSize);
			#else
						curr3Dpoint = getValueAtPoint(pointsMap, tmp2Dpoint, sceneImageSize.x);
			#endif
						if (curr3Dpoint.w >= 0.0f)
						{

							ptDiff.x = curr3Dpoint.x - tmp3Dpoint.x;
							ptDiff.y = curr3Dpoint.y - tmp3Dpoint.y;
							ptDiff.z = curr3Dpoint.z - tmp3Dpoint.z;
							float dist = ptDiff.x * ptDiff.x + ptDiff.y * ptDiff.y + ptDiff.z * ptDiff.z;

							if (dist <= distThresh)
							{

					#if KNOB4_INTERPOLATE_POINT != 0
								corr3Dnormal = interpolateBilinear_withHoles_OpenCL(normalsMap, tmp2Dpoint, sceneImageSize);
								if(corr3Dnormal.w >= 0.0f)
					#else
								corr3Dnormal = getValueAtPoint(normalsMap, tmp2Dpoint, sceneImageSize.x);
					#endif
								{

									b = corr3Dnormal.x * ptDiff.x + corr3Dnormal.y * ptDiff.y + corr3Dnormal.z * ptDiff.z;


									if (shortIteration)
									{
										if (rotationOnly)
										{
											A[0] = +tmp3Dpoint.z * corr3Dnormal.y - tmp3Dpoint.y * corr3Dnormal.z;
											A[1] = -tmp3Dpoint.z * corr3Dnormal.x + tmp3Dpoint.x * corr3Dnormal.z;
											A[2] = +tmp3Dpoint.y * corr3Dnormal.x - tmp3Dpoint.x * corr3Dnormal.y;
										}
										else
										{
											A[0] = corr3Dnormal.x;
											A[1] = corr3Dnormal.y;
											A[2] = corr3Dnormal.z;
										}
									}
									else
									{
										A[0] = +tmp3Dpoint.z * corr3Dnormal.y - tmp3Dpoint.y * corr3Dnormal.z;
										A[1] = -tmp3Dpoint.z * corr3Dnormal.x + tmp3Dpoint.x * corr3Dnormal.z;
										A[2] = +tmp3Dpoint.y * corr3Dnormal.x - tmp3Dpoint.x * corr3Dnormal.y;
										A[3] = corr3Dnormal.x;
										A[4] = corr3Dnormal.y;
										A[5] = corr3Dnormal.z;
									}


						#if KNOB4_USE_ND_RANGE == 0
									float localF = b * b;
						#else
									localF = b * b;
						#endif

									// --------------------
									// Calculate Hessian and Nabla
									// --------------------
						#if (KNOB4_UNROLL_HESSIAN == 0)
									for (int r = 0, counter = 0; r < noPara; r++)
									{
						#if (KNOB4_UNROLL_NABLA == 0)
										localNabla[r] = b * A[r];
						#endif

										for (int c = 0; c <= r; c++, counter++)
										{
											localHessian[counter] = A[r] * A[c];
										}
									}
						#endif

						#if (KNOB4_UNROLL_NABLA != 0)
									localNabla[0] = b * A[0];
									localNabla[1] = b * A[1];
									localNabla[2] = b * A[2];
						#if (KNOB4_SHORT_ITERATION_BRANCH != 0)
									if(!shortIteration)
						#endif
									{
										localNabla[3] = b * A[3];
										localNabla[4] = b * A[4];
										localNabla[5] = b * A[5];
									}
						#endif

						#if (KNOB4_UNROLL_HESSIAN != 0)

						#if (KNOB4_UNROLL_NABLA == 0)
									for (int r = 0; r < noPara; r++)
									{
										localNabla[r] = b * A[r];
									}
						#endif

									localHessian[0] = A[0] * A[0];
									localHessian[1] = A[1] * A[0];
									localHessian[2] = A[1] * A[1];
									localHessian[3] = A[2] * A[0];
									localHessian[4] = A[2] * A[1];
									localHessian[5] = A[2] * A[2];
						#if (KNOB4_SHORT_ITERATION_BRANCH != 0)
									if(!shortIteration)
						#endif
									{
										localHessian[6] = A[3] * A[0];
										localHessian[7] = A[3] * A[1];
										localHessian[8] = A[3] * A[2];
										localHessian[9] = A[3] * A[3];
										localHessian[10] = A[4] * A[0];
										localHessian[11] = A[4] * A[1];
										localHessian[12] = A[4] * A[2];
										localHessian[13] = A[4] * A[3];
										localHessian[14] = A[4] * A[4];
										localHessian[15] = A[5] * A[0];
										localHessian[16] = A[5] * A[1];
										localHessian[17] = A[5] * A[2];
										localHessian[18] = A[5] * A[3];
										localHessian[19] = A[5] * A[4];
										localHessian[20] = A[5] * A[5];
									}
						#endif




#if KNOB4_USE_ND_RANGE == 1

								} // if corr3Dnormal is valid
							} // if dist <= distThresh
						} // if curr3Dpoint is valid
					} // if tmp2Dpoint is valid
				} // if tmp3Dpoint_reproj.z > 0.0f
			} // if d != 0


			write_pipe(valid_pipe, &isValid);
			mem_fence(CLK_CHANNEL_MEM_FENCE);

			write_pipe(sumF_pipe, &localF);
			mem_fence(CLK_CHANNEL_MEM_FENCE);

			write_pipe(nabla_pipe0, &localNabla[0]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
			write_pipe(nabla_pipe1, &localNabla[1]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
			write_pipe(nabla_pipe2, &localNabla[2]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
		#if KNOB4_SHORT_ITERATION_BRANCH != 0
			if(!shortIteration)
		#endif
			{
				write_pipe(nabla_pipe3, &localNabla[3]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(nabla_pipe4, &localNabla[4]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(nabla_pipe5, &localNabla[5]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
			}

			write_pipe(hessian_pipe0, &localHessian[0]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
			write_pipe(hessian_pipe1, &localHessian[1]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
			write_pipe(hessian_pipe2, &localHessian[2]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
			write_pipe(hessian_pipe3, &localHessian[3]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
			write_pipe(hessian_pipe4, &localHessian[4]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
			write_pipe(hessian_pipe5, &localHessian[5]);
			mem_fence(CLK_CHANNEL_MEM_FENCE);
		#if KNOB4_SHORT_ITERATION_BRANCH != 0
			if(!shortIteration)
		#endif
			{
				write_pipe(hessian_pipe6, &localHessian[6]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe7, &localHessian[7]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe8, &localHessian[8]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe9, &localHessian[9]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe10, &localHessian[10]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe11, &localHessian[11]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe12, &localHessian[12]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe13, &localHessian[13]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe14, &localHessian[14]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe15, &localHessian[15]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe16, &localHessian[16]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe17, &localHessian[17]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe18, &localHessian[18]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe19, &localHessian[19]);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
				write_pipe(hessian_pipe20, &localHessian[20]);
			}
		}
#else

									// --------------------
									// Count valid points
									// --------------------
									noValidPoints_local++;

									// --------------------
									// Accumulate value
									// --------------------
						#if KNOB4_SUM_NABLA_TYPE != 2 && KNOB4_SUM_HESSIAN_TYPE != 2
									sumF_local += localF;
						#endif


									// --------------------
									// Accumulate Nabla
									// --------------------
						#if KNOB4_SUM_NABLA_TYPE == 0
									for (int i = 0; i < noPara; i++)   sumNabla_local[i]   += localNabla[i];
						#elif KNOB4_SUM_NABLA_TYPE == 1
									sumNabla_local[0]   += localNabla[0];
									sumNabla_local[1]   += localNabla[1];
									sumNabla_local[2]   += localNabla[2];
						#if KNOB4_SHORT_ITERATION_BRANCH != 0
									if(!shortIteration)
						#endif
									{
										sumNabla_local[3]   += localNabla[3];
										sumNabla_local[4]   += localNabla[4];
										sumNabla_local[5]   += localNabla[5];
									}
						#elif KNOB4_SUM_NABLA_TYPE == 2
									float8 nabla_buffer_cur;
									nabla_buffer_cur.s0 = nabla_shiftreg[KNOB4_SHIFT_REG_SIZE-1].s0 + localNabla[0];
									nabla_buffer_cur.s1 = nabla_shiftreg[KNOB4_SHIFT_REG_SIZE-1].s1 + localNabla[1];
									nabla_buffer_cur.s2 = nabla_shiftreg[KNOB4_SHIFT_REG_SIZE-1].s2 + localNabla[2];
						#if KNOB4_SHORT_ITERATION_BRANCH != 0
									if(!shortIteration)
						#endif
									{
										nabla_buffer_cur.s3 = nabla_shiftreg[KNOB4_SHIFT_REG_SIZE-1].s3 + localNabla[3];
										nabla_buffer_cur.s4 = nabla_shiftreg[KNOB4_SHIFT_REG_SIZE-1].s4 + localNabla[4];
										nabla_buffer_cur.s5 = nabla_shiftreg[KNOB4_SHIFT_REG_SIZE-1].s5 + localNabla[5];
									}

									// This one holds sumF_local
									nabla_buffer_cur.s6 = nabla_shiftreg[KNOB4_SHIFT_REG_SIZE-1].s6 + localF;

									for(unsigned int i = KNOB4_SHIFT_REG_SIZE-1; i > 0; i--)
									{
										nabla_shiftreg[i] = nabla_shiftreg[i-1];
									}
									nabla_shiftreg[0] = nabla_buffer_cur;
						#endif


									// --------------------
									// Accumulate Hessian
									// --------------------
						#if KNOB4_SUM_HESSIAN_TYPE == 0
									for (int i = 0; i < noParaSQ; i++) sumHessian_local[i] += localHessian[i];
						#elif KNOB4_SUM_HESSIAN_TYPE == 1
									sumHessian_local[0]  += localHessian[0];
									sumHessian_local[1]  += localHessian[1];
									sumHessian_local[2]  += localHessian[2];
									sumHessian_local[3]  += localHessian[3];
									sumHessian_local[4]  += localHessian[4];
									sumHessian_local[5]  += localHessian[5];
						#if KNOB4_SHORT_ITERATION_BRANCH != 0
									if(!shortIteration)
						#endif
									{
										sumHessian_local[6]  += localHessian[6];
										sumHessian_local[7]  += localHessian[7];
										sumHessian_local[8]  += localHessian[8];
										sumHessian_local[9]  += localHessian[9];
										sumHessian_local[10] += localHessian[10];
										sumHessian_local[11] += localHessian[11];
										sumHessian_local[12] += localHessian[12];
										sumHessian_local[13] += localHessian[13];
										sumHessian_local[14] += localHessian[14];
										sumHessian_local[15] += localHessian[15];
										sumHessian_local[16] += localHessian[16];
										sumHessian_local[17] += localHessian[17];
										sumHessian_local[18] += localHessian[18];
										sumHessian_local[19] += localHessian[19];
										sumHessian_local[20] += localHessian[20];
									}
						#elif KNOB4_SUM_HESSIAN_TYPE == 2
									float16 hessian_buffer_cur1;
									hessian_buffer_cur1.s0 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s0 + localHessian[0];
									hessian_buffer_cur1.s1 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s1 + localHessian[1];
									hessian_buffer_cur1.s2 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s2 + localHessian[2];
									hessian_buffer_cur1.s3 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s3 + localHessian[3];
									hessian_buffer_cur1.s4 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s4 + localHessian[4];
									hessian_buffer_cur1.s5 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s5 + localHessian[5];
						#if KNOB4_SHORT_ITERATION_BRANCH != 0
									if(shortIteration)
									{
						#if KNOB4_SUM_NABLA_TYPE != 2
										// This one holds sumF_local
										hessian_buffer_cur1.s6 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s6 + localF;
						#endif
									}
									else
						#endif
									{
										hessian_buffer_cur1.s6 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s6 + localHessian[6];
										hessian_buffer_cur1.s7 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s7 + localHessian[7];
										hessian_buffer_cur1.s8 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s8 + localHessian[8];
										hessian_buffer_cur1.s9 = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].s9 + localHessian[9];
										hessian_buffer_cur1.sa = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].sa + localHessian[10];
										hessian_buffer_cur1.sb = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].sb + localHessian[11];
										hessian_buffer_cur1.sc = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].sc + localHessian[12];
										hessian_buffer_cur1.sd = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].sd + localHessian[13];
										hessian_buffer_cur1.se = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].se + localHessian[14];
										hessian_buffer_cur1.sf = hessian_shiftreg1[KNOB4_SHIFT_REG_SIZE-1].sf + localHessian[15];
									}

									for(unsigned int i = KNOB4_SHIFT_REG_SIZE-1; i > 0; i--)
									{
										hessian_shiftreg1[i] = hessian_shiftreg1[i-1];
									}
									hessian_shiftreg1[0] = hessian_buffer_cur1;

						#if KNOB4_SHORT_ITERATION_BRANCH != 0
									if(!shortIteration)
						#endif
									{
										float8  hessian_buffer_cur2;
										hessian_buffer_cur2.s0  = hessian_shiftreg2[KNOB4_SHIFT_REG_SIZE-1].s0  + localHessian[16];
										hessian_buffer_cur2.s1  = hessian_shiftreg2[KNOB4_SHIFT_REG_SIZE-1].s1  + localHessian[17];
										hessian_buffer_cur2.s2  = hessian_shiftreg2[KNOB4_SHIFT_REG_SIZE-1].s2  + localHessian[18];
										hessian_buffer_cur2.s3  = hessian_shiftreg2[KNOB4_SHIFT_REG_SIZE-1].s3  + localHessian[19];
										hessian_buffer_cur2.s4  = hessian_shiftreg2[KNOB4_SHIFT_REG_SIZE-1].s4  + localHessian[20];

						#if KNOB4_SUM_NABLA_TYPE != 2
										// This one holds sumF_local
										hessian_buffer_cur2.s5 = hessian_shiftreg2[KNOB4_SHIFT_REG_SIZE-1].s5 + localF;
						#endif

										for(unsigned int i = KNOB4_SHIFT_REG_SIZE-1; i > 0; i--)
										{
											hessian_shiftreg2[i] = hessian_shiftreg2[i-1];
										}
										hessian_shiftreg2[0] = hessian_buffer_cur2;
									}
						#endif
								} // if corr3Dnormal is valid
							} // if dist <= distThresh
						} // if curr3Dpoint is valid
					} // if tmp2Dpoint is valid
				} // if tmp3Dpoint_reproj.z > 0.0f
			}// if d != 0
		}// x
	}// y


	// Store Nabla
#if KNOB4_SUM_NABLA_TYPE == 2
	for(unsigned int i = 0; i < KNOB4_SHIFT_REG_SIZE; i++)
	{
		nabla_buffer.s0 += nabla_shiftreg[i].s0;
		nabla_buffer.s1 += nabla_shiftreg[i].s1;
		nabla_buffer.s2 += nabla_shiftreg[i].s2;
		nabla_buffer.s3 += nabla_shiftreg[i].s3;
		nabla_buffer.s4 += nabla_shiftreg[i].s4;
		nabla_buffer.s5 += nabla_shiftreg[i].s5;
		nabla_buffer.s6 += nabla_shiftreg[i].s6;
		nabla_buffer.s7 += nabla_shiftreg[i].s7;
	}
	sumNabla[0] = nabla_buffer.s0;
	sumNabla[1] = nabla_buffer.s1;
	sumNabla[2] = nabla_buffer.s2;
#if KNOB4_SHORT_ITERATION_BRANCH != 0
	if(!shortIteration)
#endif
	{
		sumNabla[3] = nabla_buffer.s3;
		sumNabla[4] = nabla_buffer.s4;
		sumNabla[5] = nabla_buffer.s5;
	}
#else
	for (int i = 0; i < noPara; i++)   sumNabla[i]   = sumNabla_local[i];
#endif

	// Store Hessian
#if KNOB4_SUM_HESSIAN_TYPE == 2
	for(unsigned int i = 0; i < KNOB4_SHIFT_REG_SIZE; i++)
	{
		hessian_buffer1.s0 += hessian_shiftreg1[i].s0;
		hessian_buffer1.s1 += hessian_shiftreg1[i].s1;
		hessian_buffer1.s2 += hessian_shiftreg1[i].s2;
		hessian_buffer1.s3 += hessian_shiftreg1[i].s3;
		hessian_buffer1.s4 += hessian_shiftreg1[i].s4;
		hessian_buffer1.s5 += hessian_shiftreg1[i].s5;
		hessian_buffer1.s6 += hessian_shiftreg1[i].s6;
		hessian_buffer1.s7 += hessian_shiftreg1[i].s7;
		hessian_buffer1.s8 += hessian_shiftreg1[i].s8;
		hessian_buffer1.s9 += hessian_shiftreg1[i].s9;
		hessian_buffer1.sa += hessian_shiftreg1[i].sa;
		hessian_buffer1.sb += hessian_shiftreg1[i].sb;
		hessian_buffer1.sc += hessian_shiftreg1[i].sc;
		hessian_buffer1.sd += hessian_shiftreg1[i].sd;
		hessian_buffer1.se += hessian_shiftreg1[i].se;
		hessian_buffer1.sf += hessian_shiftreg1[i].sf;
	}
	sumHessian[0]  = hessian_buffer1.s0;
	sumHessian[1]  = hessian_buffer1.s1;
	sumHessian[2]  = hessian_buffer1.s2;
	sumHessian[3]  = hessian_buffer1.s3;
	sumHessian[4]  = hessian_buffer1.s4;
	sumHessian[5]  = hessian_buffer1.s5;
#if KNOB4_SHORT_ITERATION_BRANCH != 0
	if(!shortIteration)
#endif
	{
		sumHessian[6]  = hessian_buffer1.s6;
		sumHessian[7]  = hessian_buffer1.s7;
		sumHessian[8]  = hessian_buffer1.s8;
		sumHessian[9]  = hessian_buffer1.s9;
		sumHessian[10] = hessian_buffer1.sa;
		sumHessian[11] = hessian_buffer1.sb;
		sumHessian[12] = hessian_buffer1.sc;
		sumHessian[13] = hessian_buffer1.sd;
		sumHessian[14] = hessian_buffer1.se;
		sumHessian[15] = hessian_buffer1.sf;

		for(unsigned int i = 0; i < KNOB4_SHIFT_REG_SIZE; i++)
		{
			hessian_buffer2.s0 += hessian_shiftreg2[i].s0;
			hessian_buffer2.s1 += hessian_shiftreg2[i].s1;
			hessian_buffer2.s2 += hessian_shiftreg2[i].s2;
			hessian_buffer2.s3 += hessian_shiftreg2[i].s3;
			hessian_buffer2.s4 += hessian_shiftreg2[i].s4;
			hessian_buffer2.s5 += hessian_shiftreg2[i].s5;
			hessian_buffer2.s6 += hessian_shiftreg2[i].s6;
			hessian_buffer2.s7 += hessian_shiftreg2[i].s7;
		}

		sumHessian[16] = hessian_buffer2.s0;
		sumHessian[17] = hessian_buffer2.s1;
		sumHessian[18] = hessian_buffer2.s2;
		sumHessian[19] = hessian_buffer2.s3;
		sumHessian[20] = hessian_buffer2.s4;
	}
#else
	for (int i = 0; i < noParaSQ; i++) sumHessian[i] = sumHessian_local[i];
#endif

	// Store num valid points
	*noValidPoints = noValidPoints_local;

	// Store F
#if KNOB4_SUM_NABLA_TYPE == 2
	*sumF = nabla_buffer.s6;
#elif KNOB4_SUM_HESSIAN_TYPE == 2
#if KNOB4_SHORT_ITERATION_BRANCH != 0
	if(shortIteration)
	{
		*sumF = hessian_buffer1.s6;
	}
	else
#endif
	{
		*sumF = hessian_buffer2.s5;
	}
#else
	*sumF = sumF_local;
#endif

#endif // Use ND range

}
