#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
//#include <ctime>
#include <Windows.h>
#include "rdrand.h"
#include <intrin.h>
#include <immintrin.h>
#include <string.h>

#define RDRAND_MASK	0x40000000
#define RETRY_LIMIT 10
#ifdef _WIN64
typedef uint64_t _wordlen_t;
#else
typedef uint32_t _wordlen_t;
#endif

#define BLOCKS				256
#define THREADS				512
#define TRIALS				1024*16*8 // cannot be larger than 2^32
//#define TRIALS				1024*16*1024*2 // cannot be larger than 2^32
#define bit8 unsigned char
#define bit32 unsigned int
#define bit64 unsigned __int64 
#define ROTL(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d) (             \
	a += b, d ^= a, d = ROTL(d, 16), \
	c += d, b ^= c, b = ROTL(b, 12), \
	a += b, d ^= a, d = ROTL(d,  8), \
	c += d, b ^= c, b = ROTL(b,  7))

#define QR_half(a, b, c, d) (             \
	a += b, d ^= a, d = ROTL(d, 16), \
	c += d, b ^= c, b = ROTL(b, 12)) 

#define ROUNDS 20
int RdRand_cpuid() {
	int info[4] = { -1, -1, -1, -1 };
	/* Are we on an Intel processor? */
	__cpuid(info, /*feature bits*/0);
	if (memcmp((void*)&info[1], (void*)"Genu", 4) != 0 ||
		memcmp((void*)&info[3], (void*)"ineI", 4) != 0 ||
		memcmp((void*)&info[2], (void*)"ntel", 4) != 0) {
		return 0;
	}
	/* Do we have RDRAND? */
	__cpuid(info, /*feature bits*/1);
	int ecx = info[2];
	if ((ecx & RDRAND_MASK) == RDRAND_MASK)
		return 1;
	else
		return 0;
}
int RdRand_isSupported() {
	static int supported = RDRAND_SUPPORT_UNKNOWN;
	if (supported == RDRAND_SUPPORT_UNKNOWN)	{
		if (RdRand_cpuid())
			supported = RDRAND_SUPPORTED;
		else
			supported = RDRAND_UNSUPPORTED;
	}
	return (supported == RDRAND_SUPPORTED) ? 1 : 0;
}
int rdrand_16(uint16_t* x, int retry) {
	if (RdRand_isSupported())	{
		if (retry)		{
			for (int i = 0; i < RETRY_LIMIT; i++)			{
				if (_rdrand16_step(x))
					return RDRAND_SUCCESS;
			}
			return RDRAND_NOT_READY;
		}
		else		{
			if (_rdrand16_step(x))
				return RDRAND_SUCCESS;
			else
				return RDRAND_NOT_READY;
		}
	}
	else	{
		return RDRAND_UNSUPPORTED;
	}
}
int rdrand_32(uint32_t* x, int retry){
	if (RdRand_isSupported())	{
		if (retry)		{
			for (int i = 0; i < RETRY_LIMIT; i++)			{
				if (_rdrand32_step(x))
					return RDRAND_SUCCESS;
			}
			return RDRAND_NOT_READY;
		}
		else		{
			if (_rdrand32_step(x))
				return RDRAND_SUCCESS;
			else
				return RDRAND_NOT_READY;
		}
	}
	else	{
		return RDRAND_UNSUPPORTED;
	}
}
int rdrand_64(uint64_t* x, int retry) {
	if (RdRand_isSupported())	{
		if (retry)		{
			for (int i = 0; i < RETRY_LIMIT; i++)			{
				if (_rdrand64_step(x))
					return RDRAND_SUCCESS;
			}
			return RDRAND_NOT_READY;
		}
		else		{
			if (_rdrand64_step(x))
				return RDRAND_SUCCESS;
			else
				return RDRAND_NOT_READY;
		}
	}
	else	{
		return RDRAND_UNSUPPORTED;
	}
}
int rdrand_get_n_64(unsigned int n, uint64_t* dest){
	int success;
	int count;
	unsigned int i;
	for (i = 0; i < n; i++)	{
		count = 0;
		do		{
			success = rdrand_64(dest, 1);
		} while ((success == 0) && (count++ < RETRY_LIMIT));
		if (success != RDRAND_SUCCESS) return success;
		dest = &(dest[1]);
	}
	return RDRAND_SUCCESS;
}
int rdrand_get_n_32(unsigned int n, uint32_t* dest){
	int success;
	int count;
	unsigned int i;
	for (i = 0; i < n; i++)	{
		count = 0;
		do		{
			success = rdrand_32(dest, 1);
		} while ((success == 0) && (count++ < RETRY_LIMIT));
		if (success != RDRAND_SUCCESS) return success;
		dest = &(dest[1]);
	}
	return RDRAND_SUCCESS;
}
int rdrand_get_bytes(unsigned int n, unsigned char* dest){
	unsigned char* start;
	unsigned char* residualstart;
	_wordlen_t* blockstart;
	_wordlen_t i, temprand;
	unsigned int count;
	unsigned int residual;
	unsigned int startlen;
	unsigned int length;
	int success;

	/* Compute the address of the first 32- or 64- bit aligned block in the destination buffer, depending on whether we are in 32- or 64-bit mode */
	start = dest;
	if (((uint32_t)start % (uint32_t)sizeof(_wordlen_t)) == 0)	{
		blockstart = (_wordlen_t*)start;
		count = n;
		startlen = 0;
	}
	else	{
		blockstart = (_wordlen_t*)(((_wordlen_t)start & ~(_wordlen_t)(sizeof(_wordlen_t) - 1)) + (_wordlen_t)sizeof(_wordlen_t));
		count = n - (sizeof(_wordlen_t) - (unsigned int)((_wordlen_t)start % sizeof(_wordlen_t)));
		startlen = (unsigned int)((_wordlen_t)blockstart - (_wordlen_t)start);
	}
	/* Compute the number of 32- or 64- bit blocks and the remaining number of bytes */
	residual = count % sizeof(_wordlen_t);
	length = count / sizeof(_wordlen_t);
	if (residual != 0)	{
		residualstart = (unsigned char*)(blockstart + length);
	}
	/* Get a temporary random number for use in the residuals. Failout if retry fails */
	if (startlen > 0)	{
#ifdef _WIN64
		if ((success = rdrand_64((uint64_t*)&temprand, 1)) != RDRAND_SUCCESS) return success;
#else
		if ((success = rdrand_32((uint32_t*)&temprand, 1)) != RDRAND_SUCCESS) return success;
#endif
	}
	/* populate the starting misaligned block */
	for (i = 0; i < startlen; i++)	{
		start[i] = (unsigned char)(temprand & 0xff);
		temprand = temprand >> 8;
	}
	/* populate the central aligned block. Fail out if retry fails */
#ifdef _WIN64
	if ((success = rdrand_get_n_64(length, (uint64_t*)(blockstart))) != RDRAND_SUCCESS) return success;
#else
	if ((success = rdrand_get_n_32(length, (uint32_t*)(blockstart))) != RDRAND_SUCCESS) return success;
#endif
	/* populate the final misaligned block */
	if (residual > 0)	{
#ifdef _WIN64
		if ((success = rdrand_64((uint64_t*)&temprand, 1)) != RDRAND_SUCCESS) return success;
#else
		if ((success = rdrand_32((uint32_t*)&temprand, 1)) != RDRAND_SUCCESS) return success;
#endif
		for (i = 0; i < residual; i++)		{
			residualstart[i] = (unsigned char)(temprand & 0xff);
			temprand = temprand >> 8;
		}
	}
	return RDRAND_SUCCESS;
}

void chacha_block(bit32 out[16], bit32 const in[16]) {
	int i;	bit32 x[16];

	for (i = 0; i < 16; ++i)
		x[i] = in[i];
	// 10 loops × 2 rounds/loop = 20 rounds
	for (i = 0; i < ROUNDS; i += 2) {
		// Odd round
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
		// Even round
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
	}
	for (i = 0; i < 16; ++i)		out[i] = x[i] + in[i];
}
__global__ void ChaCha_exhaustive(bit32 ciphertext[16]) {
	int i;	bit32 x[16]; bit32 in[16];
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	for (bit32 j = 0; j < TRIALS; j++) {
		in[0] = 0x61707865;
		in[1] = 0x3320646e;
		in[2] = 0x79622d32;
		in[3] = 0x6b206574;
		in[4] = threadIndex;
		in[5] = j;
		in[6] = 0;	in[7] = 0;	in[8] = 0;	in[9] = 0;	in[10] = 0;	in[11] = 0;	in[12] = 0;	in[13] = 0;	in[14] = 0;	in[15] = 0;
#pragma unroll
		for (i = 0; i < 16; i++) x[i] = in[i];
		// 10 loops × 2 rounds/loop = 20 rounds
#pragma unroll
		for (i = 0; i < ROUNDS; i += 2) {
			// Odd round
			QR(x[0], x[4], x[8], x[12]); // column 1
			QR(x[1], x[5], x[9], x[13]); // column 2
			QR(x[2], x[6], x[10], x[14]); // column 3
			QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round
			QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
			QR(x[1], x[6], x[11], x[12]); // diagonal 2
			QR(x[2], x[7], x[8], x[13]); // diagonal 3
			QR(x[3], x[4], x[9], x[14]); // diagonal 4
		}
		if ((x[0] + in[0]) == ciphertext[0])
			if ((x[1] + in[1]) == ciphertext[1])
				if ((x[2] + in[2]) == ciphertext[2])
					if ((x[3] + in[3]) == ciphertext[3])
						if ((x[4] + in[4]) == ciphertext[4])
							if ((x[5] + in[5]) == ciphertext[5])
								if ((x[6] + in[6]) == ciphertext[6])
									if ((x[7] + in[7]) == ciphertext[7])
										if ((x[8] + in[8]) == ciphertext[8])
											if ((x[9] + in[9]) == ciphertext[9])
												if ((x[10] + in[10]) == ciphertext[10])
													if ((x[11] + in[11]) == ciphertext[11])
														if ((x[12] + in[12]) == ciphertext[12])
															if ((x[13] + in[13]) == ciphertext[13])
																if ((x[14] + in[14]) == ciphertext[14])
																	if ((x[15] + in[15]) == ciphertext[15])
																		printf("The key is %x %x\n", threadIndex, j);
	}
}
__global__ void ChaCha_exhaustive_noarrays(bit32 ciphertext[16]) {
	int i; bit32 in0, in1, in2, in3, in4, in5, in6, in7, in8,  in9, in10, in11, in12, in13, in14, in15;
	bit32 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j = 0; j < TRIALS; j++) {
		in0 = 0x61707865;
		in1 = 0x3320646e;
		in2 = 0x79622d32;
		in3 = 0x6b206574;
		in4 = threadIndex;
		in5 = j;
		in6 = 0;	in7 = 0;	in8 = 0;	in9 = 0;	in10 = 0;	in11 = 0;	in12 = 0;	in13 = 0;	in14 = 0;	in15 = 0;
		x0 = in0;
		x1 = in1;
		x2 = in2;
		x3 = in3;
		x4 = in4;
		x5 = in5;
		x6 = in6;
		x7 = in7;
		x8 = in8;
		x9 = in9;
		x10 = in10;
		x11 = in11;
		x12 = in12;
		x13 = in13;
		x14 = in14;
		x15 = in15;
		// 10 loops × 2 rounds/loop = 20 rounds
//#pragma unroll
		for (i = 0; i < ROUNDS; i += 2) {
			// Odd round
			QR(x0, x4, x8, x12); // column 1
			QR(x1, x5, x9, x13); // column 2
			QR(x2, x6, x10, x14); // column 3
			QR(x3, x7, x11, x15); // column 4
			// Even round
			QR(x0, x5, x10, x15); // diagonal 1 (main diagonal)
			QR(x1, x6, x11, x12); // diagonal 2
			QR(x2, x7, x8, x13); // diagonal 3
			QR(x3, x4, x9, x14); // diagonal 4
		}
		if ((x0 + in0) == ciphertext[0])
			if ((x1 + in1) == ciphertext[1])
				if ((x2 + in2) == ciphertext[2])
					if ((x3 + in3) == ciphertext[3])
						if ((x4 + in4) == ciphertext[4])
							if ((x5 + in5) == ciphertext[5])
								if ((x6 + in6) == ciphertext[6])
									if ((x7 + in7) == ciphertext[7])
										if ((x8 + in8) == ciphertext[8])
											if ((x9 + in9) == ciphertext[9])
												if ((x10 + in10) == ciphertext[10])
													if ((x11 + in11) == ciphertext[11])
														if ((x12 + in12) == ciphertext[12])
															if ((x13 + in13) == ciphertext[13])
																if ((x14 + in14) == ciphertext[14])
																	if ((x15 + in15) == ciphertext[15])
																		printf("The key is %x %x\n", threadIndex, j);
	}
}
__global__ void ChaCha_exhaustive_noarrays2(bit32 ciphertext[16]) {

	int i; bit32 in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15;
	bit32 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	bit32 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	c0 = ciphertext[0];
	c1 = ciphertext[1];
	c2 = ciphertext[2];
	c3 = ciphertext[3];
	c4 = ciphertext[4];
	c5 = ciphertext[5];
	c6 = ciphertext[6];
	c7 = ciphertext[7];
	c8 = ciphertext[8];
	c9 = ciphertext[9];
	c10 = ciphertext[10];
	c11 = ciphertext[11];
	c12 = ciphertext[12];
	c13 = ciphertext[13];
	c14 = ciphertext[14];
	c15 = ciphertext[15];
	for (int j = 0; j < TRIALS; j++) {
		in0 = 0x61707865;
		in1 = 0x3320646e;
		in2 = 0x79622d32;
		in3 = 0x6b206574;
		in4 = threadIndex;
		in5 = j;
		in6 = 0;	in7 = 0;	in8 = 0;	in9 = 0;	in10 = 0;	in11 = 0;	in12 = 0;	in13 = 0;	in14 = 0;	in15 = 0;
		x0 = in0;
		x1 = in1;
		x2 = in2;
		x3 = in3;
		x4 = in4;
		x5 = in5;
		x6 = in6;
		x7 = in7;
		x8 = in8;
		x9 = in9;
		x10 = in10;
		x11 = in11;
		x12 = in12;
		x13 = in13;
		x14 = in14;
		x15 = in15;
		// 10 loops × 2 rounds/loop = 20 rounds
//#pragma unroll
		for (i = 0; i < ROUNDS; i += 2) {
			// Odd round
			QR(x0, x4, x8, x12); // column 1
			QR(x1, x5, x9, x13); // column 2
			QR(x2, x6, x10, x14); // column 3
			QR(x3, x7, x11, x15); // column 4
			// Even round
			QR(x0, x5, x10, x15); // diagonal 1 (main diagonal)
			QR(x1, x6, x11, x12); // diagonal 2
			QR(x2, x7, x8, x13); // diagonal 3
			QR(x3, x4, x9, x14); // diagonal 4
		}
		if ((x0 + in0) == c0)
			if ((x1 + in1) == c1)
				if ((x2 + in2) == c2)
					if ((x3 + in3) == c3)
						if ((x4 + in4) == c4)
							if ((x5 + in5) == c5)
								if ((x6 + in6) == c6)
									if ((x7 + in7) == c7)
										if ((x8 + in8) == c8)
											if ((x9 + in9) == c9)
												if ((x10 + in10) == c10)
													if ((x11 + in11) == c11)
														if ((x12 + in12) == c12)
															if ((x13 + in13) == c13)
																if ((x14 + in14) == c14)
																	if ((x15 + in15) == c15)
																		printf("The key is %x %x\n", threadIndex, j);
	}
}
__global__ void ChaCha_exhaustive_final(bit32 ciphertext[16], __int64 trial) {

	int i; bit32 in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15;
	bit32 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	bit32 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	c0 = ciphertext[0];
	c1 = ciphertext[1];
	c2 = ciphertext[2];
	c3 = ciphertext[3];
	c4 = ciphertext[4];
	c5 = ciphertext[5];
	c6 = ciphertext[6];
	c7 = ciphertext[7];
	c8 = ciphertext[8];
	c9 = ciphertext[9];
	c10 = ciphertext[10];
	c11 = ciphertext[11];
	c12 = ciphertext[12];
	c13 = ciphertext[13];
	c14 = ciphertext[14];
	c15 = ciphertext[15];
	for (__int64 j = 0; j < trial; j++) {
		in0 = 0x61707865;
		in1 = 0x3320646e;
		in2 = 0x79622d32;
		in3 = 0x6b206574;
		in4 = threadIndex;
		in5 = j;
		in6 = 0;	in7 = 0;	in8 = 0;	in9 = 0;	in10 = 0;	in11 = 0;	in12 = 0;	in13 = 0;	in14 = 0;	in15 = 0;
		x0 = in0;
		x1 = in1;
		x2 = in2;
		x3 = in3;
		x4 = in4;
		x5 = in5;
		x6 = in6;
		x7 = in7;
		x8 = in8;
		x9 = in9;
		x10 = in10;
		x11 = in11;
		x12 = in12;
		x13 = in13;
		x14 = in14;
		x15 = in15;
		// 10 loops × 2 rounds/loop = 20 rounds
#pragma unroll
		for (i = 0; i < ROUNDS; i += 2) {
			// Odd round
			QR(x0, x4, x8, x12); // column 1
			QR(x1, x5, x9, x13); // column 2
			QR(x2, x6, x10, x14); // column 3
			QR(x3, x7, x11, x15); // column 4
			// Even round
			QR(x0, x5, x10, x15); // diagonal 1 (main diagonal)
			QR(x1, x6, x11, x12); // diagonal 2
			QR(x2, x7, x8, x13); // diagonal 3
			QR(x3, x4, x9, x14); // diagonal 4
		}
		if ((x0 + in0) == c0)
			if ((x1 + in1) == c1)
				if ((x2 + in2) == c2)
					if ((x3 + in3) == c3)
						if ((x4 + in4) == c4)
							if ((x5 + in5) == c5)
								if ((x6 + in6) == c6)
									if ((x7 + in7) == c7)
										if ((x8 + in8) == c8)
											if ((x9 + in9) == c9)
												if ((x10 + in10) == c10)
													if ((x11 + in11) == c11)
														if ((x12 + in12) == c12)
															if ((x13 + in13) == c13)
																if ((x14 + in14) == c14)
																	if ((x15 + in15) == c15)
																		printf("The key is %x %x\n", threadIndex, j);
	}
}
__global__ void ChaCha_linear_3round(bit32 plaintext[16], __int64 counter[], bit64 trial) {
	int i;	bit32 x[16]; bit32 in[16];
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit32 v;
#pragma unroll
	for (i = 0; i < 16; i++) x[i] = plaintext[i];
	x[10] ^= threadIndex;
	x[11] ^= (trial >> 32);
	x[12] ^= (trial & 0xffffffff);

	for (bit32 j = 0; j < TRIALS; j++) {
#pragma unroll
		for (i = 0; i < 16; i++) in[i] = x[i];
#pragma unroll
		for (i = 0; i < 16; i++) x[i] = in[i];		
			// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 4
/*		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]);*/ // diagonal 4



//		v = (in[3] & 0x2) ^ (x[3] & 0x01001901) ^ (x[7] & 0x06C02080) ^ (x[11] & 0x05840000) ^ (x[15] & 0x06191861); 
//		v = (in[3] & 0x1) ^ (x[3] & 0x01001901) ^ (x[7] & 0x06C02080) ^ (x[11] & 0x05840000) ^ (x[15] & 0x06191861); 
//		v = (in[3] & 0xade0b876) ^ x[3] & 0x01001901 ^ x[7] & 0x06C02080 ^ x[11] & 0x05840000 ^ x[15] & 0x06191861;

		v = (in[3] & 0x1) ^ (x[0] & 0xD81918D9) ^ (x[1] & 0x00CC00CC) ^ (x[2] & 0x10000) ^ (x[3] & 0x01001901) 
			^ (x[4] & 0x08C400CC) ^ (x[5] & 0xC20C6000) ^ (x[6] & 0x80)	^ (x[7] & 0xC2082080) 
			^ (x[8] & 0x18180800) ^ (x[9] & 0x01841040)	^ (x[11] & 0x05840000) 
			^ (x[12] & 0xC00D58D9)^ (x[13] & 0xC01800C0)^ (x[14] & 0x01000101) ^ (x[15] & 0x06191861);
		
		v ^= v >> 1;
		v ^= v >> 2;
		v = (v & 0x11111111U) * 0x11111111U;
		v = (v >> 28) & 1;
		if (v == 0) counter[threadIndex]++;
	}
}
__global__ void ChaCha_differential_4round(bit32 plaintext[16], __int64 counter[], bit64 trial) {
	int i;	bit32 x[16]; bit32 in[16];
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit32 v;
#pragma unroll
	for (i = 0; i < 16; i++) x[i] = plaintext[i];
	x[10] ^= threadIndex;
	x[11] ^= (trial >> 32);
	x[12] ^= (trial & 0xffffffff);

	for (bit32 j = 0; j < TRIALS; j++) {
#pragma unroll
		for (i = 0; i < 16; i++) in[i] = x[i];
#pragma unroll
		for (i = 0; i < 16; i++) x[i] = in[i];
#pragma unroll
		for (i = 0; i < 12; i++) x[i] = plaintext[i];
		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 4
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4



		v = x[3];

#pragma unroll
		for (i = 0; i < 16; i++) x[i] = in[i];
#pragma unroll
		for (i = 0; i < 12; i++) x[i] = plaintext[i];
		x[12] ^= 0x1; // introduce the difference
//		x[15] ^= 0x00000800; // best difference from our experiments
		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 4
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4

		v ^= x[3];

		if ((v&0x1) == 0x1) counter[threadIndex]++;
//		if (v == 0x1) counter[threadIndex]++;
	}
}
__global__ void ChaCha_differential_4round_RandomInput(bit32 input[], __int64 counter[]) {
	int i;	bit32 x[16]; bit32 in[16];
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit32 v;
	for (i = 0; i < 12; i++) x[i+4] = input[i + (threadIndex * 12)];
	x[0] = 0x61707865;
	x[1] = 0x3320646e;
	x[2] = 0x79622d32;
	x[3] = 0x6b206574;
	for (bit32 j = 0; j < TRIALS; j++) {
		for (i = 0; i < 8; i++) x[i+4] = input[i + (threadIndex * 12)];
		x[0] = 0x61707865;
		x[1] = 0x3320646e;
		x[2] = 0x79622d32;
		x[3] = 0x6b206574;


#pragma unroll
		for (i = 0; i < 16; i++) in[i] = x[i];


		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 4
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
		v = x[3];
#pragma unroll
		for (i = 0; i < 16; i++) x[i] = in[i];
		x[12] ^= 0x1; // introduce the difference

		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 4
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4

		v ^= x[3];
		if ((v & 0x1) == 1) counter[threadIndex]++;
	}
}
__global__ void ChaCha_differential_3round_RandomInput(bit32 input[], __int64 counter[]) {
	int i;	bit32 x[16]; bit32 in[16];
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit32 v;
	for (i = 0; i < 12; i++) x[i + 4] = input[i + (threadIndex * 12)];
	x[0] = 0x61707865;
	x[1] = 0x3320646e;
	x[2] = 0x79622d32;
	x[3] = 0x6b206574;
	for (bit32 j = 0; j < TRIALS; j++) {
		for (i = 0; i < 8; i++) x[i + 4] = input[i + (threadIndex * 12)];
		x[0] = 0x61707865;
		x[1] = 0x3320646e;
		x[2] = 0x79622d32;
		x[3] = 0x6b206574;


#pragma unroll
		for (i = 0; i < 16; i++) in[i] = x[i];


		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4


		v = x[3]^x[4];
#pragma unroll
		for (i = 0; i < 16; i++) x[i] = in[i];
		x[14] ^= 0x40; // introduce the difference

		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4



		v ^= (x[3]^x[4]);
		if ((v & 0x1) == 0) counter[threadIndex]++;
	}
}
__global__ void ChaCha_differential_4round_AutomaticSearch(bit32 input[], __int64 counter[], int word, int bit) {
	int i;	bit32 x[16]; bit32 in[16];
	bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	bit32 v;
	for (i = 0; i < 12; i++) x[i + 4] = input[i + (threadIndex * 12)];
	x[0] = 0x61707865;
	x[1] = 0x3320646e;
	x[2] = 0x79622d32;
	x[3] = 0x6b206574;
	for (bit64 j = 0; j < (bit64)TRIALS; j++) {
		for (i = 0; i < 8; i++) x[i + 4] = input[i + (threadIndex * 12)];
		x[0] = 0x61707865;
		x[1] = 0x3320646e;
		x[2] = 0x79622d32;
		x[3] = 0x6b206574;
#pragma unroll
		for (i = 0; i < 16; i++) in[i] = x[i];
		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 4
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4



		v = x[3];
#pragma unroll
		for (i = 0; i < 16; i++) x[i] = in[i];
		x[word] ^= (1<<bit); // introduce the difference

		// Odd round 1
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 2
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4
			// Odd round 3
		QR(x[0], x[4], x[8], x[12]); // column 1
		QR(x[1], x[5], x[9], x[13]); // column 2
		QR(x[2], x[6], x[10], x[14]); // column 3
		QR(x[3], x[7], x[11], x[15]); // column 4
			// Even round 4
		QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
		QR(x[1], x[6], x[11], x[12]); // diagonal 2
		QR(x[2], x[7], x[8], x[13]); // diagonal 3
		QR(x[3], x[4], x[9], x[14]); // diagonal 4


		v ^= x[3];
		if ((v & 0x1) == 0) counter[threadIndex]++;
//		if (v & 0x80000000) counter[threadIndex]++;
	}
}

int main2() {	
	bit32 plaintext[16] = { 0 }, ciphertext[16] = { 0 };
	bit32 ciphertext2[16] = { 0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653, 0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b, 0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8, 0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2 };
	plaintext[0] = 0x61707865;	plaintext[1] = 0x3320646e;	plaintext[2] = 0x79622d32;	plaintext[3] = 0x6b206574;
	chacha_block(ciphertext, plaintext);
//	for (int i = 0; i < 16; i++) printf("0x%08x, ",ciphertext[i]);
	bit32* cp;
	cudaMallocManaged(&cp, 16 * sizeof(bit32));
	for (int i = 0; i < 16; i++) cp[i] = ciphertext2[i];
	cudaDeviceSynchronize();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
//	ChaCha_exhaustive << <BLOCKS, THREADS >> > (cp);
//	ChaCha_exhaustive_noarrays << <BLOCKS, THREADS >> > (cp);
	ChaCha_exhaustive_noarrays2 << <BLOCKS, THREADS >> > (cp);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f milliseconds\n", milliseconds);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	return 0;
}
int main3() {
	bit32 plaintext[16] = { 0 }, ciphertext[16] = { 0 };
	bit32 ciphertext2[16] = { 0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653, 0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b, 0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8, 0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2 };
	plaintext[0] = 0x61707865;	plaintext[1] = 0x3320646e;	plaintext[2] = 0x79622d32;	plaintext[3] = 0x6b206574;
	chacha_block(ciphertext, plaintext);
	//	for (int i = 0; i < 16; i++) printf("0x%08x, ",ciphertext[i]);
	bit32* cp;
	__int64* counter; 
	__int64* counter_d, total_counter = 0, bias, experiment;
	bit64 trial = 1;

	printf("Trial = 2^30 +  ");
	scanf_s("%lld", &trial);
	trial = (bit64)1 << trial;

	experiment = TRIALS * trial * THREADS * BLOCKS;
	counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
	cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));
	cudaMallocManaged(&cp, 16 * sizeof(bit32));
	for (int i = 0; i < 16; i++) cp[i] = ciphertext2[i];
	cudaDeviceSynchronize();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
	for (int i = 0; i < trial; i++) {
//		ChaCha_linear_3round << <BLOCKS, THREADS >> > (cp, counter_d,trial);
		ChaCha_differential_4round << <BLOCKS, THREADS >> > (cp, counter_d, trial); // wrong somehow
//		cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
//		cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f milliseconds\n", milliseconds);

	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	printf("Total counter: %I64d Bias: %I64d\n", total_counter, bias);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	return 0;
}
int main4() {
	bit32* nonce, *nonce_d;
	__int64* counter, *counter_d, total_counter = 0, bias, experiment;
	float milliseconds = 0;
	nonce = (bit32*)calloc(BLOCKS * THREADS * 12, sizeof(bit32));
	experiment = (__int64)TRIALS * (__int64)THREADS * (__int64)BLOCKS;
	counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
	for (int j = 0; j < THREADS * BLOCKS * 12; j++) { rdrand_32(nonce + j, 0);  }
	cudaMalloc((void **)&nonce_d, BLOCKS * THREADS * 12 * sizeof(bit32));
	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 12 * sizeof(bit32), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
	

//	ChaCha_differential_4round_RandomInput << <BLOCKS, THREADS >> > (nonce_d, counter_d);
	ChaCha_differential_3round_RandomInput << <BLOCKS, THREADS >> > (nonce_d, counter_d);

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds\n", milliseconds);

	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	printf("Total counter: %I64d Bias: %I64d\n", total_counter, bias);
	free(nonce); cudaFree(nonce_d); printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	return 0;
}
int main_last() {
	bit32* nonce, * nonce_d;
	__int64* counter, * counter_d, total_counter = 0, bias, experiment;
	float milliseconds = 0;
	nonce = (bit32*)calloc(BLOCKS * THREADS * 12, sizeof(bit32));
	experiment = (__int64)TRIALS * (__int64)THREADS * (__int64)BLOCKS;	
	for (int j = 0; j < THREADS * BLOCKS * 12; j++) { rdrand_32(nonce + j, 0); }
	cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 12 * sizeof(bit32));
	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 12 * sizeof(bit32), cudaMemcpyHostToDevice);



	for (int word = 12; word < 16; word++) {
		for (int bit = 0; bit < 32; bit++) {
			total_counter = 0;
			
			counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
			cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));
			cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();
			cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
			ChaCha_differential_4round_AutomaticSearch << <BLOCKS, THREADS >> > (nonce_d, counter_d, word, bit);

			cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
			cudaEventRecord(stop);	cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);	printf("%d %d: Time elapsed: %f milliseconds ", word, bit, milliseconds);

			for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
			bias = (experiment) / 2 - total_counter;
			printf("Total counter: %I64d Bias: %I64d\n", total_counter, bias);
			free(counter); cudaFree(counter_d);
		}
	}
//	int word = 15;
//	int bit = 11;

/*	int word = 12;
	int bit = 0;

	total_counter = 0;

	counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
	cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
	ChaCha_differential_4round_AutomaticSearch << <BLOCKS, THREADS >> > (nonce_d, counter_d, word, bit);

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("%d %d: Time elapsed: %f milliseconds ", word, bit, milliseconds);

	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	printf("Total counter: %I64d Bias: %I64d\n", total_counter, bias);
	free(counter); cudaFree(counter_d);*/


	free(nonce); cudaFree(nonce_d); printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	return 0;
}
int chacha_benchmark() {
	int trial = 0;
	printf("Trial = 2^17 +  ");
	scanf_s("%d", &trial);
	trial = (__int64)1 << trial;
	bit32 plaintext[16] = { 0 }, ciphertext[16] = { 0 };
	bit32 ciphertext2[16] = { 0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653, 0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b, 0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8, 0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2 };
	plaintext[0] = 0x61707865;	plaintext[1] = 0x3320646e;	plaintext[2] = 0x79622d32;	plaintext[3] = 0x6b206574;
	chacha_block(ciphertext, plaintext);
	//	for (int i = 0; i < 16; i++) printf("0x%08x, ",ciphertext[i]);
	bit32* cp;
	cudaMallocManaged(&cp, 16 * sizeof(bit32));
	for (int i = 0; i < 16; i++) cp[i] = ciphertext2[i];
	cudaDeviceSynchronize();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//	ChaCha_exhaustive << <BLOCKS, THREADS >> > (cp);
	//	ChaCha_exhaustive_noarrays << <BLOCKS, THREADS >> > (cp);
	ChaCha_exhaustive_final << <BLOCKS, THREADS >> > (cp,trial);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f milliseconds\n", milliseconds);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	return 0;
}
int main() {
	int choice = 0;
	printf(
		"(1) ChaCha_differential_4round_AutomaticSearch for [WGM24]\n"
		"(2) ChaCha_differential_3round for [CN21]\n"
		"(3) ChaCha_differential_4round\n"
		"(4) ChaCha_exhaustive_noarrays2\n"
		"(5) ChaCha_benchmark\n"
		"Choice: "
	);
	scanf_s("%d", &choice);
	if (choice == 1) main_last();
	else if (choice == 2) main4();
	else if (choice == 3) main3();
	else if (choice == 4) main2();
	else if (choice == 5) chacha_benchmark();
}

