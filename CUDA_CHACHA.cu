#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include "rdrand.h"
#include <intrin.h>
#include <immintrin.h>
#include <string.h>
#include <math.h>

#define RDRAND_MASK	0x40000000
#define RETRY_LIMIT 10
#ifdef _WIN64
typedef uint64_t _wordlen_t;
#else
typedef uint32_t _wordlen_t;
#endif

// When no user input is asked, the experiments are performed BLOCKS x THREADS x TRIALS times
// When the user asked for the input "trial", the experiments are performed BLOCKS x THREADS x 2^trial times

#define BLOCKS				256  // Kernel call creates 256 = 2^8 blocks of threads
#define THREADS				512  // Each kernel block has 512 = 2^9 threads
#define TRIALS				1024*64 // cannot be larger than 2^32

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

// rdrand functions are used to generate random numbers.
// These functions use the AES-NI instructions for fast random number generation.
// They would not work on CPUs that do not support AES-NI instructions.

// In these codes, we use rdrand functions to generate random numbers at the CPU side for each thread that we are going to run at the GPU kernel.
// Thus, as a first step we transfer these random numbers from RAM to GPU memory which allows us to run experiments with different random input every time.
// This way, when the GPU kernels are called, every thread uses a different random input.
// When the number of encryptions we are going to perform is larger than the number of threads in our GPU kernel, we can provide a new random input again and again in this way. 
// However, this would be slow due to memory copy operations between RAM and GPU global memory.
// Thus, once a thread performs encryption on the random input, the encryption output is fed back as an input for the same thread for performance. 

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
// chacha_block() performs ChaCha round functions on the input on CPU
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
// "ChaCha_exhaustive_final" kernel performs exhaustive key search just for benchmarking ChaCha20
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
// "ChaCha_differential_4round_RandomInput" checks the correctnes of the 4-round DL introduced in [WGM24]
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
		if ((v & 0x1) == 1) counter[threadIndex]++; // check the output mask
	}
}
// "ChaCha_differential_3round_RandomInput" checks the correctnes of the 4-round DL introduced in [CN21]
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
		if ((v & 0x1) == 0) counter[threadIndex]++; // check the output mask
	}
}
//  "ChaCha_differential_4round_AutomaticSearch" kernel checks the distinguisher of [WGM24] by introducing input difference to every 128 possible bits
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
		if ((v & 0x1) == 0) counter[threadIndex]++; // check the output mask
	}
}

int verify_the_DL_CN21() {
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

	ChaCha_differential_3round_RandomInput << <BLOCKS, THREADS >> > (nonce_d, counter_d);

	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds\n", milliseconds);

	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	printf("\nTotal counter: %I64d\nDifference from the Expected Value: %I64d\nBias: 2^-%lf (For an experiment with 2^%lf data)\n", total_counter, bias, ((log(BLOCKS) + log(THREADS) + log(TRIALS)) / log(2)) - (log(abs(bias)) / log(2)), (log(BLOCKS) + log(THREADS) + log(TRIALS)) / log(2));
	free(nonce); cudaFree(nonce_d); printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	return 0;
}
int verify_the_DL_WGM24() {
	bit32* nonce, * nonce_d;
	__int64* counter, * counter_d, total_counter = 0, bias, experiment;
	float milliseconds = 0;
	nonce = (bit32*)calloc(BLOCKS * THREADS * 12, sizeof(bit32));
	experiment = (__int64)TRIALS * (__int64)THREADS * (__int64)BLOCKS;
	counter = (__int64*)calloc(BLOCKS * THREADS, sizeof(bit64));
	for (int j = 0; j < THREADS * BLOCKS * 12; j++) { rdrand_32(nonce + j, 0); }
	cudaMalloc((void**)&nonce_d, BLOCKS * THREADS * 12 * sizeof(bit32));
	cudaMemcpy(nonce_d, nonce, BLOCKS * THREADS * 12 * sizeof(bit32), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&counter_d, BLOCKS * THREADS * sizeof(bit64));
	cudaMemcpy(counter_d, counter, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
	ChaCha_differential_4round_RandomInput << <BLOCKS, THREADS >> > (nonce_d, counter_d);
	cudaMemcpy(counter, counter_d, BLOCKS * THREADS * sizeof(__int64), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds\n", milliseconds);

	for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
	bias = (experiment) / 2 - total_counter;
	printf("\nTotal counter: %I64d\nDifference from the Expected Value: %I64d\nBias: 2^%lf (For an experiment with 2^-%lf data)\n", total_counter, bias, ((log(BLOCKS) + log(THREADS) + log(TRIALS)) / log(2)) - (log(abs(bias)) / log(2)), (log(BLOCKS) + log(THREADS) + log(TRIALS)) / log(2));
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

	// Input difference will be applied to "word" at the "bit" position
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
			cudaEventElapsedTime(&milliseconds, start, stop);	printf("Input Difference -- Word: %d Bit: %d: Time elapsed: %f milliseconds ", word, bit, milliseconds);

			for (int i = 0; i < BLOCKS * THREADS; i++) total_counter += counter[i];
			bias = (experiment) / 2 - total_counter;
			printf("\nTotal counter: %I64d\nDifference from the Expected Value: %I64d\nBias: 2^-%lf (For an experiment with 2^%lf data)\n\n", total_counter, bias, ((log(BLOCKS) + log(THREADS) + log(TRIALS)) / log(2)) - (log(abs(bias)) / log(2)), (log(BLOCKS) + log(THREADS) + log(TRIALS)) / log(2));
			free(counter); cudaFree(counter_d);
		}
	}
	free(nonce); cudaFree(nonce_d); printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	return 0;
}
int chacha_benchmark() {
	int trial = 0;
	// In our GPU optimizations, we observed that the best performance is obtained when the GPU kernel is run with 256 blocks of 512 threads, resulting in 2^17 threads.
	// Each thread performs more than one encryption when the requested number of encryptions exceeds 2^17. 
	// Using this configuration, we achieved 2^34.92 20-round ChaCha encryptions per second on an RTX 4090.
	printf("Enter number of trials you want to perform\n");
	printf("Enter an integer value in [0,32]\n");
	printf("Trial = 2^17 +  ");
	scanf_s("%d", &trial);
	trial = (__int64)1 << trial;
	// e.g. User input 0 here means 2^17 encryptions
	// e.g. User input 20 here means 2^37 encryptions
	bit32 plaintext[16] = { 0 }, ciphertext[16] = { 0 };
	bit32 ciphertext2[16] = { 0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653, 0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b, 0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8, 0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2 };
	plaintext[0] = 0x61707865;	plaintext[1] = 0x3320646e;	plaintext[2] = 0x79622d32;	plaintext[3] = 0x6b206574;
	chacha_block(ciphertext, plaintext);
	bit32* cp;
	cudaMallocManaged(&cp, 16 * sizeof(bit32));
	for (int i = 0; i < 16; i++) cp[i] = ciphertext2[i];
	cudaDeviceSynchronize();
	cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
	// Start the GPU kernel
	ChaCha_exhaustive_final << <BLOCKS, THREADS >> > (cp,trial);
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	// Measure the time taken by the GPU kernel
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f milliseconds for 2^%lf brute force key trials\n", milliseconds, ((log(BLOCKS)+log(THREADS)+log(trial))/log(2)));
	printf("2^%lf key trials per second\n", ((log(BLOCKS) + log(THREADS) + log(trial)) / log(2))-(log(milliseconds/1000))/log(2));
	printf("%s\n", cudaGetErrorString(cudaGetLastError())); // Prints the last error on the GPU side
	return 0;
}
int main() {
	int choice = 0;
	printf(
		"(1) ChaCha_differential_4round_AutomaticSearch for [WGM24]\n"
		"(2) ChaCha_differential_3round for [CN21]\n"
		"(3) ChaCha_differential_4round for [WGM24]\n"
		"(4) ChaCha_benchmark\n"
		"Choice: "
	);
	scanf_s("%d", &choice);
	if (choice == 1) main_last();
	else if (choice == 2) verify_the_DL_CN21();
	else if (choice == 3) verify_the_DL_WGM24();
	else if (choice == 4) chacha_benchmark();
}

