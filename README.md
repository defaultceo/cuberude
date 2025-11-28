# cuberude
Fast approximations for ∛(a³ + b³) using SIMD-optimized methods.  
FARCe (fast) & Hybrid (accuracy)

Installation  
#include "cuberude.h"

Place cuberude.h in your project’s include path.

Dependencies  
"Requires AVX2-capable compiler (GCC, Clang, MSVC)."

Build Notes  
"No build required — header-only."

Usage  
All functions follow this pattern:  
	void cuberude_farce_avx2_ps(float* a, float* b, float* result, size_t size);

Input:  Vectors a, b of length size  
Output: result vector filled with approximated ∛(a³ + b³)

## Benchmarks
|  Function                       	| Throughput (GElements/s)  |
| ----------------------------------|---------------------------|
|   Exact_f  (std::cbrt)*         	|  0.04 					|
|   farce_f 			  	        |  0.12						|
|   farce_d   			  	        |  0.07						|
|   hybrid_f 			  	        |  0.05						|
|   hybrid_d   			  	        |  0.05						|
|   farce_avx2_f 		  	        |  0.37 					|
|   farce_avx2   		  	        |  0.19 					|
|   hybrid_avx2_f 		  	        |  0.20						|
|   hybrid_avx2   		  	        |  0.10						|

*The Exact_f function has no implementation of ours; Uses std::cbrt(a*a*a + b*b*b);
 Comparing throughput with the standard library.


IEEE compliance
## Float Test Results
| Case | Input (a, b) | Expected   | Exact_f 	| Farce_f         | Hybrid_f   	  |  Hybrid_AVX2_f 	| Farce_AVX2_f 	  |
|------|--------------|------------|----------|-----------------|---------------|-----------------|-----------------|
| 0    |    (∞, -∞)   | NaN        |  NaN ✅  |  NaN ✅        | NaN ✅ 	      |  NaN ✅     	  |  NaN ✅         |
| 1    |    (NaN, 1)  | NaN        |  NaN ✅  |  NaN ✅        | NaN ✅ 	      |  NaN ✅ 	      |  NaN ✅         |
| 2    |    (NaN, ∞)  | NaN        |  NaN ✅  |  NaN ✅        | NaN ✅ 	      |  NaN ✅ 	      |  NaN ✅         |
| 3    |    (0, -0)   | ±0         | +0.0 ✅ 	| +0.0 ✅        | +0.0 ✅ 	      | +0.0 ✅         | +0.0 ✅         |
| 4    |    (0, 0)    | +0         | +0.0 ✅ 	| +0.0 ✅        | +0.0 ✅ 	      | +0.0 ✅         | +0.0 ✅         |
| 5    |    (−0, 0)   | ±0         | +0.0 ✅ 	| +0.0 ✅        | +0.0 ✅ 	      | +0.0 ✅         | +0.0 ✅         |
| 6    |    (−0, −0)  | −0         | +0.0 ❌ 	| −0.0 ✅        | −0.0 ✅ 	      | -0.0 ✅         | -0.0 ✅         |
| 7    |(1e-38,-1e-38)| ±0         | +0.0 ✅ 	| +0.0 ✅        | +0.0 ✅ 	      | +0.0 ✅         | +0.0 ✅         |
| 8    |    (subn, 0) | subn       | +0.0 ❌ 	| −1.4013e-45 ✅ | −1.4013e-45 ✅ | −1.4013e-45  ✅ | −1.4013e-45 ✅  |
| 9    |(subn+, subn+)| 2.8026e-45 | +0.0 ❌ 	|  2.8026e-45 ✅ | 2.8026e-45  ✅ |  2.8026e-45  ✅ |  2.8026e-45  ✅ |
| 10   |(subn-, subn-)|-2.8026e-45 | +0.0 ❌ 	| −2.8026e-45 ✅ | −2.8026e-45 ✅ | -2.8026e-45  ✅ | -2.8026e-45  ✅ |
| 11   |(subn-, subn+)| ±0         | +0.0 ✅ 	| +0.0 ✅        | +0.0 ✅ 	      | +0.0 ✅         | +0.0 ✅         |
| 12   |   (subn, 4)  | ~4         |  4   ✅  |  4   ✅        |  4   ✅        |  4   ✅         |  4   ✅         |
| 13   |(FLT_MAX,-FLT)| ±0         |  NaN ❌  | +0.0 ✅        | +0.0 ✅ 	      | +0.0 ✅         | +0.0 ✅         |
| 14   |   (∞, 1e30)  | +Inf       | +Inf ✅ 	| +Inf ✅        | +Inf ✅ 	      | +Inf ✅         | +Inf ✅         |
| 15   |   (∞, 1)     | +Inf       | +Inf ✅ 	| +Inf ✅        | +Inf ✅ 	      | +Inf ✅         | +Inf ✅         |
| 16   |(1e-40, 1e40) | +Inf       | +Inf ✅ 	| +Inf ✅        | +Inf ✅ 	      | +Inf ✅         | +Inf ✅         |
| 17   |(1e-38, 1e38) | 1e38       | +Inf ❌ 	| 1e+38 ✅       | 1e+38 ✅ 	    | 1e+38 ✅        | 1e+38 ✅        |


✅ = Correct result per IEEE 754 edge-case rules
❌ = Deviation from expected behavior

License
SPDX-License-Identifier: AGPL-3.0-only OR Commercial


Contact: defaultceo@atomicmail.io 
