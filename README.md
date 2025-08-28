# CUDA_CHACHA

This CUDA Optimization of **ChaCha** is used in the ToSC publication _Cryptanalysis: Theory versus Practice - Correcting Cryptanalysis Results on ASCON, ChaCha, and SERPENT using GPUs_ by Cihangir Tezcan, Gregor Leander, and Hosein Hadipour.

You can use benchmark to see how many ChaCha encryptions your GPU can perform in a second. These codes allowed us to perform 2^{34.92} ChaCha encryptions per second on an RTX 4090. 

We used these codes to experimentally verify the theoretically obtained best 7 and 7.25-round differential-linear distinguishers on ChaCha which have claimed biases of 2^{-16.6} and 2^{-19.93}, respectively. We performed many experiments using 2^{50} data and concluded that the claimed distinguishers cannot be used as a distinguisher to distinguish 7 or 7.25 rounds of ChaCha from a random permutation.
