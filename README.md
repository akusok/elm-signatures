# elm-signatures

New paper on fake signature analysis using ELM model on pre-computed DL features. Side effect in improving `hpelm` repository.


### Summary of the previous paper:

Detect forged signatures by an analysis of local image content, cut by a sliding window going over a signature. 
Never use whole signature - large windows may include significant portion of the signature, but they are few and have less impact overall.
The method runs offline, and is built from off-the-shelf components.


### Goals of the current paper

Moonshot towards user-independent signature verification. A model should distinguish between user writing its own signature, and forging a signature - without seeing correct examples of signature being forged!

The idea is to distinguish between a motor level writing of own well-trained signature, and a coginitive level forging of an another person's signature. The original GPDSS10000 paper generated data under these precise hypotheses; our model aims at solving a reverse problem of detecting the method a particular signature was written in. Final test will be performed on an actual signature data of 75 real people from MCYT75 dataset.

 - Validate on users that model has never seen before
 - Test on real people from MCYT75 dataset; use this dataset for testing only
