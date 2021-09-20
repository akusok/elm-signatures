# elm-signatures

New paper on fake signature analysis using ELM model on pre-computed DL features. Side effect in improving `hpelm` repository.


### Summary of the previous paper:

Detect forged signatures by an analysis of local image content, cut by a sliding window going over a signature. 
Never use whole signature - large windows may include significant portion of the signature, but they are few and have less impact overall.
The method runs offline, and is built from off-the-shelf components.


### Goals of the current paper

Moonshot towards user-independent signature verification. A model should distinguish between user writing its own signature, and forging a signature - without seeing correct examples of signature being forged!

The idea is to distinguish between a motor-level writing of own well-trained signature, and a coginitive-level forging of an another person's signature. The original GPDSS10000 paper generated data under these precise hypotheses; our model aims at solving a reverse problem of detecting the method a particular signature was written in. Final test will be performed on an actual signature data of 75 real people from MCYT75 dataset.

 - Validate on users that model has never seen before
 - Test on real people from MCYT75 dataset; use this dataset for testing only

A signature is defined by a set of morphological, static and dynamic parameters for one individual. The intra-personal variability is enabled by the variability variables, that are basically element-wise variances of all signature parameters for the same person over different writing tools and body postures during writing. Aging is simulated separately as simplified writing style by dropping nodes from letter signatures.

Forgeries are imitated in a similar way, but on a wider variation of inter-personal parameter variability. Creating a different writing speed profile is handled with a special care, as it tends to be too similar to the original one, resulting in an unrealistic forgeries. The process of forging includes detecting anchor points based on curvature as to simulate visual inspection of an original signature by the forger; 10% to 30% of those points are then dropped. The remaining points define stroke limits, changing the velocity profile of the original signature. 

### Experimental setup in the original paper:

Different datasets with incompatible results due to various protocols, data collection devices, and geographical areas. Two offline methods based on HMM with geometrical features, and SVM with local binary patterns; same configuration for all datasets to enhance comparison at the expence of per-dataset performance.

Experiments are repeated 10 times, report results with standard deviation on Equal Error Rates and DET curves.

- Take 5 random genuine signatures for training
- Remaining geunine signatures evaluate false rejection rate
- False acceptance rate comes from genuine signatures of all other users
- False acceptance rate on deliberate forgeries is computed separately, on attempted forgeries

