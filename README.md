# elm-signatures

New paper on fake signature analysis using ELM model on pre-computed DL features. Side effect in improving `hpelm` repository.


### Summary of the previous paper:

Detect forged signatures by an analysis of local image content, cut by a sliding window going over a signature. 
Never use whole signature - large windows may include significant portion of the signature, but they are few and have less impact overall.
The method runs offline, and is built from off-the-shelf components.


### Goals of the current paper

Moonshot towards user-independent signature verification. A model should distinguish between user writing its own signature, and forging a signature - without seeing correct examples of signature being forged!

 - Test on users that model has never seen before
 - Forged signatures are written by users; try to remove forged signatures written by test users from training set
