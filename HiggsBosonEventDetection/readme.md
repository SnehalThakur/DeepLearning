**Searching for Higgs Boson Decay Modes with Deep Learning**

The ATLAS experiment and the CMS experiment recently claimed the discovery of the Higgs boson. The discovery was acknowledged by the 2013 Nobel prize in physics given to Franc¸ois Englert and Peter Higgs. This particle was theorized almost 50 years ago to have the role of giving mass to other elementary particles. It is the final ingredient of the Standard Model of particle physics, ruling subatomic particles and forces. The experiments are running at the Large Hadron Collider (LHC) at CERN (the European Organization for Nuclear Research), Geneva, which began operating in 2009 after about 20 years of design and construction, and which will continue
operating for at least the next 10 years.
The Higgs boson has many different processes through which it can decay. When it decays, it produces other particles. In physics, a decay into specific particles is called a channel. The Higgs boson has been seen first in three distinct decay channels which are all boson pairs. One of the next important topics is to seek evidence on the decay into fermion pairs, namely tau-leptons or b-quarks, and to precisely measure their characteristics. The first evidence of the H to tau tau channel was recently reported by the ATLAS experiment [3], which, in the rest of this paper, will be referred to as ”the reference document”. The subject of the Challenge is to try and improve on this analysis.


**About Dataset**

Dataset of 250000 events, with an ID column, 30 feature columns, a weight column and a label 
column.

Some details to get started:

• all variables are floating point, except PRI_jet_num which is integer

• variables prefixed with PRI (for PRImitives) are “raw” quantities about the bunch collision as 
measured by the detector.

• variables prefixed with DER (for DERived) are quantities computed from the primitive features, 
which were selected by the physicists of ATLAS

• it can happen that for some entries some variables are meaningless or cannot be computed; in
this case, their value is −999.0, which is outside the normal range of all variables


Kaggle Competition
https://www.kaggle.com/competitions/higgs-boson
