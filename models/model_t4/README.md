# Overview

Difference from model_t0 is that model_t1 predicts cartesian coordinates x,y,z and direction dx,dy,dz

Since we are competing against SGP4 which actually only has cartesian output, we should also be predicting it!  This allows the loss function to do a much more spatial error than using keplerian elements (since 1 degree difference can be huge or 0 depending on other elements)