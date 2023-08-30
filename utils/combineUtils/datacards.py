simple_datacard = """
imax * number of channels
jmax * number of backgrounds
kmax * number of nuisance parameters (sources of systematical uncertainties)

shapes * * model.root $PROCESS
----
bin bin1
observation -1
bin           bin1      bin1  
process		    sig     bkg  
process		     0         1    
rate		     -1         -1
----
* autoMCStats 0
"""