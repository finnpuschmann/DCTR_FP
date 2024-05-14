# main01.py is a part of the PYTHIA event generator.
# Copyright (C) 2022 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

# Keywords: basic usage; charged multiplicity; python;

# This is a simple test program. It fits on one slide in a talk.  It
# studies the charged multiplicity distribution at the LHC. To set the
# path to the Pythia 8 Python interface do either (in a shell prompt):
#      export PYTHONPATH=$(PREFIX_LIB):$PYTHONPATH
# or the following which sets the path from within Python.
#
# Use "python-config --include" to find the include directory and
# then configure Pythia ""--with-python-include=*".

import sys
cfg = open("Makefile.inc")
lib = "../lib"
for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
sys.path.insert(0, lib)


# import pdb to debug
import pdb
import uproot_methods

# Import the Pythia module.
import pythia8
pythia = pythia8.Pythia()
pythia.readString("Beams:frameType = 4") # read info from a LHEF
pythia.readString("Beams:LHEF = /nfs/dust/cms/user/vaguglie/simSetup/Box2/POWHEG-BOX-V2/hvq/testrun-tdec-lhc/Hdamp13TeV/BaseNom/Test/Results100/pwgevents.lhe") # the LHEF to read from
pythia.readString("SpaceShower:pTmaxMatch = 1")
pythia.readString("TimeShower:pTmaxMatch = 1.")

### CP5 tune
pythia.readString("Tune:pp = 14")
pythia.readString("Tune:ee = 7")
pythia.readString("MultipartonInteractions:ecmPow = 0.03344")
pythia.readString("MultipartonInteractions:bProfile = 2")
pythia.readString("MultipartonInteractions:pT0Ref = 1.41")
pythia.readString("MultipartonInteractions:coreRadius = 0.7634")
pythia.readString("MultipartonInteractions:coreFraction = 0.63")
pythia.readString("ColourReconnection:range = 5.176")
pythia.readString("SigmaTotal:zeroAXB = off")
pythia.readString("SpaceShower:alphaSorder = 2")
pythia.readString("SpaceShower:alphaSvalue = 0.118")
pythia.readString("SigmaProcess:alphaSvalue = 0.118")
pythia.readString("SigmaProcess:alphaSorder = 2")
pythia.readString("MultipartonInteractions:alphaSvalue = 0.118")
pythia.readString("MultipartonInteractions:alphaSorder = 2")
pythia.readString("TimeShower:alphaSorder = 2")
pythia.readString("TimeShower:alphaSvalue = 0.118")
pythia.readString("SigmaTotal:mode = 0")
pythia.readString("SigmaTotal:sigmaEl = 21.89")
pythia.readString("SigmaTotal:sigmaTot = 100.309")
pythia.readString("PDF:pSet = 20")
#pythia.readString("PDF:pSet=LHAPDF6.5.4:NNPDF31_nnlo_as_0118")

### Additional parameters
#pythia.readString("Top:gg2ttbar = on")  # Switch on process.
#pythia.readString("Top:qqbar2ttbar = on")
#pythia.readString("Beams:eCM = 13000.") # Set 14 TeV CM energy.
pythia.readString("ParticleDecays:limitTau = on")
pythia.readString("ParticleDecays:tauMax = 10")
#pythia.readString("Beams:idA = 2212")
#pythia.readString("Beams:idB = 2212")
pythia.readString("6:onMode = on")
pythia.readString("-6:onMode = on")
pythia.readString("StringZ:rFactB = 1.056")
pythia.readString("Main:timesAllowErrors = 500")
pythia.readString("PartonLevel:MPI = on")
pythia.readString("24:mayDecay = off")
pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 2")

#Read in commands from external file.
#pythia.readFile("/nfs/dust/cms/user/vaguglie/simSetup/Box2/POWHEG-BOX-V2/hvq/testrun-tdec-lhc/Hdamp13TeV/BaseNom/Test/Results100/pwgevents.lhe");

# Initialize, incoming pp beams are default.
pythia.init()

import numpy as np
N = 1000000
P0 = []
P1 = []
# Begin event loop. Generate event. Skip if error. List first one.
for iEvent in range(0, N):
    if not pythia.next(): continue
    #pythia.event.list()
    # Find number of all final charged particles and fill histogram.
    partVec = []
    TT = []
    top = None
    antitop = None  
    for prt in pythia.event:

        ##selecting only last top
        if prt.id() == 6: 
            top = prt
            #print("top E")
        if prt.id() == -6:
            antitop = prt
            #print("antitop E")


    partVec.append([top.pT(), top.y(), top.eta(), top.phi(), top.m(), 6])
    partVec.append([antitop.pT(), antitop.y(), antitop.eta(), antitop.phi(), antitop.m(), -6])
    patop = uproot_methods.TLorentzVector.from_ptetaphim(antitop.pT(), antitop.eta(), antitop.phi(), antitop.m())
    ptop = uproot_methods.TLorentzVector.from_ptetaphim(top.pT(), top.eta(), top.phi(), top.m())
    P0.append(partVec)
    
    p_tt = ptop + patop
    TT.append([p_tt.pt, p_tt.y, p_tt.eta, p_tt.phi, p_tt.mass, p_tt.E])
    P1.append(TT)


# End of event loop. Statistics. Histogram. Done.
pythia.stat();

P0 = np.array(P0)
P1 = np.array(P1)
np.savez_compressed('ShowerTop_1M.npz', a=P0)
np.savez_compressed('ShowerTT_1M.npz', a=P1)


print(P0.shape)
print(P1.shape)
