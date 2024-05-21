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
pythia.readString("HadronLevel:all = on")

### Additional parameters
pythia.readString("ParticleDecays:limitTau = on")
pythia.readString("ParticleDecays:tauMax = 10")
pythia.readString("6:onMode = on")
pythia.readString("-6:onMode = on")
pythia.readString("StringZ:rFactB = 1.056")
pythia.readString("Main:timesAllowErrors = 500")
pythia.readString("PartonLevel:MPI = on")
pythia.readString("24:mayDecay = on")
pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 2")


# Define jet clustering parameters
R = 0.4  # Jet radius
min_pT = 30.0
max_eta = 2.4
jet_def = pythia8.SlowJet(-1, R, min_pT, max_eta)

# Initialize, incoming pp beams are default.
pythia.init()

import numpy as np

#pseudrapidity function
def pseudorapidity(px, py, pz):
    p = np.sqrt(np.power(px, 2) + np.power(py, 2) + np.power(pz, 2))
    if (p-pz) == 0.0:
        raise Exception("Error calculating pseudorapidity (divide by zero)")
    elif ((p+pz)/(p-pz)) <= 0.0:
        raise Exception("Error calculating pseudorapidity (log of negative number)")
    else:
        pseudorapidity = 0.5*np.log((p+pz)/(p-pz))
        return pseudorapidity

# num events to process
N = 10000
max_jets = 10

# Begin event loop. Generate event. 
# Skip if error. List first one.

# Initialize arrays to store jet multiplicities
nJets = []
jets_4vectors = []

for iEvent in range(0, N):
    if not pythia.next(): 
        continue

    # jet multiplicity
    # Find number of jets with given conditions.
    jet_def.analyze(pythia.event)

    nJet = jet_def.sizeJet()
    nJets.append(nJet)

    # Extract the 4-vectors for each jet
    event_jets = []
    for i in range(max_jets):
        if i < nJet:
       	    pT = jet_def.pT(i)
            p = jet_def.p(i)
            px = p.px()
            py = p.py()
            pz = p.pz()
            eta = pseudorapidity(px, py, pz)
            phi = jet_def.phi(i)
            mass = jet_def.m(i)
            energy = p.e()
            rapidity = jet_def.y(i)
            jet_4vector = [pT, rapidity, phi, mass, eta, energy]
        else:
            jet_4vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        event_jets.append(jet_4vector)

    jets_4vectors.append(event_jets)

# End of event loop. Statistics. Histogram. Done.
pythia.stat();


np.save('jet_multiplicity_hvq.npy', nJets)

print(np.shape(nJets))

print(nJets[:20])

np.save('jet_4vectors_hvq.npy', jets_4vectors)

print(np.shape(jets_4vectors))

print(jets_4vectors[:20], sep='\n')

