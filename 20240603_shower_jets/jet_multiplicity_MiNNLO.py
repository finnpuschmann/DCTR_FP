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
import argparse
import numpy as np

# madgraph import
from madgraph.various.lhe_parser import EventFile

cfg = open("Makefile.inc")
lib = "../lib"
for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
sys.path.insert(0, lib)

# parse cli arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which LHE File to Open")
    # LHE arg
    parser.add_argument("-l", "--lhe", help="String. Which MiNNLO LHE File to open. Values between 1 and 100. Default = '1000'", type = str, default = '1000')
    # NUM arg
    parser.add_argument("-n", "--num", help="Int. Number of events to shower. Default = 10000", type = int, default = 10000)
    args = parser.parse_args()
    LHE = args.lhe
    NUM = args.num
else:
    LHE = '1000'
    NUM = 10000

# import pdb to debug
import pdb
import uproot_methods

# Import the Pythia module.
import pythia8
pythia = pythia8.Pythia()

lhe_file = f'/nfs/dust/cms/user/amoroso/powheg/POWHEG-BOX-V2/ttJ_MiNNLOPS_v1.0_beta1/decay-ll/pwgevents-{LHE}.lhe'
pythia.readString("Beams:frameType = 4") # read info from a LHEF
pythia.readString(f'Beams:LHEF = {lhe_file}') # the LHEF to read from
print(f'Using LHE File: {lhe_file}')

# Veto Settings # # Veto Settings # https://github.com/cms-sw/cmssw/blob/master/Configuration/Generator/python/Pythia8PowhegEmissionVetoSettings_cfi.py
pythia.readString("SpaceShower:pTmaxMatch = 1")
pythia.readString("TimeShower:pTmaxMatch = 1")

'''
pythia.readString('POWHEG:veto = 1')
pythia.readString('POWHEG:pTdef = 1')
pythia.readString('POWHEG:emitted = 0')
pythia.readString('POWHEG:pTemt = 0')
pythia.readString('POWHEG:pThard = 0')
pythia.readString('POWHEG:vetoCount = 100')
'''

# CP5 tune # https://github.com/cms-sw/cmssw/blob/1234e950f3ee35b6f39abdeba60b2d1a53c0c891/Configuration/Generator/python/MCTunes2017/PythiaCP5Settings_cfi.py#L4
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

# Common settings CFI # https://github.com/cms-sw/cmssw/blob/master/Configuration/Generator/python/Pythia8CommonSettings_cfi.py

# pythia.readString('Tune:preferLHAPDF = 2')
# pythia.readString('Main:timesAllowErrors = 10000')
# pythia.readString('Check:epTolErr = 0.01')
# pythia.readString('Beams:setProductionScalesFromLHEF = off')
# pythia.readString('SLHA:minMassSM = 1000.')
# pythia.readString('ParticleDecays:limitTau0 = on')
# pythia.readString('ParticleDecays:tau0Max = 10')
# pythia.readString('ParticleDecays:allowPhotonRadiation = on')


### Additional parameters

pythia.readString("ParticleDecays:limitTau = on")
pythia.readString("ParticleDecays:tauMax = 10")
pythia.readString("6:onMode = on")
pythia.readString("-6:onMode = on")
pythia.readString("StringZ:rFactB = 0.855")
pythia.readString("Main:timesAllowErrors = 500")
pythia.readString("PartonLevel:MPI = on")
pythia.readString("HadronLevel:all = on")
pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 1")

# MiNNLO parameter
pythia.readString("SpaceShower:dipoleRecoil = on")

# Initialize, incoming pp beams are default.
pythia.init()


# Initialize, incoming pp beams are default.
pythia.init()

# Define jet clustering parameters
R = 0.4  # Jet radius
min_pT = 30.0
max_eta = 2.4
jet_def = pythia8.SlowJet(-1, R, min_pT, max_eta)

# pseudrapidity function
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
N = NUM
max_jets = 20
theta = 1 # for MiNNLO 

# Begin event loop. Generate event.
# Skip if error. List first one.

# Initialize arrays to store jet multiplicities
nJets = []
jets_4vectors = []

# init particle vector array
P0 = []

# use madgraph lhe_parser EventFile, because I don't know how to get event weights from pythia
lhe = EventFile(lhe_file)

# check that number of events in lhe file is >= N, since there were missmathces in the past
wgts_list = []
for event in lhe:
    wgts_list.append(event.wgt)

if len(wgts_list) <= N:
    N = len(lhe)


for iEvent in range(N):
    if not pythia.next():
        continue
    # showering
    partVec = []
    TT = []
    top = None
    antitop = None  
    for particle in pythia.event:
        # selecting only last top
        if particle.id() == 6:
            top = particle
        if particle.id() == -6:
            antitop = particle

    patop = uproot_methods.TLorentzVector.from_ptetaphim(antitop.pT(), antitop.eta(), antitop.phi(), antitop.m())
    ptop = uproot_methods.TLorentzVector.from_ptetaphim(top.pT(), top.eta(), top.phi(), top.m())

    p_tt = ptop + patop

    wgt = wgts_list[iEvent]

                # [pt, y, phi, mass, eta, E, PID, w, theta]
                # [0 , 1, 2  , 3   , 4  , 5, 6  , 7, 8    ]
    partVec.append([p_tt.pt, p_tt.rapidity, p_tt.phi, p_tt.mass, p_tt.eta, p_tt.E, 0, wgt, theta])

    partVec.append([ptop.pt, ptop.rapidity, ptop.phi, ptop.mass, ptop.eta, ptop.E, 6, wgt, theta])
    partVec.append([patop.pt, patop.rapidity, patop.phi, patop.mass, patop.eta, patop.E, -6, wgt, theta])

    P0.append(partVec)

    # jet multiplicity
    # Find number of jets with given conditions.
    jet_def.analyze(pythia.event)

    nJet = jet_def.sizeJet()
    nJets.append([nJet, wgt])

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
pythia.stat()

# save shower
P0 = np.array(P0)
np.save(f'./output/MiNNLO/converted_lhe_MiNNLO_{LHE}_dileptonic.npy', P0)
print(f'{np.shape(P0) = }')

# save multiplicity and jet observables
np.save(f'./output/MiNNLO/jet_multiplicity_MiNNLO_{LHE}_dileptonic.npy', nJets)
print(f'{np.shape(nJets) = }')

np.save(f'./output/MiNNLO/jet_4vectors_MiNNLO_{LHE}_dileptonic.npy', jets_4vectors)
print(f'{np.shape(jets_4vectors) = }')
