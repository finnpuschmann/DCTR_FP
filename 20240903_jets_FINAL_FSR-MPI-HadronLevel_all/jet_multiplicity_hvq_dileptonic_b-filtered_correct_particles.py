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

# import uproot_methods # deprecated
import vector

# Import the Pythia module.
import pythia8
# from pythia8 import deltaR

# madgraph import
from madgraph.various.lhe_parser import EventFile

cfg = open("/nfs/dust/cms/user/puschman/pythia8309/examples/Makefile.inc")
lib = "/nfs/dust/cms/user/puschman/pythia8309/lib"

for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
sys.path.insert(0, lib)

# parse cli arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="How many events to shower and get jets for")
    # LHE arg
    parser.add_argument("-l", "--lhe", help="String. Which hvq LHE File to open. Values between 1 and 1000. Default = '1'", type = str, default = '1')
    # NUM arg
    parser.add_argument("-n", "--num", help="Int. Number of events to shower. Default = 10000", type = int, default = 10000)
    # MIN_PT arg
<<<<<<< HEAD
    parser.add_argument("-p", "--pt", "--min_pt", help="Float. Minimum pt for jet finding algorithm. Default = 30", type = float, default = 30)
=======
    parser.add_argument("-p", "--pt", "--min_pt", help="Float. Minimum pt for jet finding algorithm. Default = 15.0", type = float, default = 15.0)
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8

    args = parser.parse_args()
    LHE = args.lhe
    NUM = args.num
    MIN_PT = args.pt
else:
    LHE = '1'
    NUM = 10000
<<<<<<< HEAD
    MIN_PT = 30
=======
    MIN_PT = 15.0
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8


# start pythia
pythia = pythia8.Pythia()

lhe_file = f'/nfs/dust/cms/user/vaguglie/simSetup/Box2/POWHEG-BOX-V2/hvq/testrun-tdec-lhc/Hdamp13TeV/BaseNom/dileptonic_fast/Results{LHE}/pwgevents.lhe'
pythia.readString("Beams:frameType = 4") # read info from a LHEF
pythia.readString(f'Beams:LHEF = {lhe_file}') # the LHEF to read from
print(f'Using LHE File: {lhe_file}')

# Veto Settings # https://github.com/cms-sw/cmssw/blob/master/Configuration/Generator/python/Pythia8PowhegEmissionVetoSettings_cfi.py
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

# HadronLevel
pythia.readString("HadronLevel:all = on") # on by default
# If off then stop the generation after the hard process and parton-level activity has been generated, but before the hadron-level steps.
# https://pythia.org/latest-manual/MasterSwitches.html
# pythia.readString("HadronLevel:all = off")


# Common settings CFI # https://github.com/cms-sw/cmssw/blob/master/Configuration/Generator/python/Pythia8CommonSettings_cfi.py
'''
pythia.readString('Tune:preferLHAPDF = 2')
pythia.readString('Main:timesAllowErrors = 10000')
pythia.readString('Check:epTolErr = 0.01')
pythia.readString('Beams:setProductionScalesFromLHEF = off')
pythia.readString('SLHA:minMassSM = 1000.')
pythia.readString('ParticleDecays:limitTau0 = on')
pythia.readString('ParticleDecays:tau0Max = 10')
pythia.readString('ParticleDecays:allowPhotonRadiation = on')
'''

### Additional parameters

pythia.readString("ParticleDecays:limitTau = on")
pythia.readString("ParticleDecays:tauMax = 10")
pythia.readString("6:onMode = on")
pythia.readString("-6:onMode = on")
pythia.readString("StringZ:rFactB = 0.855")
pythia.readString("Main:timesAllowErrors = 500")

pythia.readString("Random:setSeed = on")
pythia.readString(f"Random:seed = {LHE}")

# MiNNLO parameter
# pythia.readString("SpaceShower:dipoleRecoil = on") # off for hvq

# turn off parton settings | on by default
pythia.readString("PartonLevel:MPI = on") # default: on
pythia.readString("PartonLevel:FSR = on") # default: on
# pythia.readString("PartonLevel:MPI = off") # multiparton interaction
# pythia.readString("PartonLevel:FSR = off") # final state radiation


# this is why events were empty
# If off then stop the generation after the hard process has been generated, but before the parton-level and hadron-level steps. The process record is filled, but the event one is then not.
# https://pythia.org/latest-manual/MasterSwitches.html
# pythia.readString("PartonLevel:all = off")


# initialize pythia
pythia.init()

# Define jet clustering parameters
R = 0.4  # Jet radius
# min_pT = 30.0
max_eta = 2.4
jet_def = pythia8.SlowJet(-1, R, MIN_PT, max_eta)


def delta_R(delta_eta, delta_phi):
    if delta_phi > np.pi:
        delta_phi = 2*np.pi - delta_phi
    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
    return delta_R

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



def FindBquarks(particle, pdgid):
    is_from_top = False  # Initialize flag outside the `if` block
    current = particle  # Initialize `current` with the given `particle`
    
    # Check for b-quarks (id == 5) or anti-b-quarks (id == -5)
    if particle.id() == pdgid:
        # Traverse up the mother chain
        while current.mother1() > 0:  # Check if there's a valid mother particle
            mother1 = pythia.event[current.mother1()]
            mother2 = pythia.event[current.mother2()]

            # Check if any of the mothers are a top quark (id == 6) or anti-top quark (id == -6)
            if mother1.id() == 6 or mother2.id() == 6 or mother1.id() == -6 or mother2.id() == -6:
                is_from_top = True

            # Stop if the mothers are not b-quarks anymore (i.e., we are at the last copy)
            if mother1.id() != particle.id() and mother2.id() != particle.id():
                break

            # Move up the chain to the next particle (assuming mother1 is the relevant one)
            current = mother1  # Adjust based on how you want to prioritize mother1 or mother2

<<<<<<< HEAD
    return is_from_top, particle  # Return `current` which now represents the final b-quark
=======
    return is_from_top, current  # Return `current` which now represents the final b-quark
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8


def FindW(particle, pdgid):
    return FindBquarks(particle, pdgid)



def FindLepton(particle, sign):
    is_from_W = False  # Initialize flag outside the `if` block
    current = particle  # Initialize `current` with the given `particle`
    
    # Check for b-quarks (id == 5) or anti-b-quarks (id == -5)
    if particle.id() == sign*11 or particle.id() == sign*13 or particle.id() == sign*15:
        # Traverse up the mother chain
        while current.mother1() > 0:  # Check if there's a valid mother particle
            mother1 = pythia.event[current.mother1()]
            mother2 = pythia.event[current.mother2()]

            # Check if any of the mothers are a top quark (id == 6) or anti-top quark (id == -6)
            if mother1.id() == 24 or mother2.id() == 24 or mother1.id() == -24 or mother2.id() == -24:
                is_from_W = True

            # Stop if the mothers are not b-quarks anymore (i.e., we are at the last copy)
            if mother1.id() != particle.id() and mother2.id() != particle.id():
                break

            # Move up the chain to the next particle (assuming mother1 is the relevant one)
            current = mother1  # Adjust based on how you want to prioritize mother1 or mother2

<<<<<<< HEAD
    return is_from_W, particle  # Return `current` which now represents the final b-quark
=======
    return is_from_W, current  # Return `current` which now represents the final b-quark
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8


# num events to process
N = NUM
max_jets = 20
theta = 1 # for MiNNLO

# Begin event loop. Generate event.
# Skip if error. List first one.

# Initialize arrays to store jet multiplicities
nJets = []
jets_4vectors = []
<<<<<<< HEAD
jets_4vectors_no_b = []
=======
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8

# init particle vector array
P0 = []
W_Bosons_array = []
leptons_array = []
b_quark_array = []
b_jets_array  = []

count = 0

# use madgraph lhe_parser EventFile, because I don't know how to get event weights from pythia
lhe = EventFile(lhe_file)

# check that number of events in lhe file is >= N, since there were missmathces in the past
wgts_list = []
for i, event in enumerate(lhe):
    wgts_list.append(event.wgt)
    if i >= N:
        break

if len(wgts_list) < N:
    print(f'less then {NUM} events in LHE, using all LHE events')
    N = len(wgts_list)


for iEvent in range(N):
    if not pythia.next():
        continue
    # showering
    partVec = []
    TT = []
    top = None
    antitop = None

    W_Bosons = []
    Wp  = None
    Wm  = None
    b_quarks = []
    b_quark  = None
    ab_quark  = None
    b_jets   = []
    leptons  = []
    lp = None
    lm = None

    for particle in pythia.event:
        # selecting only last top
        if particle.id() == 6:
            top = particle
        if particle.id() == -6:
            antitop = particle

        # b quark
        is_from_top_b, particle = FindBquarks(particle, +5)
        is_from_top_ab, particle = FindBquarks(particle, -5)

        if is_from_top_b:
            b_quark = particle
        if is_from_top_ab:
            ab_quark = particle

        # W Boson
        is_from_top_Wp, particle = FindW(particle, +24)
        is_from_top_Wm, particle = FindW(particle, -24)

        if is_from_top_Wp:
            Wp = particle
        if is_from_top_Wm:
            Wm = particle


        # lepton
        is_from_W_lp, particle = FindLepton(particle, sign=+1)
        is_from_W_lm, particle = FindLepton(particle, sign=-1)

        if is_from_W_lp:
            lp = particle
        if is_from_W_lm:
            lm = particle

    # there can (very rarely) not be any b-quark, when the top decays into some other quark
    if b_quark==None:
        b_quarks.append([0, 0, 0, 0, 0, 0, 0])
        count = count +1
        print('b quark None')
        print(f'iEvent: {iEvent}')
    else: 
        b_quarks.append([b_quark.pT(), b_quark.y(), b_quark.phi(), b_quark.m(), b_quark.eta(), b_quark.e(), b_quark.id()])
    if ab_quark==None:
        b_quarks.append([0, 0, 0, 0, 0, 0, 0])
        count = count +1
        print('anti b quark None')
        print(f'iEvent: {iEvent}')
    else:
        b_quarks.append([ab_quark.pT(), ab_quark.y(), ab_quark.phi(), ab_quark.m(), ab_quark.eta(), ab_quark.e(), ab_quark.id()])

    # there are always two Ws, b/c they come directly from top
    W_Bosons.append([Wm.pT(), Wm.y(), Wm.phi(), Wm.m(), Wm.eta(), Wm.e(), Wm.id()])
    W_Bosons.append([Wp.pT(), Wp.y(), Wp.phi(), Wp.m(), Wp.eta(), Wp.e(), Wp.id()])

    # there are always two leptins, b/c they come directly from the Ws
    leptons.append([lp.pT(), lp.y(), lp.phi(), lp.m(), lp.eta(), lp.e(), lp.id()])    
    leptons.append([lm.pT(), lm.y(), lm.phi(), lm.m(), lm.eta(), lm.e(), lm.id()])

    b_quark_array.append(b_quarks)
    W_Bosons_array.append(W_Bosons)
    leptons_array.append(leptons)

    # print(f'{len(b_quarks) = }')
    # print(f'{len(leptons)  = }')

    patop = vector.obj(pt = antitop.pT(), eta = antitop.eta(), phi = antitop.phi(), mass = antitop.m())
    ptop  = vector.obj(pt = top.pT(),     eta = top.eta(),     phi = top.phi(),     mass = top.m())

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
            continue
            # jet_4vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        event_jets.append(jet_4vector)

    # b-filtering
    # print('before b-filtering')
    # print(f'{np.shape(event_jets) = }')
    # print(*event_jets, sep='\n')
    delta_R_list = []
    min_R_0 = 1.0
    min_R_1 = 1.0

    # get delta_R between ech b_quark and every jet
    for jet in event_jets:
        delta_eta_0 = abs(jet[4] - b_quarks[0][4])
        delta_phi_0 = abs(jet[2] - b_quarks[0][2])
        delta_R_0   = delta_R(delta_eta_0, delta_phi_0)

        delta_eta_1 = abs(jet[4] - b_quarks[1][4])
        delta_phi_1 = abs(jet[2] - b_quarks[1][2])
        delta_R_1   = delta_R(delta_eta_1, delta_phi_1)

        delta_R_list.append([delta_R_0, delta_R_1])

<<<<<<< HEAD
    if not delta_R_list: # list is empty -> there are no jets
=======
    if not delta_R_list: # there are no jets
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8
        print(f'Event {iEvent} has no jets')

    else: # delta_R_list is not empty, proceed with checking the jets
        delta_R_list = np.array(delta_R_list)

        min_index_0 = np.argmin(delta_R_list[:,0])
        min_index_1 = np.argmin(delta_R_list[:,1])

        # check if min sizes are below threshold
        if delta_R_list[min_index_0, 0] <= R: # R is defined with the jet as R = 0.4
            min_R_0 = delta_R_list[min_index_0, 0]

<<<<<<< HEAD
        if delta_R_list[min_index_1, 1] <= R:
=======
        if delta_R_list[min_index_0, 0] <= R:
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8
            min_R_1 = delta_R_list[min_index_1, 1]
      
        # check if both quarks have same jet with least delta_R
        if min_index_0 == min_index_1:
            # check which delta_R is smallest
            if min_R_0 < min_R_1:
                # find new smallest delta_R, setting old lowest to high value to exclude it
                min_R_1 = 1.0
                delta_R_list[min_index_1, 1] = 1.0
                min_index_1 = np.argmin(delta_R_list[:,1])
                if delta_R_list[min_index_1, 1] <= R:
                    min_R_1 = delta_R_list[min_index_1, 1]
            else: # min_R_1 <= min_R_0
                min_R_0 = 1.0
                delta_R_list[min_index_0, 0] = 1.0
                min_index_0 = np.argmin(delta_R_list[:,0])
                if delta_R_list[min_index_0, 0] <= R:
                    min_R_0 = delta_R_list[min_index_0, 0]

    b_jet_mask = []
    if min_R_0 <= R: # R is defined with the jet as R = 0.4
        b_jet_mask.append(min_index_0)
    if min_R_1 <= R:
        b_jet_mask.append(min_index_1)

    b_jet_mask.sort()
    # print(f'{b_jet_mask = }')
    # print(f'{len(event_jets) = }')

    # possible to have only one jet, for which both b_quarks have a low enough delta_R, in that case we can't remove the jet twice
    if b_jet_mask == [0, 0]:
        b_jet_mask = [0]

    # add b jets the list. If less than two b_jets, add empty
    for index in b_jet_mask:
        b_jets.append(event_jets[index])
    while len(b_jets) < 2:
        b_jets.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # sorting with highest index on top, so that when it is removed in that order, the lower indices are not changed
    b_jet_mask.sort(reverse = True)
    # remove b jets from rest of the jets
<<<<<<< HEAD
    event_jets_no_b = event_jets.copy()
    for index in b_jet_mask:
        event_jets_no_b = np.delete(event_jets_no_b, obj=index, axis=0)
=======
    for index in b_jet_mask:
        event_jets = np.delete(event_jets, obj=index, axis=0)
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8
    # print('finished b filtering:')

    # pad arrays to avoid jagged arrays
    while len(event_jets) < max_jets:
        jet_4vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if len(event_jets) == 0: # np.append gives an error when event_jets doesn't have correct shape, therefore if empty needs to be initialized
            event_jets = [jet_4vector]
        event_jets = np.append(event_jets, [jet_4vector], axis=0)

<<<<<<< HEAD
    
    # pad arrays to avoid jagged arrays
    while len(event_jets_no_b) < max_jets:
        jet_4vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if len(event_jets_no_b) == 0: # np.append gives an error when event_jets doesn't have correct shape, therefore if empty needs to be initialized
            event_jets_no_b = [jet_4vector]
        event_jets_no_b = np.append(event_jets_no_b, [jet_4vector], axis=0)

=======
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8

    # print(f'{iEvent = }')
    # print(f'{len(b_jets) = }')
    # print(f'{len(b_quarks) = }')

    # print('after b-filtering')
    # print(f'{np.shape(event_jets) = }')
    # print(*event_jets, sep='\n')
    # print(f'{iEvent = }')
    # print(f'{event_jets = }')
    jets_4vectors.append(event_jets)
<<<<<<< HEAD
    jets_4vectors_no_b.append(event_jets_no_b)
=======
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8
    b_jets_array.append(b_jets)


# End of event loop. Statistics. Histogram. Done.
pythia.stat()

<<<<<<< HEAD
print(f"{count} events didn't have two b quarks")

import os

dir = '/nfs/dust/cms/user/puschman/DCTR_Paper/Data/POWHEG_hvq/dileptonic/20240917'
=======

import os

dir = '/nfs/dust/cms/user/puschman/DCTR_Paper/Data/POWHEG_hvq/dileptonic/20240916'
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8
os.makedirs(dir, exist_ok=True)

# save shower
P0 = np.array(P0)
np.save(f'{dir}/converted_lhe_hvq_dileptonic_10K_{LHE}_b-filtered.npy', P0)
print(f'{np.shape(P0) = }')

# save multiplicity and jet observables
nJets = np.array(nJets)
np.save(f'{dir}/jet_multiplicity_hvq_dileptonic_10K_{LHE}_b-filtered.npy', nJets)
print(f'{np.shape(nJets) = }')

<<<<<<< HEAD
# all jets incl b
=======
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8
jets_4vectors = np.array(jets_4vectors)
np.save(f'{dir}/jet_4vectors_hvq_dileptonic_10K_{LHE}_b-filtered.npy', jets_4vectors)
print(f'{np.shape(jets_4vectors) = }')

<<<<<<< HEAD
# all jets without b
jets_4vectors_no_b = np.array(jets_4vectors_no_b)
np.save(f'{dir}/jet_4vectors_no_b_hvq_dileptonic_10K_{LHE}_b-filtered.npy', jets_4vectors_no_b)
print(f'{np.shape(jets_4vectors_no_b) = }')

=======
>>>>>>> e8dafdf0c6e30e46bc0ff90c25a9652ca55ebec8
# save W_boson array
W_Bosons_array = np.array(W_Bosons_array)
np.save(f'{dir}/W_Bosons_array_hvq_dileptonic_10K_{LHE}_b-filtered.npy', W_Bosons_array)
print(f'{np.shape(W_Bosons_array) = }')

# save lepton array
leptons_array = np.array(leptons_array)
np.save(f'{dir}/leptons_array_hvq_dileptonic_10K_{LHE}_b-filtered.npy', leptons_array)

# save b_quark array
b_quark_array = np.array(b_quark_array)
np.save(f'{dir}/b_quarks_hvq_dileptonic_10K_{LHE}_b-filtered.npy', b_quark_array)
print(f'{np.shape(b_quark_array) = }')

# save b_jets array
b_jets_array = np.array(b_jets_array)
np.save(f'{dir}/b_jets_hvq_dileptonic_10K_{LHE}_b-filtered.npy', b_jets_array)
print(f'{np.shape(b_jets_array) = }')
