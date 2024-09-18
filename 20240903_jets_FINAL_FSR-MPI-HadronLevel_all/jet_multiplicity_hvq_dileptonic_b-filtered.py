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
lib = "../lib"

for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
sys.path.insert(0, lib)

# parse cli arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="How many events to shower and get jets for")
    # LHE arg
    parser.add_argument("-l", "--lhe", help="String. Which hvq LHE File to open. Values between 1 and 20. Default = '1'", type = str, default = '1')
    # NUM arg
    parser.add_argument("-n", "--num", help="Int. Number of events to shower. Default = 1000000", type = int, default = 1000000)
    # MIN_PT arg
    parser.add_argument("-p", "--pt", "--min_pt", help="Float. Minimum pt for jet finding algorithm. Default = 15.0", type = float, default = 15.0)

    args = parser.parse_args()
    LHE = args.lhe
    NUM = args.num
    MIN_PT = args.pt
else:
    LHE = '1'
    NUM = 1000000
    MIN_PT = 15.0


# start pythia
pythia = pythia8.Pythia()

lhe_file = f'/nfs/dust/cms/user/vaguglie/simSetup/Box2/POWHEG-BOX-V2/hvq/testrun-tdec-lhc/Hdamp13TeV/BaseNom/dileptoninc/Results{LHE}/pwgevents.lhe'
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

# Initialize, incoming pp beams are default.
pythia.init()

# Define jet clustering parameters
R = 0.4  # Jet radius
# min_pT = 30.0
max_eta = 2.4
jet_def = pythia8.SlowJet(-1, R, MIN_PT, max_eta)


def delta_R(delta_eta, delta_phi):
    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
    return delta_R

'''
def deltaR(eta1, phi1, eta2, phi2):
    dphi = np.abs(phi1 - phi2)
    if dphi > np.pi:
        dphi = 2 * np.pi - dphi
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)
'''

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
N = NUM
max_jets = 24
theta = 0 # for hvq 

# Begin event loop. Generate event.
# Skip if error. List first one.

# Initialize arrays to store jet multiplicities
nJets = []
jets_4vectors = []

# init particle vector array
P0 = []
W_and_lepton_array = []
b_quark_array = []
b_jets_array  = []

# use madgraph lhe_parser EventFile, because I don't know how to get event weights from pythia
lhe = EventFile(lhe_file)

# check that number of events in lhe file is >= N, since there were missmathces in the past
wgts_list = []
for i, event in enumerate(lhe):
    wgts_list.append(event.wgt)
    if i >= N:
        break

if len(wgts_list) <= N:
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

    # W_bosons = []
    W_boson  = None
    b_quarks = []
    b_quark  = None
    # leptons  = []
    lepton   = None
    b_jets   = []
    for particle in pythia.event:
        # selecting only last top
        if particle.id() == 6:
            top = particle
        elif particle.id() == -6:
            antitop = particle
        # only consider children of top anti-top
        if pythia.event[particle.mother1()].idAbs()==6 or pythia.event[particle.mother2()].idAbs()==6:
            # W_boson
            if particle.idAbs() == 24:
                W_boson = particle

            # b_quark
            elif particle.idAbs() == 5:
                b_quark = particle
                b_quarks.append([b_quark.pT(), b_quark.y(), b_quark.phi(), b_quark.m(), b_quark.eta(), b_quark.e(), particle.id()])

        # leptons come from W
        elif pythia.event[particle.mother1()].idAbs()==24 or pythia.event[particle.mother2()].idAbs()==24:
            # e
            if particle.idAbs() == 11:
                # print('found electron')
                lepton = particle
            # mu
            elif particle.idAbs() == 13:
                # print('found mu')
                lepton = particle
            # tau
            elif particle.idAbs() == 15:
                # print('found tau')
                lepton = particle

    p_lepton = [lepton.pT(), lepton.y(), lepton.phi(), lepton.m(), lepton.eta(), lepton.e(), lepton.id()]
    p_W = [W_boson.pT(), W_boson.y(), W_boson.phi(), W_boson.m(), W_boson.eta(), W_boson.e(), W_boson.id()]

    W_and_lepton_array.append([p_W, p_lepton])
    b_quark_array.append(b_quarks)

    # print(f'{len(b_quarks) = }')
    # print(f'{len(leptons)  = }')

    # deprecated
    # patop = uproot_methods.TLorentzVector.from_ptetaphim(antitop.pT(), antitop.eta(), antitop.phi(), antitop.m())
    # ptop = uproot_methods.TLorentzVector.from_ptetaphim(top.pT(), top.eta(), top.phi(), top.m())

    patop = vector.obj(pt = antitop.pT(), eta = antitop.eta(), phi = antitop.phi(), mass = antitop.m())
    ptop  = vector.obj(pt = top.pT(),     eta = top.eta(),     phi = top.phi(),     mass = top.m())

    p_W = vector.obj(pt = W_boson.pT(), eta = W_boson.eta(), phi = W_boson.phi(), mass = W_boson.m())
    # p_lepton = vector.obj(pt = lepton.pT(), eta = lepton.eta(), phi = lepton.phi(), mass = lepton.m())

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


    # removed_counter = 0
    # filter b-jets
    b_jet_mask = []
    # for b_quark in b_quarks:
    tmp_R = 1.0
    tmp_index = None

    delta_R_list = []
    for jet in event_jets:

        delta_eta_0 = abs(jet[4] - b_quarks[0][4])
        delta_phi_0 = abs(jet[2] - b_quarks[0][2])
        if delta_phi_0 > np.pi:
            delta_phi_0 = 2 * np.pi - delta_phi_0
        delta_R_0   = delta_R(delta_eta_0, delta_phi_0)

        delta_eta_1 = abs(jet[4] - b_quarks[1][4])
        delta_phi_1 = abs(jet[2] - b_quarks[1][2])
        if delta_phi_1 > np.pi:
            delta_phi_1 = 2 * np.pi - delta_phi_1
        delta_R_1   = delta_R(delta_eta_1, delta_phi_1)

        if delta_R_0 > R:
            delta_R_0 = 1000
	if delta_R_1 > R:
            delta_R_1 = 1000
        delta_R_list.append([delta_R_0, delta_R_1])

    delta_R_list = np.array(delta_R_list)

    min_index_0 = np.argmin(delta_R_list[:,0])
    min_index_1 = np.argmin(delta_R_list[:,1])

    min_R_0 = delta_R_list[min_index_0, 0]
    min_R_1 = delta_R_list[min_index_1, 1]


    if min_index_0 == min_index_1:
        tmp_dR = 1000
        tmp_index = None
        if min_R_0 < min_R_1:
            b_jet_mask.append(min_index_0)
            for index, d in enumerate(delta_R_list[:, 1]):
                if d <= tmp_dR and d > min_R_1:
                    tmp_dR = d
                    tmp_index = index
            b_jet_mask.append(tmp_index)
        else:
           b_jet_mask.append(min_index_1)
           for index, d in enumerate(delta_R_list[:, 0]):
                if min_R_1 > R:
                    continue
                if d <= tmp_dR and d > min_R_0:
                    tmp_dR = d
                    tmp_index = index
           if tmp_index is not none:
                b_jet_mask.append(tmp_index)
    else:
        b_jet_mask.append(min_index_0)
        b_jet_mask.append(min_index_1)


    '''
    for i, jet in enumerate(event_jets):
        del_eta = abs(jet[4] - b_quark[4])
        del_phi = abs(jet[2] - b_quark[2])
        del_R = delta_R(del_eta, del_phi)

        if del_R <= R:
            # print(f'found jet with {del_R = }, removing from event jets array and saving to separate b_jets array')
            if del_R <= tmp_R: # check if second a new match is closer
                tmp_R = del_R
                tmp_index = i
    if tmp_index is not None:
        b_jet_mask.append(tmp_index)
    '''

    b_jet_mask.sort(reverse = True)
    print(f'{b_jet_mask = }')
    # b_jet_mask = np.array(b_jet_mask)
    # b_jets = event_jets[b_jet_mask]
    for index in b_jet_mask:
        b_jets.append(event_jets[index])
    for index in b_jet_mask:
    	event_jets = np.delete(event_jets, obj=index, axis=0)
    # print('finished b filtering:')

    while len(event_jets) < max_jets:
        jet_4vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if len(event_jets) == 0:
            event_jets = [jet_4vector]
        event_jets = np.append(event_jets, [jet_4vector], axis=0)

    while len(b_jets) < 2:
        jet_4vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b_jets.append(jet_4vector)
    if len(b_jets) >= 3:
        print(f'{iEvent = }')
        print(f'{len(b_jets) = }')
        print(f'{len(b_quarks) = }')

    # print('after b-filtering')
    # print(f'{np.shape(event_jets) = }')
    # print(*event_jets, sep='\n')
    # print(f'{iEvent = }')
    # print(f'{event_jets = }')
    jets_4vectors.append(event_jets)
    b_jets_array.append(b_jets)

# End of event loop. Statistics. Histogram. Done.
pythia.stat()


import os

dir = '/nfs/dust/cms/user/puschman/DCTR_Paper/Data/POWHEG_hvq/dileptonic/20240913'
os.makedirs(dir, exist_ok=True)

# save shower
P0 = np.array(P0)
np.save(f'{dir}/converted_lhe_hvq_dileptonic_1M_{LHE}_b-filtered.npy', P0)
print(f'{np.shape(P0) = }')

# save multiplicity and jet observables
nJets = np.array(nJets)
np.save(f'{dir}/jet_multiplicity_hvq_dileptonic_1M_{LHE}_b-filtered.npy', nJets)
print(f'{np.shape(nJets) = }')

jets_4vectors = np.array(jets_4vectors)
np.save(f'{dir}/jet_4vectors_hvq_dileptonic_1M_{LHE}_b-filtered.npy', jets_4vectors)
print(f'{np.shape(jets_4vectors) = }')

# save W_boson and lepton array
W_and_lepton_array = np.array(W_and_lepton_array)
np.save(f'{dir}/W_and_lepton_hvq_dileptonic_1M_{LHE}_b-filtered.npy', W_and_lepton_array)
print(f'{np.shape(W_and_lepton_array) = }')

# save b_quark array
b_quark_array = np.array(b_quark_array)
np.save(f'{dir}/b_quarks_hvq_dileptonic_1M_{LHE}_b-filtered.npy', b_quark_array)
print(f'{np.shape(b_quark_array) = }')

# save b_jets array
b_jets_array = np.array(b_jets_array)
np.save(f'{dir}/b_jets_hvq_dileptonic_1M_{LHE}_b-filtered.npy', b_jets_array)
print(f'{np.shape(b_jets_array) = }')
