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

Rb = float(sys.argv[1])
N = 1000000
# Configure the Pythia object.
pythia.readString("Top:gg2ttbar = on")  # Switch on process.
pythia.readString("Top:qqbar2ttbar = on")
pythia.readString("Beams:eCM = 13000.") # Set 14 TeV CM energy.
pythia.readString("ParticleDecays:limitTau = on")
pythia.readString("ParticleDecays:tauMax = 10")
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString("6:onMode = on")
pythia.readString("-6:onMode = on")
pythia.readString("StringZ:rFactB = %f" %Rb)
pythia.readString("Main:timesAllowErrors = 500")
pythia.readString("PartonLevel:MPI = off")
pythia.readString("24:mayDecay = off")
pythia.readString("TimeShower:alphaSvalue = 0.1365")
pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 3")

# Initialize, incoming pp beams are default.
pythia.init()

import numpy as np

multH = pythia8.Hist("charged multiplicity", 100, -0.5, 799.5)
mult=[]
multC=[]
multNeutra = []
listBtop = []
listBextra=[]

TopP = []
antiTopP = []

bFirstP = []
bLastP = []
abFirstP = []
abLastP = []

BposP = []
BnegP = []
AllP4 = []

listXb = []
error= 0
error1= 0
error2= 0
# Begin event loop. Generate event. Skip if error. List first one.
for iEvent in range(0, N):
    if not pythia.next(): continue
    #if (iEvent != 105): continue
    #pythia.event.list()
    # Find number of all final charged particles and fill histogram.
    partVec = []
    top = None
    antitop = None
    bFirst = None
    abFirst = None
    bLast = None
    abLast = None
    Bpos = None
    Bneg = None
    Wplus = None
    Wminus = None
    nCharged = 0
    count = 0
    countB = 0
    countBtop = 0
    m2top = 0
    m2antitop = 0
    m2wplus = 0
    m2wminus = 0
  
    q = (0,0,0,0)
    b1 = (0,0,0,0)
    b = (0,0,0,0)
    antiq = (0,0,0,0)
    ab1 = (0,0,0,0)
    ab = (0,0,0,0)

    for prt in pythia.event:

        ##selecting only last top
        if prt.id() == 6: 
            top = prt
            #print("top E")
        if prt.id() == -6:
            antitop = prt
            #print("antitop E")


        if (prt.id() == 24 and (pythia.event[prt.mother1()].idAbs()==6 or pythia.event[prt.mother2()].idAbs()==6)): 
            Wplus = prt 
            #print("W+ E")
        if (prt.id() == -24 and (pythia.event[prt.mother1()].idAbs()==6 or pythia.event[prt.mother2()].idAbs()==6)):
            Wminus = prt 
            #print("W- E")


        ##selecting abFirst and abLast
        if prt.id() == -5:
            #print("antib E")
            if (pythia.event[prt.mother1()].id()==-6):
                abFirst = prt
                #print("abtiB first E")
            if (pythia.event[prt.mother1()].id()==-5):    
                abLast = prt
        
        ##selecting bFirst and bLast
        if prt.id() == 5:
            #print("b E")
            if (pythia.event[prt.mother1()].id()==6):
                bFirst = prt
                #print("b first E")
            if (pythia.event[prt.mother1()].id()==5):    
                bLast = prt

        #pdb.set_trace()
        if (bFirst==None) or (abFirst==None): continue
        #if (bLast==None) or (abLast==None): continue
 

        ##selecting first B hadron
        if '5' in str(prt.id()) and prt.id() != (5) and prt.id() != -5:
            countB +=1
            if ( pythia.event[prt.mother1()].id()==5 or pythia.event[prt.mother1()].id()==-5) or (pythia.event[prt.mother2()].id()==5 or pythia.event[prt.mother2()].id()==-5):         
                #print("id bHadron: "+str(prt.id()))
                #print("no: "+str(prt.index()))
                countBtop +=1
                if (pythia.event[prt.mother1()].id()==5 or pythia.event[prt.mother2()].id()==5):
                    Bpos = prt

                if (pythia.event[prt.mother1()].id()==-5 or pythia.event[prt.mother2()].id()==-5):
                    Bneg = prt
                

        ## Selecting only final charged particles, i.e. charged multiplicity 
        if prt.isFinal() and prt.isCharged(): nCharged += 1
        if prt.isFinal(): count += 1

    if (bFirst==None) or (abFirst==None):
        error = error + 1
        continue
    if (bLast==None) or (abLast==None): 
        error1 = error1 + 1
        continue
    if (Bpos==None) or (Bneg==None): 
        error2 = error2 + 1
        continue
    
    multH.fill(nCharged)
    mult.append(count)
    multC.append(nCharged)
    multNeutra.append(count-nCharged)
    partVec.append([top.pT(), top.y(), top.phi(), top.m(), 6, Rb])
    partVec.append([antitop.pT(), antitop.y(), antitop.phi(), antitop.m(), -6, Rb])
    TopP.append([top.pT(), top.y(), top.phi(), top.m(), 6, Rb])
    antiTopP.append([antitop.pT(), antitop.y(), antitop.phi(), antitop.m(), -6, Rb])

    m2top = top.m2()
    m2wplus = Wplus.m2()
    q = top.p()
    b1 = bFirst.p()
    #b = bLast.p()
    b = Bpos.p()
    num = 2* (b.e()*q.e()-b.px()*q.px()-b.py()*q.py()-b.pz()*q.pz()) / m2top
    den = 1 - (m2wplus/m2top)
    #num = 2* (b.e()*q.e()-b.px()*q.px()-b.py()*q.py()-b.pz()*q.pz()) / 172.5*172.5
    #den = 1 - (80.4*80.4/172.5*172.5)
    xb = num / den
    
    antim2top = antitop.m2()
    m2wminus = Wminus.m2()
    antiq = antitop.p()
    ab1 = abFirst.p()
    #ab = abLast.p()
    ab = Bneg.p()
    anum = 2* (ab.e()*antiq.e()-ab.px()*antiq.px()-ab.py()*antiq.py()-ab.pz()*antiq.pz()) / antim2top
    aden = 1 - (m2wminus/antim2top)
    #anum = 2* (ab.e()*antiq.e()-ab.px()*antiq.px()-ab.py()*antiq.py()-ab.pz()*antiq.pz()) / 172.5*172.5
    #aden = 1 - (80.4*80.4/172.5*172.5)
    axb = anum /aden
    
    
    partVec.append([bFirst.pT(), bFirst.eta(), bFirst.phi(), bFirst.m(), 5, Rb])
    partVec.append([bLast.pT(), bLast.eta(), bLast.phi(), bLast.m(), 5, Rb])
    partVec.append([abFirst.pT(), abFirst.eta(), abFirst.phi(), abFirst.m(), -5, Rb])
    partVec.append([abLast.pT(), abLast.eta(), abLast.phi(), abLast.m(), -5, Rb])
    partVec.append([Bpos.pT(), Bpos.eta(), Bpos.phi(), Bpos.m(), Bpos.id(), Rb])
    partVec.append([Bneg.pT(), Bneg.eta(), Bneg.phi(), Bneg.m(), Bneg.id(), Rb])
    partVec.append([Wminus.pT(), Wminus.eta(), Wminus.phi(), Wminus.m(), 24, Rb])
    partVec.append([Wplus.pT(), Wplus.eta(), Wplus.phi(), Wplus.m(), 24, Rb])
    AllP4.append(partVec)

    bFirstP.append([bFirst.pT(), bFirst.eta(), bFirst.phi(), bFirst.m(), 5, Rb])
    bLastP.append([bLast.pT(), bLast.eta(), bLast.phi(), bLast.m(), 5, Rb])
    abFirstP.append([abFirst.pT(), abFirst.eta(), abFirst.phi(), abFirst.m(), -5, Rb])
    abLastP.append([abLast.pT(), abLast.eta(), abLast.phi(), abLast.m(), -5, Rb])
    BposP.append([Bpos.pT(), Bpos.pz(), Bpos.eta(), Bpos.phi(), Bpos.m(), Bpos.id(), Rb])
    BnegP.append([Bneg.pT(), Bneg.pz(), Bneg.eta(), Bneg.phi(), Bneg.m(), Bneg.id(), Rb])
    
    listBextra.append(countB-countBtop)
    listBtop.append(countBtop)
    
    listXb.append(xb)
    listXb.append(axb)


# End of event loop. Statistics. Histogram. Done.
pythia.stat();


P0 = np.array(listXb)
P1 = np.array(multC)
P2 = np.array(multNeutra)
P3 = np.array(listBtop)
P4 = np.array(listBextra)

P5 = np.array(bFirstP)
P6 = np.array(bLastP)
P7 = np.array(abFirstP)
P8 = np.array(abLastP)
P9 = np.array(BposP)
P10 = np.array(BnegP)
P11 = np.array(TopP)
P12 = np.array(antiTopP)

Pall = np.array(AllP4)

np.savez_compressed('Xb_multC_multNeutra_listBtop_listBextra-Rb_%.3f_1M_seed3.npz' %Rb, a=P0, b=P1, c=P2, d=P3, e=P4)
np.savez_compressed('pT:bfirst_blast_abfirst_ablast_Bpos_Bneg_top_antitop-Rb_%.3f_1M_seed3.npz' %Rb, a=P5, b=P6, c=P7, d=P8, e=P9, f=P10, g=P11, h=P12)
np.savez_compressed('AllP4:TopP-antiTopP-bFirstP-bLastP-abFirstP-abLastP-BposP-BnegP-Rb_%.3f_1M_seed3.npz' %Rb, a=Pall)
np.savez_compressed('AllP4:TopP-antiTopP-bLastP-abLastP-BposP-BnegP-wplus-wminus-Rb_%.3f_1M_seed3.npz' %Rb, a=Pall)


print(mult)
print(len(TopP))
print(len(antiTopP))
print(len(bFirstP))
print(len(bLastP))
print(len(abFirstP))
print(len(abLastP))
print(len(BposP))
print(len(BnegP))
print(len(listBtop))
print(len(listBextra))
print(Pall.shape)
print(error)
print(error1)
print(error2)
