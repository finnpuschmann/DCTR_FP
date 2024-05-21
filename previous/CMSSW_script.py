import onnxruntime as ort
import uproot
import numpy as np
import math
import uproot_methods


def norm(X, nrm, ln=False):
    # print(f'X shape: {X.shape()}')
    # load normalization
    (mean, std, ln) = nrm
    # use log
    if ln == True:
        X = np.log(np.clip(X, a_min = 1e-6, a_max = None))
    # adjust mean to be 0
    X -= mean 
    # adjust std to be 1
    if std >= 1e-2: # for avoiding divide by zero error
        X /= std

    return X


def normalize_data(X, nrm_array):
    # [pt, rapidity, phi, mass, pseudorapidity, E, PID, w, theta]
    # [0 , 1       , 2  , 3   , 4             , 5, 6  , 7, 8    ]
    # print(f'X shape: {np.array(X).shape()}')
    for particle, nrm_part in enumerate(nrm_array):
        for arg, nrm_arg in enumerate(nrm_part):
            # print(f'part: {particle}, arg: {arg}')
            X[particle, arg] = norm(X[particle, arg], nrm = nrm_arg)

    return X


# open the root file and get the desired tree
file = uproot.open("/pnfs/desy.de/cms/tier2/store/mc/RunIISummer20UL18NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/1F39E43A-2869-4540-8A95-B46F63B7D7B0.root")
tree = file["Events"]


# my datasets:
# only contain tt-pair; every event has order: 
    # tt-pair, top, anti-top
# every particle has arguments: 
    # [pt, y, phi, mass, eta, E, PID, w, theta]
    # [0 , 1, 2  , 3   , 4  , 5, 6  , 7, 8    ]

# Get the desired arrays from the data
GenPart_pdgId = tree["GenPart_pdgId"].array()
GenPart_statusFlags = tree["GenPart_statusFlags"].array()
GenPart_pt = tree["GenPart_pt"].array()
GenPart_phi = tree["GenPart_phi"].array()
GenPart_eta = tree["GenPart_eta"].array()
GenPart_mass = tree["GenPart_mass"].array()


## create inference session using ort.InferenceSession from a given model
ort_sess = ort.InferenceSession('/nfs/dust/cms/user/vaguglie/Finn_git/DCTR_FP/mymodel12_13TeV_MiNNLO.onnx')
input_name = ort_sess.get_inputs()[0].name
label_name = ort_sess.get_outputs()[0].name
print("input_name: "+str(input_name))
print("label_name: "+str(label_name))


# Values come from normalization paramters of POWHEG hvq 
# [mean, std, use_log]
nrm_array = [[(3.6520673599656903, 1.0123402362573612, True), # tt-pair
              (0.0001718810581680775, 1.0362455506718102, False),
              (2.8943571877384285e-05, 1.8139038706413384, False),
              (6.21729978047307, 0.2771419580231537, True)],
             [(4.595855742518925, 0.7101176940989488, True), # top
              (0.00022746366634849002, 1.213207643109532, False),
              (-0.00028213870737636996, 1.8136544140703632, False),
              (171.93706459943778, 6.9652037622153, False)],
             [(4.5986175957604045, 0.7103218938891299, True), # anti-top
              (0.00011712322394057398, 1.2076422016031159, False),
              (0.0003628069129526392, 1.8139415747773364, False),
              (171.93691192651536, 6.9500586980501575, False)]]


# PDGid to small float dictionary
PID2FLOAT_MAP = {6:  0.6,
                 0:  0.0,
                -6: -0.6}

countTop = 0


# Loop over the entries
for jEntry in range(10):
    print("Entry: ", jEntry)
    particlesvector=[]
    P0 = []

    # Loop on the genParticles, selecting only INITIAL top and antitop (considering parton shower) 
    for i in range(0, len(GenPart_pdgId[jEntry])):
        if GenPart_pdgId[jEntry][i] == 6:

            if (((GenPart_statusFlags[jEntry][i] >> 12) & 0x1) > 0):
                countTop += 1
                ptop = uproot_methods.TLorentzVector.from_ptetaphim(GenPart_pt[jEntry][i], GenPart_eta[jEntry][i], GenPart_phi[jEntry][i], GenPart_mass[jEntry][i])
        
        if GenPart_pdgId[jEntry][i] == -6:

            if (((GenPart_statusFlags[jEntry][i] >> 12) & 0x1) > 0):
                countTop += 1
                patop = uproot_methods.TLorentzVector.from_ptetaphim(GenPart_pt[jEntry][i], GenPart_eta[jEntry][i], GenPart_phi[jEntry][i], GenPart_mass[jEntry][i])
   

    # Creating the array with all info needed to pass to the NN model, then normalize it
    # every particle in training has arguments: 
    # [pt, y, phi, mass, PID]

    p_tt = ptop + patop

    # create non normalized vectors 
    particlesvector.append([ p_tt.pt,  p_tt.rapidity,  p_tt.phi,  p_tt.mass, PID2FLOAT_MAP.get( 0, 0.0)])
    particlesvector.append([ ptop.pt,  ptop.rapidity,  ptop.phi,  ptop.mass, PID2FLOAT_MAP.get( 6, 0.0)])
    particlesvector.append([patop.pt, patop.rapidity, patop.phi, patop.mass, PID2FLOAT_MAP.get(-6, 0.0)])
    
    particlesvector = np.array(particlesvector)
    # normalize particlesvector
    particlesvector = normalize_data(particlesvector, nrm_array)

    P0.append(particlesvector)
    P0=np.array(P0)
    print(P0.shape)
    print(P0)
    

    ## run inference
    pred = ort_sess.run([label_name], {input_name: P0.astype(np.float32)})[0]

    weight = pred[:,1]/pred[:,0]
    print(f'weight: {weight}')


print('countTop:'+str(countTop))
