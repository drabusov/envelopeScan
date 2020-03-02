import os
import string
import sys
from copy import copy, deepcopy
from pyLine import *
#from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.constants import c
from numba import jit
import matplotlib.pyplot as plt
from os.path import expanduser
import numpy as np
from scipy.optimize import minimize as som

#------------------------------------------------------
# Transport of twiss values through one madx sectormap
# tw: initial twiss values
#-------------------------------------------------------

#@jit
def twiss_transport(M,tw):
	gamx0=(1.0+tw.alpx**2)/tw.betx
	betx=M[0,0]**2*tw.betx-2.0*M[0,0]*M[0,1]*tw.alpx+M[0,1]**2*gamx0
	alpx=-M[0,0]*M[1,0]*tw.betx+(M[0,0]*M[1,1]+M[0,1]*M[1,0])*tw.alpx-M[1,1]*M[0,1]*gamx0
	gamx=M[1,0]**2*tw.betx-2.0*M[1,1]*M[1,0]*tw.alpx+M[1,1]**2*gamx0
	gamy0=(1.0+tw.alpy**2)/tw.bety
	bety=M[2,2]**2*tw.bety-2.0*M[2,2]*M[2,3]*tw.alpy+M[2,3]**2*gamy0
	alpy=-M[2,2]*M[3,2]*tw.bety+(M[2,2]*M[3,3]+M[2,3]*M[3,2])*tw.alpy-M[3,3]*M[2,3]*gamy0
	gamy=M[3,2]**2*tw.bety-2.0*M[3,3]*M[3,2]*tw.alpy+M[3,3]**2*gamy0
	Dx=M[0,0]*tw.Dx+M[0,1]*tw.Dpx+M[0,5]
	Dpx=M[1,0]*tw.Dx+M[1,1]*tw.Dpx+M[1,5]
	tw.betx=betx
	tw.alpx=alpx
	tw.bety=bety
	tw.alpy=alpy
	tw.Dx=Dx
	tw.Dpx=Dpx

#------------------------------------------------------
# Transport of twiss values through Ncell cells
# tw: initial Twiss elements
# Ksc: space charge perveance
# emitx, emity: rms emittances
#-------------------------------------------------------

#@jit
def twiss_evolution(tw,Ksc,emitx,emity,sigma_p,Ncells):
	Nelements=len(BeamLine)
	Lb=get_bline_length()
	tw0=deepcopy(tw)
	twiss_vec=zeros((Nelements*Ncells,6))
	Msc=SectorMap("sc_kick","matrix")
	ax_lens=0.0
	ay_lens=0.0
	for l in range(Ncells):
		for j, b in enumerate(BeamLine):
			ds=b.get_twiss().L
			twiss_transport(b.get_M(),tw0) # transport without space charge
			ax=np.sqrt(tw0.betx*emitx + (tw0.Dx*sigma_p)**2)
			ay=np.sqrt(tw0.bety*emity)
			kick_gradient_x=0.5*Ksc/(ax*(ax+ay))*ds
			kick_gradient_y=0.5*Ksc/(ay*(ax+ay))*ds
			sccomp=b.get_twiss().sccomp
			if sccomp > 0.0:
				if ax_lens == 0.0:
					ax_lens=ax
					ay_lens=ay
				kick_gradient_x=-0.5*Ksc/(ax_lens*(ax_lens+ay_lens))*Lb*sccomp
				kick_gradient_y=-0.5*Ksc/(ay_lens*(ax_lens+ay_lens))*Lb*sccomp
			#else:
			#	kick_gradient_x=0.0
			#	kick_gradient_y=0.0
			Msc.kick(kick_gradient_x,kick_gradient_y)
			twiss_transport(Msc.get_M(),tw0)     # space charge kick
			twiss_vec[Nelements*l+j,0]=tw0.betx
			twiss_vec[Nelements*l+j,1]=tw0.alpx
			twiss_vec[Nelements*l+j,2]=tw0.bety
			twiss_vec[Nelements*l+j,3]=tw0.alpy
			twiss_vec[Nelements*l+j,4]=tw0.Dx
			twiss_vec[Nelements*l+j,5]=tw0.Dpx
		tw0.betx=twiss_vec[Nelements*(l+1)-1,0]
		tw0.alpx=twiss_vec[Nelements*(l+1)-1,1]
		tw0.bety=twiss_vec[Nelements*(l+1)-1,2]
		tw0.alpy=twiss_vec[Nelements*(l+1)-1,3]
		tw0.Dx=twiss_vec[Nelements*(l+1)-1,4]
		tw0.Dpx=twiss_vec[Nelements*(l+1)-1,5]
	return twiss_vec


#------------------------------------------------------
# get phase advances
# betax, betay: beta function along one cell
#-------------------------------------------------------

@jit
def phase_advance_dispersion(emitx,sigma_p,betx,bety,Dx):
	phasex=0.0
	phasey=0.0
	for j, b in enumerate(BeamLine):
		ds=b.get_twiss().L
		phasex+=emitx*ds/(emitx*betx[j]+(Dx[j]*sigma_p)**2)
		phasey+=ds/bety[j]
	return phasex, phasey

def phase_advance(betx,bety):
	phasex=0.0
	phasey=0.0
	for j, b in enumerate(BeamLine):
		ds=b.get_twiss().L
		phasex+=ds/betx[j]
		phasey+=ds/bety[j]
	return phasex, phasey


def phase_shift_analytic(emitx,emity,Ksc,lc):
	return Ksc*lc/(4.0*emitx*(1.0+np.sqrt(emity/emitx)))  , Ksc*lc/(4.0*emity*(1.0+np.sqrt(emitx/emity)))

#------------------------------------------------------
# help function for root in match_twiss
#-------------------------------------------------------

@jit
def func_fsolve(x,Ksc,emitx,emity,sigma_p):
	N=len(BeamLine)
	tw=deepcopy(BeamLine[-1].stwiss)
	tw.betx=x[0]
	tw.alpx=x[1]
	tw.bety=x[2]
	tw.alpy=x[3]
	tw.Dx=x[4]
	tw.Dpx=x[5]
	twiss_vec=twiss_evolution(tw,Ksc,emitx,emity,sigma_p,1)
	return[twiss_vec[N-1,0]-x[0],twiss_vec[N-1,1]-x[1],twiss_vec[N-1,2]-x[2],twiss_vec[N-1,3]-x[3],twiss_vec[N-1,4]-x[4],twiss_vec[N-1,5]-x[5]]

#------------------------------------------------------
# find matched twiss values in one cell with space charge
# Ksc: space charge perveance
# emitx, emity: rms emittances
#-------------------------------------------------------

@jit
def match_twiss_matrix(emitx,emity,sigma_p,Ksc):
		Nb=len(BeamLine)
   	# start value
		tw=deepcopy(BeamLine[-1].stwiss)
   	# solver
		sol = root(func_fsolve, [tw.betx,tw.alpx,tw.bety,tw.alpy,tw.Dx,tw.Dpx], args=(Ksc,emitx,emity,sigma_p),method='lm')
		tw.betx=sol.x[0]
		tw.alpx=sol.x[1]
		tw.bety=sol.x[2]
		tw.alpy=sol.x[3]
		tw.Dx=sol.x[4]
		tw.Dpx=sol.x[5]
		twiss_vec=twiss_evolution(tw,Ksc,emitx,emity,sigma_p,1)
		return twiss_vec


def get_matched_envelopes(twiss_vec,emitx,emity,sigma_p):
	envx=np.sqrt(emitx*twiss_vec[:,0]+(twiss_vec[:,4]*sigma_p)**2)
	envy=np.sqrt(emity*twiss_vec[:,2])
	return envx,envy

def get_matched_twiss(twiss_vec):
	Nb=len(BeamLine)
	tw=Twiss()
	tw.s=BeamLine[Nb-1].stwiss.s
	tw.L=BeamLine[Nb-1].stwiss.L
	tw.alpx=twiss_vec[Nb-1,1]
	tw.alpy=twiss_vec[Nb-1,3]
	tw.betx=1.0*twiss_vec[Nb-1,0]   # no mismatch !
	tw.bety=1.0*twiss_vec[Nb-1,2]   # no mismatch !
	phasex,phasey =  phase_advance(twiss_vec[:,0],twiss_vec[:,2])
	tw.mux=0.5*phasex/pi
	tw.muy=0.5*phasey/pi
	tw.Dx=twiss_vec[Nb-1,4]
	tw.Dpx=twiss_vec[Nb-1,5]
	tw.angle=BeamLine[Nb-1].stwiss.angle
	tw.k1=BeamLine[Nb-1].stwiss.k1
	return tw

#------------------------------------------------------
# mismatch factor: can be obtained from beta functions
#-------------------------------------------------------

def mismatch_factor_xy(betx,alpx,betx0,alpx0):
	gamx=(1.0+alpx**2)/betx
	gamx0=(1.0+alpx0**2)/betx0
	delta_betx=betx-betx0
	delta_alpx=alpx-alpx0
	delta_gamx=gamx-gamx0
	deltax=delta_alpx**2-delta_betx*delta_gamx
	#mismatch_xy=np.sqrt(betx/betx0)
	#mismatch_xy=np.sqrt(1.0+1.0/2.0*(deltax+np.sqrt(deltax*(deltax+4.0))))-1.0
	mismatch_xy=(deltax+np.sqrt(deltax*(deltax+4.0)))/2.0
	return mismatch_xy


def betabeat_xy(twiss_vec):
	Nb=len(BeamLine)
	betx_vec=twiss_vec[:,0]
	bety_vec=twiss_vec[:,2]
	Nvec=len(betx_vec)
	s=zeros(Nvec)
	betx0=zeros(Nvec)
	bety0=zeros(Nvec)
	for l in range(int(Nvec/Nb)):
		for j, b in enumerate(BeamLine):
			s[j+l*Nb]=b.get_twiss().s
			betx0[j+l*Nb]=b.get_twiss().betx
			bety0[j+l*Nb]=b.get_twiss().bety
	return max(max(betx_vec/betx0),max(bety_vec/bety0))-1.0

def findElements(filename):
	f = open(filename)
	line = f.readline()
	while "$" not in line:
		line = f.readline()
		if "*" in line:
			names = line.split()

	line = f.readline()
	values = line.split()
	varsDict ={name:float(x) for name,x in zip(names[1:],values)}
	f.close()
	return varsDict

def matchBareTunes(qx0,qy0,skipMatching):
	f = open("match_base.str")
	w = open("match.str","w")
	if not skipMatching:
		for line in f:
			if "global" not in line.lower():
				w.write(line)
			else:
				w.write("GLOBAL,sequence=sis100ring,Q1={},Q2={};".format(qx0,qy0))
	else:
		w.write("k1nl_S00QD2F := k1nl_S00QD1F;")
	w.close()
	f.close()

def setQuads(dk,varsDict):
	w = open("tmp.str","w")
	tmp = zip(dk,varsDict.items())
	for x,(key,val) in tmp:
		w.write("{}:={};\n".format(quadDict[key],val+x))
	w.close()

def normalize(theta, alpha, groupDict):

	print("theta is {}".format(theta))

	tmp = zip(theta,groupDict.items())
	dk=[x*alpha/float(nQuads[name]) for x,(name,val) in tmp]
		
	print("dk is {}".format(dk))
	return np.array(dk)


def hyperSurface(theta,qx0,qy0,alpha):

	dk =normalize(theta, alpha, varsDict)
	print("hyper-surface computation")

	setQuads(dk,varsDict)

	set_beamline_from_madx(madx_file,temp_dir,beta0)
	qx = line_para["qx"]
	qy = line_para["qy"]

	print("current tunes are {} {}".format(qx,qy))
	out = np.sqrt((qx-qx0)**2+(qy-qy0)**2)
	print("the distance from set tunes hor:{} ver:{}".format(qx-qx0,qy-qy0))
	print("out is {}\n".format(out))

	return -out 

def observable(theta,sVar,sTwiss,betaX0,betaY0,alpha,quadDict):

	print("observable/metric computation")

	dk=normalize(theta,alpha,quadDict)
	setQuads(dk,quadDict)

	fftBeatX,fftBeatY=calculateBeat(sVar,sTwiss,betaX0,betaY0,emitx,emity,sigma_p,Ksc)

#	metric=calculateMax(fftBeatX,fftBeatY)
	metric=calculateMax(fftBeatX,fftBeatY)
	print("the observable/metric is {}\n".format(metric))

	return metric



def calculateBeat(sVar,sTwiss,betaX0,betaY0,emitx,emity,sigma_p,Ksc):

	set_beamline_from_madx(madx_file,temp_dir,beta0)
	twiss_vec=match_twiss_matrix(emitx,emity,sigma_p,Ksc)
	betaX,betaY = twiss_vec[:,0],twiss_vec[:,2]

	qx = line_para["qx"]
	qy = line_para["qy"]
	print("current tunes are {} {}".format(qx,qy))

	betX = np.interp(x=sVar, xp=sTwiss,fp=betaX)
	betY = np.interp(x=sVar, xp=sTwiss,fp=betaY)

	beatX = (betX-betaX0)
	beatY = (betY-betaY0)
	n = len(beatX)

	fftBeatX =np.abs(np.fft.rfft(beatX))/n
	fftBeatY =np.abs(np.fft.rfft(beatY))/n

	return fftBeatX,fftBeatY


def calculateMax(fftX,fftY):
	return np.sqrt(np.max(fftX)**2 + np.max(fftY)**2)

def calculateAperiodic(fftX,fftY,S=6):

	AperiodicHarmX = [x for i,x in enumerate(fftX) if i%S]
	AperiodicHarmY = [x for i,x in enumerate(fftY) if i%S]

	return np.sqrt(np.sum(AperiodicHarmX)**2 + np.sum(AperiodicHarmY)**2)


def correction(qx0,qy0,varsDict,sTwiss,betaX0,betaY0,sVar):

	method = "COBYLA"
	#method = "SLSQP"

	if not varsDict:
		print("problem!")
#		findElements(patternDict, quadsDict)
	print(varsDict)
		
	theta = np.zeros(len(varsDict))

	epsilon,ftol = 1e-1,1e-3
	# a normalization const
	alpha = (qx0**2+qy0**2)*(2*np.pi/sVar[-1])**2 # update!
	print(alpha)
	cons = [{"type": "eq", "fun":hyperSurface, "args":(qx0,qy0,alpha)}]		
	optionsDict = {'disp': True, "eps": epsilon, "ftol": ftol} 
	arg=(sVar,sTwiss,betaX0,betaY0,alpha,varsDict)

	if method == "COBYLA":
		for con in cons:
			con["type"]="ineq"
		optionsDict = {'rhobeg':epsilon, 'catol':1e-6, "tol":ftol,'disp': True}

	fIni = observable(theta,*arg)

	# !!!
	fftX,fftY = calculateBeat(sVar,sTwiss,betaX0,betaY0,emitx,emity,sigma_p,Ksc)

	aIni = calculateAperiodic(fftX,fftY)

	vec = som(observable, theta, method=method, constraints=cons, options=optionsDict,args=arg)

	# careful! you set dk here! not an absolute value!
	fCorr = observable(vec.x,*arg)

	# !!!
	fftX,fftY = calculateBeat(sVar,sTwiss,betaX0,betaY0,emitx,emity,sigma_p,Ksc)

	aCorr = calculateAperiodic(fftX,fftY)
	
	return fIni,fCorr,aIni,aCorr


if __name__ == "__main__":
	# beam velocity (to convert dispersion function from madx)
	beta0=0.567    # sis100 injection energy
	# example beam
	emitx=0.25*35.0e-6  #12.5e-6   # rms emittance_x
	emity=0.25*15.0e-6  #12.5e-6  # rms emittance_y
	sigma_p=2.0e-4  # rms momentum spread
#	Ksc=1.2e-7 # 1.5e-6     # space charge perveance
	Ksc=1.5e-6*0

	# TUNES
	qx0,qy0=18.84,18.73

	matchBareTunes(qx0,qy0,skipMatching=False)

	# first run madx (cold lattice)
	madx_file="sis100cold.madx"
	temp_dir="/home/dmitrii/tmp"

	set_beamline_from_madx(madx_file,temp_dir,beta0)
	Nb=len(BeamLine)
	sCold=zeros(Nb)
	betx0=zeros(Nb)
	bety0=zeros(Nb)
	for j, b in enumerate(BeamLine):
		sCold[j]=b.get_twiss().s
		betx0[j]=b.get_twiss().betx
		bety0[j]=b.get_twiss().bety


#=========================================================

	# the second run madx (warm lattice)
	madx_file="sis100.madx"

	set_beamline_from_madx(madx_file,temp_dir,beta0)
	Nb=len(BeamLine)
	sWarm=zeros(Nb)
	betx=zeros(Nb)
	bety=zeros(Nb)
	for j, b in enumerate(BeamLine):
		sWarm[j]=b.get_twiss().s
		betx[j]=b.get_twiss().betx
		bety[j]=b.get_twiss().bety


	sVar = np.linspace(0,sWarm[-1],2*len(betx))
#	twiss_vec=match_twiss_matrix(emitx,emity,sigma_p,0.0)
#	betxCold,betyCold = np.interp(x=sVar, xp=sCold,fp=twiss_vec[:,0],period=1083.6/6.0), np.interp(x=sVar, xp=sCold,fp=twiss_vec[:,2],period=1083.6/6.0)
	betxCold,betyCold = np.interp(x=sVar, xp=sCold,fp=betx0,period=1083.6/6.0), np.interp(x=sVar, xp=sCold,fp=bety0,period=1083.6/6.0)

#	twiss_vec=match_twiss_matrix(emitx,emity,sigma_p,0.0)
#	betxWarm,betyWarm = np.interp(x=sVar, xp=sWarm,fp=twiss_vec[:,0]), np.interp(x=sVar, xp=sWarm,fp=twiss_vec[:,2])
	betxWarm,betyWarm = np.interp(x=sVar, xp=sWarm,fp=betx), np.interp(x=sVar, xp=sWarm,fp=bety)

	plt.figure()
	plt.plot(sVar,100*(betxWarm/betxCold-1))
	plt.plot(sVar,100*(betyWarm/betyCold-1))

	plt.figure()
	plt.plot(sWarm,betx)
	plt.plot(sWarm,bety)

	plt.figure()
	plt.plot(sCold,betx0)
	plt.plot(sCold,bety0)

	plt.show()
