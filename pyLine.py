import string
import os
from numpy import *
from copy import copy, deepcopy
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import m_p

# Create a helper

class Helper:
	def __init__(self):
		self.qx = 0.0
		self.qy = 0.0
		self.quadDict = {}


# Create a simple MAD-like twiss object
# Contains the data from the twiss file for each element.

class Twiss:
	def __init__(self):
		self.s=0.0  # position of element
		self.L=0.0 # length of element
		self.alpx=0.0
		self.alpy=0.0
		self.betx=0.0
		self.bety=0.0
		self.mux=0.0
		self.muy=0.0
		self.Dx=0.0   # dispersion in x
		self.Dpx=0.0
		self.angle=0.0
		self.k1=0.0   # k1 multipole kick
		self.k2=0.0   # k2 multipole kick
		self.sccomp=0.0  # space charge compensation


# A sectormap object to store the 6x6 matrix for each element.

class SectorMap:

	def __init__(self, name, keyword):
		self.name=name
		self.keyword=keyword

		self.K=array([0.0,0.0,0.0,0.0,0.0,0.0])

		self.M=array([[1.0,0.0,0.0,0.0,0.0,0.0],
		              [0.0,1.0,0.0,0.0,0.0,0.0],
		              [0.0,0.0,1.0,0.0,0.0,0.0],
		              [0.0,0.0,0.0,1.0,0.0,0.0],
		              [0.0,0.0,0.0,0.0,1.0,0.0],
		              [0.0,0.0,0.0,0.0,0.0,1.0]])

		self.stwiss=Twiss()

	def from_matrix(self,M):
		self.M=M

	def from_twiss(self,TW):
		self.stwiss=TW

	# a simple sectormap for a constant focusing element of length s2-s1
	def oneturn(self,tunex,tuney,s1,s2,eta0,rhoB):
		self.stwiss.mux=2.0*pi*tunex
		self.stwiss.muy=2.0*pi*tuney
		self.stwiss.s=s2
		self.stwiss.L=s2-s1
		self.stwiss.betx=self.stwiss.L/self.stwiss.mux
		self.stwiss.bety=self.stwiss.L/self.stwiss.muy
		self.stwiss.k1=(self.stwiss.mux/self.stwiss.L)**2
		if (rhoB > 0.0):
			self.stwiss.angle=(s2-s1)/rhoB
			self.stwiss.Dx=1.0/(self.stwiss.k1*rhoB)

		self.M[0,0]=cos(self.stwiss.mux)
		self.M[0,1]=self.stwiss.L/self.stwiss.mux*sin(self.stwiss.mux)
		self.M[1,0]=-self.stwiss.mux/self.stwiss.L*sin(self.stwiss.mux)
		self.M[1,1]=cos(self.stwiss.mux)
		self.M[2,2]=cos(self.stwiss.muy)
		self.M[2,3]=self.stwiss.L/self.stwiss.muy*sin(self.stwiss.muy)
		self.M[3,2]=-self.stwiss.muy/self.stwiss.L*sin(self.stwiss.muy)
		self.M[3,3]=cos(self.stwiss.muy)
		self.M[4,5]=-eta0*self.stwiss.L
		if (rhoB > 0.0):
			self.M[0,5]=(1.0-cos(self.stwiss.mux))*(self.stwiss.L/self.stwiss.mux)**2*1.0/rhoB
			self.M[1,5]=sin(self.stwiss.mux)*self.stwiss.L/(rhoB*self.stwiss.mux)

	# simple linear kick
	def kick(self,kick_gradient_x,kick_gradient_y):
		self.M[0,0]=1.0
		self.M[0,1]=0.0
		self.M[1,0]=kick_gradient_x
		self.M[1,1]=1.0
		self.M[2,2]=1.0
		self.M[2,3]=0.0
		self.M[3,2]=kick_gradient_y
		self.M[3,3]=1.0


	def get_twiss(self):
		return self.stwiss

	def get_M(self):
		return self.M

	def get_K(self):
		return self.K


# A container list (beam line) for SectorMap Elements:
BeamLine=[]

# Global beam line parameters (extracted from madx twiss file)
line_para={"alpha0":1.0}

# empty beamline
def clear_bline():
	BeamLine[:]=[]

# most simple constant focusing beamline:
def constant_focusing(tunex,tuney,s1,s2,eta0,rhoB):
	SM=SectorMap('CF','CF')
	SM.oneturn(tunex,tuney,s1,s2,eta0,rhoB)
	BeamLine.append(SM)

#------------------------------------------------------
# Read MADX TFS file, Init BeamLine
#-------------------------------------------------------

def init_bline_twiss_madx(filename,beta0):
	# first clear beamline:
	BeamLine[:]=[]
	twissfile=open(filename,'r')
	line=twissfile.readline()

	H = Helper()
	while line.find('$START')==-1:
		line=twissfile.readline()
		if line.find('ALFA') != -1:
			words=line.split()
			line_para['alpha0']=float(words[3])

		arr=line.split()
		if "Q1" in arr:
			line_para['qx']=float(arr[3])
		if "Q2" in arr:
			line_para['qy']=float(arr[3])


	TW0=Twiss()
	while line.find('$END')==-1:
		line=twissfile.readline()
		if line.find('$END') >=0: break
		# split line and convert to float if necessary:
		words=line.split()
		TW=Twiss()
		TW1=Twiss()
		SM=SectorMap(words[0],words[1])
		TW.s=float(words[2])
		TW.L=float(words[3])
		TW.alpx=float(words[4])
		TW.alpy=float(words[5])
		TW.betx=float(words[6])
		TW.bety=float(words[7])
		TW.Dx=float(words[8])
		TW.Dpx=float(words[9])
		TW.mux=float(words[10])
		TW.muy=float(words[11])
		TW.angle=float(words[12])
		TW.k1=float(words[13])
		TW.k2=float(words[14])

	    #correct Dispersion
		TW.Dx*=beta0
		TW.Dpx*=beta0

	    # init BeamLine
		SM.from_twiss(TW)
		BeamLine.append(SM)


    # close twissfile:
	twissfile.close()

#-------------------------------------------------------

def read_bline_maps_madx(filename,beta0):
	BLineLen=len(BeamLine)
	if BLineLen == 0:
		print("BLineLen=0")
		return

	madfile=open(filename,'r')

	line=madfile.readline()
	while line.find('$START')==-1:
		line=madfile.readline()

	for b in BeamLine:
		line=madfile.readline()
		words=line.split()
		for l in range(6):
			b.K[l]=float(words[2+l])
		for l in range(6):
			for j in range(6):
				b.M[j,l]=float(words[2+6+j+6*l])
		# correct for beta0 (we use z, delta; MAD tau, delta E):
		b.M[4,5]*=beta0**2
		b.M[5,4]/=beta0**2
		b.M[0,4]/=beta0
		b.M[0,5]*=beta0
		b.M[4,0]*=beta0
		b.M[5,0]/=beta0
		b.M[1,4]/=beta0
		b.M[1,5]*=beta0
		b.M[4,1]*=beta0
		b.M[5,1]/=beta0
		b.M[2,4]/=beta0
		b.M[2,5]*=beta0
		b.M[4,2]*=beta0
		b.M[5,2]/=beta0
		b.M[3,4]/=beta0
		b.M[3,5]*=beta0
		b.M[4,3]*=beta0
		b.M[5,3]/=beta0

	madfile.close()


# the electron lens: If there is an element 'DRLENS' in the MADX file, it adds a gradient.

def add_sccomp(sccomp):
	for b in BeamLine:
		if b.name.find('ELENS')==1:
			b.stwiss.sccomp=sccomp


def get_bline_matrix():
	Mc=array([[1.0,0.0,0.0,0.0,0.0,0.0],
		              [0.0,1.0,0.0,0.0,0.0,0.0],
		              [0.0,0.0,1.0,0.0,0.0,0.0],
		              [0.0,0.0,0.0,1.0,0.0,0.0],
		              [0.0,0.0,0.0,0.0,1.0,0.0],
		              [0.0,0.0,0.0,0.0,0.0,1.0]])
	for b in BeamLine:
		Mc_tmp=dot(b.M,Mc)
		Mc=Mc_tmp
	return Mc


# calculate symplectic error  (for academic purposes)
def get_bline_symplectic_error():
	Mc=get_bline_matrix()[0:4,0:4]
	# symplectic matrix
	Ssym=array([[0.0,1.0,0.0,0.0],[-1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,-1.0,0.0]])
	# symplectic error
	eta_sym=linalg.norm(dot(Mc.T,dot(Ssym,Mc))-Ssym)
	return eta_sym, dot(Mc.T,dot(Ssym,Mc))


def get_bline_phase_advance():
	Mc=get_bline_matrix()
	return 180.0*arccos(0.5*(Mc[0,0]+Mc[1,1]))/pi, 180.0*arccos(0.5*(Mc[2,2]+Mc[3,3]))/pi


def get_bline_element(s):
	Nb=len(BeamLine)
	Lb=BeamLine[Nb-1].stwiss.s
	if s > Lb:
		s=s-floor(s/Lb)*Lb
	if BeamLine[0].stwiss.s >= s and s >= 0.0:
		return BeamLine[0].stwiss
	for j in xrange(1,Nb):
		if BeamLine[j-1].stwiss.s < s and BeamLine[j].stwiss.s >= s :
			return BeamLine[j].stwiss
		if BeamLine[Nb-1].stwiss.s < s :
			return BeamLine[0].stwiss
		if s < 0.0 :
			BeamLine[Nb-1].stwiss
			print("error: s not in range")
	return BeamLine[0].stwiss



def get_bline_length():
	Nb=len(BeamLine)
	return BeamLine[Nb-1].stwiss.s

def get_bline_chroma():
	chromax=0.0
	chromay=0.0
	for b in BeamLine:
		if (b.stwiss.angle > 0.0):
			chromax+=b.M[1,0]*b.stwiss.betx/(2.0*pi) # why only 2 pi in sbends ?
		else:
			chromax+=b.M[1,0]*b.stwiss.betx/(4.0*pi)
		chromay+=b.M[3,2]*b.stwiss.bety/(4.0*pi)

		chromax+=b.stwiss.k2*b.stwiss.Dx*b.stwiss.betx/(4.0*pi)
		chromay-=b.stwiss.k2*b.stwiss.Dx*b.stwiss.bety/(4.0*pi)

	return chromax,chromay


#------------------------------------------------------
# Change tunes in MADX file. Run MADX.
# Read MADX lattice file into BeamLine.
# temporary files in temp_dir.
#-------------------------------------------------------

def set_beamline_from_madx(madx_file,temp_dir,beta0):

	mad_in_file=open(madx_file,'r')
	file_id=str(os.getpid()) # str(phasex_deg)+str(phasey_deg)
	mad_out_file=open('%s/temp%s.madx' %(temp_dir,file_id),'w')

	for line in mad_in_file:

#		if (line.find('GLOBAL')!=-1) and (line.find('DQ1') == -1) and (mux0 > 0.0):
#			line='GLOBAL,sequence=cella,Q1=%f,Q2=%f;\n' %(mux0,muy0)  # !!!! cella !!!!
		if line.find('twiss.txt')!=-1:
			line='twiss, chrom, file=%s/twiss%s.txt;\n' %(temp_dir,file_id)
		if line.find('sectormap.txt')!=-1:
			line='twiss,sectormap,sectorfile=%s/sectormap%s.txt;\n' %(temp_dir,file_id)
		mad_out_file.write(line)

	mad_in_file.close()
	mad_out_file.close()
	# run madx
	os.system("./madx < %s/temp%s.madx > %s/out%s.dat" %(temp_dir,file_id,temp_dir,file_id))
	# read twiss.txt and sectormap:
	init_bline_twiss_madx("%s/twiss%s.txt" %(temp_dir,file_id),beta0)
	read_bline_maps_madx("%s/sectormap%s.txt" %(temp_dir,file_id),beta0)
	#print os.getpid(), "length: ",get_bline_length()
	print(file_id, "phase advances 0 : ",get_bline_phase_advance())
	# delete temporary files
	os.remove("%s/temp%s.madx" %(temp_dir,file_id))
	os.remove("%s/out%s.dat" %(temp_dir,file_id))
	os.remove("%s/twiss%s.txt" %(temp_dir,file_id))
	os.remove("%s/sectormap%s.txt" %(temp_dir,file_id))


def set_beamline_cf(phasex_deg,phasey_deg,eta0,rhoB,length,nsm):
	mux0=phasex_deg/360.0 # tunes
	muy0=phasey_deg/360.0 # tunes
	for j in range(nsm):
		constant_focusing(mux0/nsm,muy0/nsm,j*length/nsm,(j+1.0)*length/nsm,eta0,rhoB)


# transform AG lattice into symmetric focusing (for academic purposes)
def make_symmetric():
	for b in BeamLine:
		b.M[2,2]=b.M[0,0]
		b.M[2,3]=b.M[0,1]
		b.M[3,2]=b.M[1,0]
		b.M[3,3]=b.M[1,1]
		b.stwiss.alpy=b.stwiss.alpx
		b.stwiss.bety=b.stwiss.betx

# example: read madx file and print elements
if __name__ == "__main__":
	# run madx
	e_kin=200.0 # MeV
	gamma0=1.0+(e_kin*1e6*qe)/(m_p*clight*clight)
	beta0=sqrt((gamma0*gamma0-1.0)/(gamma0*gamma0))
	madx_file="sis100.madx"
	temp_dir="/tmp"
	phasex_deg=3.14*360.0   # 60.1
	phasey_deg=3.16*360.0 # 60.1
	set_beamline_from_madx(madx_file,temp_dir,phasex_deg,phasey_deg,beta0)
	print("length: ",get_bline_length())
	print("fractional phase advances: ",get_bline_phase_advance())
	print("alpha0: ",line_para["alpha0"])
	print("chromax:", get_bline_chroma())
	for b in BeamLine:
	 	print("name: ", b.name,"key: ",b.keyword ,"L: ",b.stwiss.L,"s: ",b.stwiss.s, "K1:", b.K[0],"K2:",b.K[2])
