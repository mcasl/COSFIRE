#   Keypoint detection by a COSFIRE filter
#   Version 02/09/2017
#   Implementation in python by: Daniel Acosta
#
#   COSFIRE is an algorithm created by George Azzopardi and Nicolai Petkov,
#   for more details view original paper:
#   George Azzopardi and Nicolai Petkov, "Trainable COSFIRE filters for
#   keypoint detection and pattern recognition", IEEE Transactions on Pattern 
#   Analysis and Machine Intelligence, vol. 35(2), pp. 490-503, 2013.

import numpy as np
import cv2
import GaborFilter as GF
from peakutils.peak import indexes
import DoGFilter as DoG


class COSFIRE:
    def __init__(self, image):
        self.patternImage=image#pattern prototipe Image
        self.nameFilter='NN'#basic filter (Gabor,DoG, etc.)
        self.coorX=0#center of COSFIRE
        self.coorY=0#" "
        self.rhoList=[] #rho list
        self.eta=0 #
        self.t1=0 #
        self.filterParam=[] #Parameters of filter
        self.operator=[] #Struct of filter COSFIRE (S_f)
        self.input={}#Bank of responses pattern Image
        self.maxi=0 #Maximum response
        self.t2=0#
        self.alpha=0#
        self.sigma0=0#
        self.t3=0#
        self.invarianteReflexion=0#Parameter reflection Yes=1 No=0
        self.invarianteEscala=[]#scale parameters (default=1)
        self.invarianteRotacion=[]#rotation parameters (default=0)
        self.tuple={}#Bank of responses to image
        self.operator1=[]#operator invariant to rotation, escala and reflection 
    
    #1-Configuration COSFIRE Filter
    def configureCOSFIRE(self):
        self.input=self.getResponseFilter()#1.1
        self.suppresResponsesT1()#1.2
        self.operator=self.getCOSFIRETuples(self.input,self.coorX,self.coorY)#1.3

    #2-Apply the COSFIRE filtler
    def applyCOSFIRE(self, Image,):
        inputImage=Image
        #print "calculando banco"
        self.tuple=self.computeTuples(inputImage) #2.1
        self.tuple=self.shiftResponses(self.tuple) #2.2
        #print "calculando respuesta"
        output=self.iReflexionCOSFIRE(inputImage,self.operator,self.tuple) #2.3
        mx=0
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i][j]>mx:
                    mx=output[i][j]
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i][j]<mx*self.t3:
                    output[i][j]=0
        return output
        
    # (1.1) Get response filter
    def getResponseFilter(self):
        if self.nameFilter=='Gabor':
            return GF.getGaborResponse(self.patternImage,self.filterParam,
                                       self.filterParam[2],self.filterParam[3])
        elif self.nameFilter=='DoG':
            return DoG.getDoGResponse(self.patternImage, self.filterParam, self.filterParam[0])
    
    # (1.2) Suppres Resposes
    def suppresResponsesT1(self):
        eMax = [self.input[i].max() for i in self.input]
        self.maxi=max(eMax)
        for clav in self.input:
            for k in range(self.input[clav].shape[0]):
                for l in range(self.input[clav].shape[1]):
                    if self.input[clav][k][l]<self.t1*self.maxi:
                        self.input[clav][k][l]=0
        return  
    
    #(1.3) Get descriptor set (Sf)
    def getCOSFIRETuples(self,bank,xc,yc):
        operator=[]
        phiList=np.arange(360)*np.pi/180.0 #Discretizacion del circulo
        for i in range(len(self.rhoList)):  #Iteramos en lista de radios
            if self.rhoList[i]==0: #Caso rho=0
                if self.nameFilter=='Gabor':
                    for k in range(self.filterParam[2].size):
                        ind=0
                        val=-1
                        tupla=np.zeros(4)
                        for l in range(self.filterParam[3].size):
                            par=(self.filterParam[2][k],self.filterParam[3][l])
                            if self.input[par][xc][yc] > self.maxi*self.t2:
                                ind=l
                                val=self.input[par][xc][yc]
                        if val > -1:
                            tupla[2]=self.filterParam[3][ind]
                            tupla[3]=self.filterParam[2][k]
                            operator.append(tupla)
                elif self.nameFilter=='DoG':
                    for k in range(self.filterParam[0].size):
                        if self.input[self.filterParam[0][k]][xc][yc] > self.maxi*self.t2:
                            tupla=np.zeros(3)
                            tupla[2]=self.filterParam[0][k]
                            operator.append(tupla)
            elif self.rhoList[i]>0: #Caso rho>0
                listMax=np.zeros(360)
                direcciones=[]
                for k in range(phiList.size):
                    yi=int(yc+np.floor(self.rhoList[i]*np.cos(phiList[k])))
                    xi=int(xc-np.floor(self.rhoList[i]*np.sin(phiList[k])))
                    val=0
                    nr=self.patternImage.shape[0]
                    nc=self.patternImage.shape[1]
                    if xi>=0 and yi>=0 and xi<nr and yi<nc:
                        for l in self.input:
                            if self.input[l][xi][yi]> val:
                                val=self.input[l][xi][yi]
                    listMax[k]=val
                    direcciones.append((xi,yi))
                ss = 360 // 16
                #nn=np.arange(360)
                #plt.plot(nn,listMax)       
                if len(np.unique(listMax))==1:
                    continue
                listMax1=np.zeros(listMax.size+1)
                for p in range(listMax.size):
                    listMax1[p+1]=listMax[p]
                index = indexes(listMax1, thres=0.2, min_dist=ss)
                index=list(index-1)
                index=np.array(index)
                for k in range(index.size):
                    if self.nameFilter=='Gabor':
                        for l in range(self.filterParam[2].size):
                            mx=-1
                            ind=0
                            for m in range(self.filterParam[3].size):
                                par=(self.filterParam[2][l],self.filterParam[3][m])
                                var=self.input[par][direcciones[index[k]][0]][direcciones[index[k]][1]]
                                if var > self.t2 * self.maxi and mx < var:
                                    mx=var
                                    ind=m
                            if mx !=-1:
                                tupla=np.zeros(4)
                                tupla[0]=self.rhoList[i]
                                tupla[1]=index[k]*(np.pi/180.0)
                                tupla[2]=self.filterParam[3][ind]
                                tupla[3]=self.filterParam[2][l]
                                operator.append(tupla)
                    elif self.nameFilter=='DoG':
                        for l in self.input:
                            var=self.input[l][direcciones[index[k]][0]][direcciones[index[k]][1]]
                            if var > self.t2*self.maxi:
                                tupla=np.zeros(3)
                                tupla[0]=self.rhoList[i]
                                tupla[1]=index[k]*(np.pi/180.0)
                                tupla[2]=l
                                operator.append(tupla)
        return operator

    #2.1 For each tuple in the set Sf compute response
    def computeTuples(self,inputImage):
        normal = [self.operator[i] for i in range(len(self.operator))]
        ope = [normal]
        if self.invarianteReflexion==1:
            refleccion=[]
            for i in range(len(self.operator)):
                a=(self.operator[i][0],np.pi-self.operator[i][1],self.operator[i][2],np.pi-self.operator[i][3])
                refleccion.append(a)
            ope.append(refleccion)
        for item in ope:
            for l in range(len(item)):
                for j in range(len(self.invarianteRotacion)):
                    for k in range(len(self.invarianteEscala)):
                        if self.nameFilter=='Gabor':
                            aux=np.zeros(4)
                            aux[0] = item[l][0] * self.invarianteEscala[k]
                            aux[1] = item[l][1] + self.invarianteRotacion[j]
                            aux[2] = item[l][2] * self.invarianteEscala[k]
                            aux[3] = item[l][3] + self.invarianteRotacion[j]
                            self.operator1.append(aux)
                        elif self.nameFilter=='DoG':
                            aux=np.zeros(3)
                            aux[0] = item[l][0] * self.invarianteEscala[k]
                            aux[1] = item[l][1] + self.invarianteRotacion[j]
                            aux[2] = item[l][2] * self.invarianteEscala[k]
                            self.operator1.append(aux)
        unicos={}
        for i in range(len(self.operator1)):
            if self.nameFilter=='Gabor':
                a=(self.operator1[i][3],self.operator1[i][2])
                if a not in unicos:
                    l1=np.array(1*[a[0]])
                    l2=np.array(1*[a[1]])
                    tt=GF.getGaborResponse(inputImage,self.filterParam,l1,l2)
                    unicos[a]=tt[a]
            elif self.nameFilter=='DoG':
                if self.operator1[i][2] not in unicos:
                    l1=np.array(1*[self.operator1[i][2]])
                    tt=DoG.getDoGResponse(inputImage,self.filterParam,l1)
                    unicos[self.operator1[i][2]]=tt[self.operator1[i][2]]
        max=0
        for i in unicos:
            t=unicos[i].shape
            for j in range(t[0]):
                for k in range(t[1]):
                    sig=unicos[i][j][k]
                    if sig > max:
                        max=unicos[i][j][k]
        for i, value in unicos.items():
            t=unicos[i].shape
            for j in range(t[0]):
                for k in range(t[1]):
                    if value[j][k] < max * self.t1:
                        unicos[i][j][k]=0
        return self.blurGaussian(unicos)
    
    #(2.1.1)Blurr
    def blurGaussian(self,bankFilters):
        dic={}
        for i in range(len(self.operator1)):
            if self.nameFilter=='Gabor':
                a1=(self.operator1[i][3],self.operator1[i][2])
                sigma=self.alpha*self.operator1[i][0] + self.sigma0
                #cv2.imshow("wsw",cv2.GaussianBlur(bankFilters[a1], (15,15),sigma, sigma))
                a2=(self.operator1[i][0],self.operator1[i][1],self.operator1[i][2],self.operator1[i][3])
                if a2 not in dic:
                    dic[a2]=cv2.GaussianBlur(bankFilters[a1], (9,9),sigma, sigma)
            elif self.nameFilter=='DoG':
                sigma=self.alpha*self.operator1[i][0] + self.sigma0
                a2=(self.operator1[i][0],self.operator1[i][1],self.operator1[i][2])
                dic[a2]=cv2.GaussianBlur(bankFilters[self.operator1[i][2]], (9,9),sigma, sigma)
        return dic
     
    #(2.2) Shift
    def shiftResponses(self,resp):
        Resp={}
        for clave,img in resp.items():
            nr=img.shape[0]
            nc=img.shape[1]
            x=clave[0]*np.sin(np.pi+clave[1])
            y=clave[0]*np.cos(np.pi+clave[1])
            nw=np.copy(img)
            for k in range(nr):
                for l in range(nc):
                    xx=int(k+x)
                    yy=int(l-y)
                    nw[k][l] = img[xx][yy] if xx>=0 and xx<nr and yy>=0 and yy<nc else 0
            Resp[clave]=nw
        return Resp
    
    #(2.3) invariant under reflection
    def iReflexionCOSFIRE(self,imagen,operator,respBank):
        out1=self.iRotationCOSFIRE(imagen,operator,respBank)
        #cv2.imshow("dsds1111",out1)
        if self.invarianteReflexion==1:
            operatorI=[]
            for i in range(len(operator)):
                a=(operator[i][0],np.pi-operator[i][1],operator[i][2],np.pi-operator[i][3])
                operatorI.append(a)
            out2=self.iRotationCOSFIRE(imagen,operatorI,respBank)
            #cv2.imshow("dsfsd222222",out2)
            self.returnMaxi(out2,out1) ##Definir maxi
        return out1
        
    #(2.4) invariant to rotation
    def iRotationCOSFIRE(self,imagen,operator,respBank):
        output=np.zeros((imagen.shape[0],imagen.shape[1]))
        for i in range(len(self.invarianteRotacion)):
            operatorRotacion=[]
            for j in range(len(operator)):
                if self.nameFilter=='Gabor':
                    a=(operator[j][0],operator[j][1]+self.invarianteRotacion[i],operator[j][2],operator[j][3]+self.invarianteRotacion[i])
                elif self.nameFilter=='DoG':
                    a=(operator[j][0],operator[j][1]+self.invarianteRotacion[i],operator[j][2])
                operatorRotacion.append(a)
            outR=self.iEscalaCOSFIRE(imagen,operatorRotacion,respBank)
            self.returnMaxi(outR,output)
        return output
    
    #(2.5) invariant to scale
    def iEscalaCOSFIRE(self,imagen,operator,respBank):
        output=np.zeros((imagen.shape[0],imagen.shape[1]))
        for i in range(len(self.invarianteEscala)):
            operatorEscala=[]
            for j in range(len(operator)):
                if self.nameFilter=='Gabor':
                    a=(operator[j][0]*self.invarianteEscala[i],operator[j][1],operator[j][2]*self.invarianteEscala[i],operator[j][3])
                elif self.nameFilter=='DoG':
                    a=(operator[j][0]*self.invarianteEscala[i],operator[j][1],operator[j][2]*self.invarianteEscala[i])
                operatorEscala.append(a)
            outR=self.computeResponse(respBank,operatorEscala)
            self.returnMaxi(outR,output)
        return output
    
    #Compute response
    def computeResponse(self,conj,operator):
        nr=conj[operator[0]].shape[0]
        nc=conj[operator[0]].shape[1]
        resp=np.zeros((nr,nc))
        rhoMax=0
        for i in range(len(operator)):
            if operator[i][0]>rhoMax:
                rhoMax=operator[i][0]
        tupleweightsigma= np.sqrt(-((rhoMax**2 )/ (2*np.log(0.5) ) ))
        for i in range(nr):
            for j in range(nc):
                val=1
                suma=0
                for k in range(len(operator)):
                    aux=np.exp(-(operator[k][0]**2)/(2*tupleweightsigma**2))
                    #if self.nameFilter=='DoG':
                    #    aux=1
                    suma+=aux
                    wi=aux
                    val*=(conj[operator[k]][i][j]**wi)
                val=np.power(val,1.0/suma)
                resp[i][j]=val
        return resp
    
    #Select the maximum of two answers
    def returnMaxi(self,mat1,mat2):
        for i in range(mat1.shape[0]):
            for j in range(mat1.shape[1]):
                mat2[i][j]=max(mat1[i][j],mat2[i][j])