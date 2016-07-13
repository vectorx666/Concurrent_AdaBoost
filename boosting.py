import numpy
from scipy import *
import numpy.random as npr
from inspect import getargspec
from sys import *
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pp
import math
from astroML.datasets import fetch_dr7_quasar
from astroML.datasets import fetch_sdss_sspp
from sklearn.datasets import load_svmlight_file
#import pyroc
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score
import time
import warnings

def load_csv(filename, cols, sep = ',', skip = 0):
    from numpy import loadtxt
    data = loadtxt(filename, delimiter = sep, usecols = cols, skiprows = skip)
    return data

def cargartweets(archivo):
	print "Archivo Cargado"
	datos=[]
	with open(archivo,'r') as arch:
		for linea in arch:
			largo=len(linea)
			linea=linea[1:largo-1]
			lista=linea.strip().replace("[","").replace("]","").replace(" ","").split(",")
			temp=lista[len(lista)-1]
			lista.pop()
			lista.pop()
			lista.pop()
			lista.append(temp)
			lista2=map(float,lista)
			datos.append(lista2)
	return array(datos)
    
def cargardataset(nombrearchivo):

	archivo=open(nombrearchivo)
	dataset=list()
	#dataset=dataset[1:,:]
	i=1
	for linea in archivo:
		if i==1:
			i=i+1
			continue
		linea=linea.replace(';',',').replace('\t',',').replace('+','').replace('b','-1.0').replace('s','1.0').replace('g','1.0').replace('b','-1.0').replace('M','1.0').replace('B','-1.0')
		linea=linea.strip().split(',')
		linea=array(linea)
		dataset.append(map(float,linea))
		i=i+1
	return array(dataset)

def cargardatasettemp(nombrearchivo):

	archivo=open(nombrearchivo)
	dataset=list()
	i=1
	for linea in archivo:
		print 'LINEA '
		if i==1:
			i+=1
			#print linea
			continue
		linea=linea.replace(';',',').replace('\t',',').replace('b','-1.0').replace('s','1.0').replace('g','1.0').replace('b','-1.0').replace('M','1.0').replace('B','-1.0')
		linea=linea.strip().split(',')
		#print linea
		dataset.append(map(float,linea))	
	#return array(dataset,dtype=float)
	return array(dataset)

def bubblesort(arr):
    done = False
    while not done:
		done = True
    for i in range(len(arr)-1):
		if arr[i] > arr[i+1]:
			arr[i], arr[i+1] = arr[i+1], arr[i]
			done = False
    return arr
	

def remuestrar(datos,distribucion,semilla,porcentaje):
	nuevadata=[]
	fil,col=datos.shape	
	distriacum=numpy.zeros(fil+1)
	for j in range(len(distribucion)):
		distriacum[j+1]=distriacum[j]+distribucion[j]
	#print distriacum
	#muestra=bubblesort(numpy.random.rand(1,fil))
	numpy.random.seed(semilla)
	muestra=numpy.random.rand(1,int(fil*porcentaje))
	largoacum=len(distriacum)
	#print 'Muestra: ', muestra[0]
	for num in muestra[0]:
		for j in range(1,largoacum):	
			if distriacum[j-1]<num<distriacum[j]:
				nuevadata.append(datos[j-1,:])
				break;
	#nuevadata=numpy.array(nuevadata)
	#print 'NUEVA DATA: ',nuevadata[:,col]
	#if nuevadata[:,col].count(nuevadata[0,col])!=len(nuevadata[:,col]):
	#	print 'Misma CLASE: '
	
	return nuevadata

def performance(Y_pred,Y_real):
	accu=accuracy_score(Y_real,Y_pred)
	roc=roc_auc_score(Y_real,Y_pred)
	f1=f1_score(Y_real,Y_pred)
	prec=precision_score(Y_real,Y_pred)
	rec=recall_score(Y_real,Y_pred)
	return [accu,roc,f1,prec,rec]
	
def ejec_modelo(X,Y,i):

		if i==1:	
			modelo=sklearn.naive_bayes.GaussianNB()
		elif i==2:	
			modelo=sklearn.tree.DecisionTreeClassifier(max_depth=2)
		else: 
			modelo=sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
		modelo.fit(X,Y)
		return modelo	
	
def entrenarmodelo(X,Y,i):	
	#print 'Entreno'
	return ejec_modelo(X,Y,i)

def load_astro_dataset():
    """
    The BUPA dataset can be obtained from
    http://www.cs.huji.ac.il/~shais/datasets/ClassificationDatasets.html
    See description of this dataset at
    http://www.cs.huji.ac.il/~shais/datasets/bupa/bupa.names
    """
    quasars = fetch_dr7_quasar()
    stars = fetch_sdss_sspp()
    quasars = quasars[::5]
    stars = stars[::5]
    
    Nqso = len(quasars)
    #print 'Numero quasars: ',Nqso
    Nstars = len(stars)
    #print 'Numero estrellas: ',Nstars
    X = empty((Nqso + Nstars, 4), dtype=float)
    
    X[:Nqso, 0] = quasars['mag_u'] - quasars['mag_g']
    X[:Nqso, 1] = quasars['mag_g'] - quasars['mag_r']
    X[:Nqso, 2] = quasars['mag_r'] - quasars['mag_i']
    X[:Nqso, 3] = quasars['mag_i'] - quasars['mag_z']
    
    X[Nqso:, 0] = stars['upsf'] - stars['gpsf']
    X[Nqso:, 1] = stars['gpsf'] - stars['rpsf']
    X[Nqso:, 2] = stars['rpsf'] - stars['ipsf']
    X[Nqso:, 3] = stars['ipsf'] - stars['zpsf']
    
    y = zeros(Nqso + Nstars, dtype=int)
    y[:Nqso] = 1
    y[y==0]=-1
    #print 'Salida', y
    stars=map(tuple,stars)
    quasars=map(tuple,quasars)
    stars=array(stars)
    quasars=array(quasars)
   
    #print "Tamano Astro: ", len(X)
    return X, y	

def main(argv):
	
	#print 'Starting pp with ',job_server1.get_ncpus(),' workers' 
	#Numero CPUS
	
	for porcentaje in [0.01,0.1,0.5,1]:
		for numerocpus in [1,5,15,25,35,39]:
			#numerocpus=7
			cpusstr=str(numerocpus)
			porcstr=str(porcentaje)
			
			resultados=[]
			
			
			for experimento in range(10):
				
				print 'Experimento: ',experimento+1
				T=int(argv[1])
				
				
				if argv[2]=='cancer':
					datos=cargardataset("wdbc.data")
					filas,columnas=datos.shape
					temp=numpy.copy(datos[:,1])
					datos[:,1]=datos[:,columnas-1]
					datos[:,columnas-1]=temp
					datos=datos[:,1:]
					#print temp
					
				if argv[2]=='astro':
					X,y=load_astro_dataset()
					#print 'Dime y: ',len(y)
					ejemplos=len(y)
					y=y.reshape((ejemplos,1))
					datos=concatenate((X,y),axis=1)
					
				if argv[2]=='ionosphere':
					datos=cargardataset("ionosphere.data")
					
				if argv[2]=='higgs':
					datos=cargardataset('training.csv')
					print size(datos)
					datos=datos[:,1:]
					
				if argv[2]=='diabetes':
					#datos=readcsv('diabetes_scale.txt')
					x,y=load_svmlight_file('diabetes_scale')
					x=x.todense()
					print x.shape
					y=y.reshape((len(y),1))
					
					datos=numpy.concatenate((x,y),axis=1)
					datos=array(datos)
				if argv[2]=='heart':
					#datos=readcsv('diabetes_scale.txt')
					x,y=load_svmlight_file('heart_scale')
					x=x.todense()
					print x.shape
					y=y.reshape((len(y),1))
					
					datos=numpy.concatenate((x,y),axis=1)
					datos=array(datos)	
				if argv[2]=='liver':
					#datos=readcsv('diabetes_scale.txt')
					x,y=load_svmlight_file('liver-disorders_scale')
					x=x.todense()
					print x.shape
					for i in range(len(y)):
						if y[i]==2:
							y[i]=-1
					y=y.reshape((len(y),1))
					
					datos=numpy.concatenate((x,y),axis=1)
					datos=array(datos)
				if argv[2]=='twitter':
					#datos=readcsv('diabetes_scale.txt')
					datos=cargartweets("clasificadostwitter.txt")	
					print "Tamano Twitter: ", len(datos)
				if argv[2]=='mushrooms':
					#datos=readcsv('diabetes_scale.txt')
					x,y=load_svmlight_file('mushrooms')
					x=x.todense()
					print x.shape
					for i in range(len(y)):
						if y[i]==2:
							y[i]=-1
					y=y.reshape((len(y),1))
					
					datos=numpy.concatenate((x,y),axis=1)
					datos=array(datos)	
				
				permutacion=list(datos)
				random.shuffle(permutacion)
				permutacion=array(permutacion)
				datos=permutacion
				
				fil,col=datos.shape
				
				entre=datos[0:int(0.8*fil),:]
				test=datos[int(0.8*fil):,:]
				fil,col=entre.shape

				X_test=test[:,:col-1]
				Y_test=test[:,col-1]

				Y_train=entre[:,col-1]
				
				ejemplostotal,atributos=entre.shape
				
				#print 'Salidas: ',Y_train
				
				fil,col=entre.shape	
				
				# distbag son los pesos de cada dato (al comienzo 1/n)
				distbag1=(numpy.zeros(fil)+1.0/fil)
				distbag2=(numpy.zeros(fil)+1.0/fil)
				distbag3=(numpy.zeros(fil)+1.0/fil)
				RULES=[]
				ALPHA=[]
				
				
				iteracion=1
				hipotesis=[]
				
				hipo_final_train=[]
				hipo_final_test=[]
				Prop_hipo_final_train=[]
				Prop_hipo_final_test=[]
				error_final_test=0
				while iteracion<=T:
					ppservers1=()
					job_server1 = pp.Server(ppservers=ppservers1)
					ppservers2=()
					job_server2 = pp.Server(ppservers=ppservers2)
					archivo1=open('DockerServerNB_errorpond_'+argv[2]+cpusstr+'_'+porcstr+'.txt','a')
					archivo2=open('DockerServerDT_errorpond_'+argv[2]+cpusstr+'_'+porcstr+'.txt','a')
					archivo3=open('DockerServerQDA_errorpond_'+argv[2]+cpusstr+'_'+porcstr+'.txt','a')
					Xnuevo1=[]
					Ynuevo1=[]
					Xnuevo2=[]
					Ynuevo2=[]
					Xnuevo3=[]
					Ynuevo3=[]
					resultados1=[]
					resultados2=[]
					resultados3=[]
					X_final=array(entre[:,:col-1])
					Y_final=array(entre[:,col-1])
					
					#Paralelo NB	
					jobs = [(num, job_server1.submit(remuestrar,(entre,distbag1,num*random.randint(1,1000),porcentaje,),(bubblesort,asanyarray,),("numpy",))) for num in range(numerocpus)]
					datosnuev1=[]
					for numero,trab in jobs:
						datosnuev1.append(array(trab()))
						while not(trab.finished):
								i=1	
					for datos in datosnuev1:
						Xnuevo1.append(datos[:,:col-1])
					
						Ynuevo1.append(datos[:,col-1])
					#Paralelo DT	
					jobs = [(num, job_server1.submit(remuestrar,(entre,distbag2,num*random.randint(1,1000),porcentaje,),(bubblesort,asanyarray,),("numpy",))) for num in range(numerocpus)]
					datosnuev2=[]
					for numero,trab in jobs:
						datosnuev2.append(array(trab()))
						while not(trab.finished):
								i=1	
					for datos in datosnuev2:
						Xnuevo2.append(datos[:,:col-1])
					
						Ynuevo2.append(datos[:,col-1])
					#Paralelo QDA	
					jobs = [(num, job_server1.submit(remuestrar,(entre,distbag3,num*random.randint(1,1000),porcentaje,),(bubblesort,asanyarray,),("numpy",))) for num in range(numerocpus)]
					datosnuev3=[]
					for numero,trab in jobs:
						datosnuev3.append(array(trab()))
						while not(trab.finished):
								i=1	
					for datos in datosnuev3:
						Xnuevo3.append(datos[:,:col-1])
					
						Ynuevo3.append(datos[:,col-1])
					#Medicion Tiempo Inicio

					#Train Naive Bayes##########################################################################################
					start_time = time.time()
					jobs2 = [(num, job_server2.submit(entrenarmodelo,(Xnuevo1[num],Ynuevo1[num],1),(ejec_modelo,),("sklearn.tree","sklearn.discriminant_analysis","sklearn.naive_bayes",))) for num in range(numerocpus)]	
					
					modelos_ens1=[]
				
					for numero2,trab2 in jobs2:
						modelos_ens1.append(trab2())
						while not(trab2.finished):
							i=1
					#job_server2.print_stats()
					#job_server2.destroy()
					#Medicion Tiempo Final
					finaltime1=(time.time() - start_time)

					#Train Decision Tree#########################################################################################
					start_time = time.time()
					jobs2 = [(num, job_server2.submit(entrenarmodelo,(Xnuevo2[num],Ynuevo2[num],2),(ejec_modelo,),("sklearn.tree","sklearn.discriminant_analysis","sklearn.naive_bayes",))) for num in range(numerocpus)]	
					
					modelos_ens2=[]
				
					for numero2,trab2 in jobs2:
						modelos_ens2.append(trab2())
						while not(trab2.finished):
							i=1
					#job_server2.print_stats()
					#job_server2.destroy()
					#Medicion Tiempo Final
					finaltime2=(time.time() - start_time)

					#Train Decision Tree########################################################################################
					start_time = time.time()
					jobs2 = [(num, job_server2.submit(entrenarmodelo,(Xnuevo3[num],Ynuevo3[num],3),(ejec_modelo,),("sklearn.tree","sklearn.discriminant_analysis","sklearn.naive_bayes",))) for num in range(numerocpus)]	
					
					modelos_ens3=[]
				
					for numero2,trab2 in jobs2:
						modelos_ens3.append(trab2())
						while not(trab2.finished):
							i=1
					#job_server2.print_stats()
					#job_server2.destroy()
					#Medicion Tiempo Final
					finaltime3=(time.time() - start_time)
					



					#####Naive Bayes############################################################################################
					Y_ens=zeros(len(Y_final))
					sumaerrorpond=0
					pond_bag=[]
					itera=0
					variableiter=ones(len(modelos_ens1))
					for model in modelos_ens1:	
						salida=model.predict(X_final)
						errorpond=sum(salida==Y_final)/float(len(Y_final))
						sumaerrorpond+=errorpond
						Difebag=(salida==Y_final)
						Difebag2=numpy.ones(len(Difebag))
						err=0
						for z in range(len(Difebag)):
							if Difebag[z]==False:
								err+=distbag1[z]
								Difebag2[z]=0
						if err==0:
							err+=0.000000000001	
						if err>0.5:
							#print 'Error modelo mayor ',itera
							variableiter[itera]=-1
						alp = 0.5 * log((1-err)/err)
						Y_ens+=variableiter[itera]*salida*(1-errorpond)
						pond_bag.append(errorpond)
						itera+=1
					Y_out_ens=numpy.sign(array(Y_ens))
					Difebag=(Y_out_ens==Y_final)
					invertir=1
					e=0
					for z in range(len(Difebag)):	
						if Difebag[z]==False:
							e+=distbag1[z]
					if e==0:
						e+=0.000000000001
					if e>0.5:
						#print "Error mayor!!!!!! "
						invertir=1
					alphabag = 0.5 * log((1-e)/e)
					equiv=[]
					Y_out_ens= Y_out_ens*invertir
					w = zeros(len(Y_final))
					for i in range(len(Y_final)): 
						if Y_final[i]!=Y_out_ens[i]:
							equiv.append(i)
							
						w[i] = distbag1[i] * exp(Y_final[i]*Y_out_ens[i]*-alphabag)

					distbag1 = w / w.sum()
					
					
					#TRAIN
					Prop_hipo_simple_train=Y_out_ens
					Prop_hipo_final_train.append(Y_out_ens*alphabag)	
					#print hipo_final_train
					error_simple_train=sum(Prop_hipo_simple_train==Y_final)/float(len(Y_final))
					Prop_error_final_train=sum(numpy.sign(array(Prop_hipo_final_train).sum(axis=0))==Y_final)/float(len(Y_final))
					resultados1.append(iteracion)
					#print ' Prop Train Hipo: ',error_simple_train,'Prop Train Final: ',Prop_error_final_train, ' Ronda: ',iteracion	
					
					medidas_prop_train=performance(numpy.sign(array(Prop_hipo_final_train).sum(axis=0)),Y_final)
					for med in medidas_prop_train:
						resultados1.append(med)
					del medidas_prop_train

					Y_ens_test=zeros(len(Y_test))
					variable=0
					itera=0
					for model in modelos_ens1:
						#print Xnuevo[mod]	
						salida=model.predict(X_test)*variableiter[itera]		
						ponderacion=float(pond_bag[variable])	
						Y_ens_test+=salida*ponderacion*invertir
						#Y_ens_test+=salida*invertir
						itera+=1
						variable+=1
					###### FIN WHILE

					
					Y_out_ens_test=numpy.sign(array(Y_ens_test))
					
					#TEST
					Prop_hipo_simple_test=Y_out_ens_test
					Prop_hipo_final_test.append(Y_out_ens_test*alphabag)	
					error_simple_test=sum(Prop_hipo_simple_test==Y_test)/float(len(Y_test))
					Prop_error_final_test=sum(numpy.sign(array(Prop_hipo_final_test).sum(axis=0))==Y_test)/float(len(Y_test))
					
					#print ' Prop Test Hipo: ',error_simple_test,'Prop Test Final: ',Prop_error_final_test, ' Ronda: ',iteracion	
					medidas_prop_test=performance(numpy.sign(array(Prop_hipo_final_test).sum(axis=0)),Y_test)
					for med in medidas_prop_test:
						resultados1.append(med)
					del medidas_prop_test
					resultados1.append(finaltime1)
		
					resultados1=map(str,resultados1)
					archivo1.write('&'.join(resultados1)+'\n')
					########################################NAIVE BAYES###############################################################

					#####DECISION TREE################################################################################################
					Y_ens=zeros(len(Y_final))
					sumaerrorpond=0
					pond_bag=[]
					itera=0
					variableiter=ones(len(modelos_ens2))
					for model in modelos_ens2:	
						salida=model.predict(X_final)
						errorpond=sum(salida==Y_final)/float(len(Y_final))
						sumaerrorpond+=errorpond
						Difebag=(salida==Y_final)
						Difebag2=numpy.ones(len(Difebag))
						err=0
						for z in range(len(Difebag)):
							if Difebag[z]==False:
								err+=distbag2[z]
								Difebag2[z]=0
						if err==0:
							err+=0.000000000001	
						if err>0.5:
							#print 'Error modelo mayor ',itera
							variableiter[itera]=1
						alp = 0.5 * log((1-err)/err)
						Y_ens+=variableiter[itera]*salida*errorpond
						pond_bag.append(errorpond)
						itera+=1
					Y_out_ens=numpy.sign(array(Y_ens))
					Difebag=(Y_out_ens==Y_final)
					invertir=1
					e=0
					for z in range(len(Difebag)):	
						if Difebag[z]==False:
							e+=distbag2[z]
					if e==0:
						e+=0.000000000001
					if e>0.5:
						#print "Error mayor!!!!!! "
						invertir=1
					alphabag = 0.5 * log((1-e)/e)
					equiv=[]
					Y_out_ens= Y_out_ens*invertir
					w = zeros(len(Y_final))
					for i in range(len(Y_final)): 
						if Y_final[i]!=Y_out_ens[i]:
							equiv.append(i)
							
						w[i] = distbag2[i] * exp(Y_final[i]*Y_out_ens[i]*-alphabag)

					distbag2 = w / w.sum()
					
					
					#TRAIN
					Prop_hipo_simple_train=Y_out_ens
					Prop_hipo_final_train.append(Y_out_ens*alphabag)	
					#print hipo_final_train
					error_simple_train=sum(Prop_hipo_simple_train==Y_final)/float(len(Y_final))
					Prop_error_final_train=sum(numpy.sign(array(Prop_hipo_final_train).sum(axis=0))==Y_final)/float(len(Y_final))
					resultados2.append(iteracion)
					#print ' Prop Train Hipo: ',error_simple_train,'Prop Train Final: ',Prop_error_final_train, ' Ronda: ',iteracion	
					
					medidas_prop_train=performance(numpy.sign(array(Prop_hipo_final_train).sum(axis=0)),Y_final)
					for med in medidas_prop_train:
						resultados2.append(med)
					del medidas_prop_train

					Y_ens_test=zeros(len(Y_test))
					variable=0
					itera=0
					for model in modelos_ens2:
						#print Xnuevo[mod]	
						salida=model.predict(X_test)*variableiter[itera]		
						ponderacion=float(pond_bag[variable])	
						Y_ens_test+=salida*ponderacion*invertir
						#Y_ens_test+=salida*invertir
						itera+=1
						variable+=1
					###### FIN WHILE

					
					Y_out_ens_test=numpy.sign(array(Y_ens_test))
					
					#TEST
					Prop_hipo_simple_test=Y_out_ens_test
					Prop_hipo_final_test.append(Y_out_ens_test*alphabag)	
					error_simple_test=sum(Prop_hipo_simple_test==Y_test)/float(len(Y_test))
					Prop_error_final_test=sum(numpy.sign(array(Prop_hipo_final_test).sum(axis=0))==Y_test)/float(len(Y_test))
					
					#print ' Prop Test Hipo: ',error_simple_test,'Prop Test Final: ',Prop_error_final_test, ' Ronda: ',iteracion	
					medidas_prop_test=performance(numpy.sign(array(Prop_hipo_final_test).sum(axis=0)),Y_test)
					for med in medidas_prop_test:
						resultados2.append(med)
					del medidas_prop_test
					resultados2.append(finaltime2)
		
					resultados2=map(str,resultados2)
					archivo2.write('&'.join(resultados2)+'\n')
					########################################DECISION TREE###############################################################

					#####QDA################################################################################################
					Y_ens=zeros(len(Y_final))
					sumaerrorpond=0
					pond_bag=[]
					itera=0
					variableiter=ones(len(modelos_ens3))
					for model in modelos_ens3:	
						salida=model.predict(X_final)
						errorpond=sum(salida==Y_final)/float(len(Y_final))
						sumaerrorpond+=errorpond
						Difebag=(salida==Y_final)
						Difebag2=numpy.ones(len(Difebag))
						err=0
						for z in range(len(Difebag)):
							if Difebag[z]==False:
								err+=distbag3[z]
								Difebag2[z]=0
						if err==0:
							err+=0.000000000001	
						if err>0.5:
							#print 'Error modelo mayor ',itera
							variableiter[itera]=1
						alp = 0.5 * log((1-err)/err)
						Y_ens+=variableiter[itera]*salida*errorpond
						pond_bag.append(errorpond)
						itera+=1
					Y_out_ens=numpy.sign(array(Y_ens))
					Difebag=(Y_out_ens==Y_final)
					invertir=1
					e=0
					for z in range(len(Difebag)):	
						if Difebag[z]==False:
							e+=distbag3[z]
					if e==0:
						e+=0.000000000001
					if e>0.5:
						#print "Error mayor!!!!!! "
						invertir=1
					alphabag = 0.5 * log((1-e)/e)
					equiv=[]
					Y_out_ens= Y_out_ens*invertir
					w = zeros(len(Y_final))
					for i in range(len(Y_final)): 
						if Y_final[i]!=Y_out_ens[i]:
							equiv.append(i)
							
						w[i] = distbag3[i] * exp(Y_final[i]*Y_out_ens[i]*-alphabag)

					distbag3 = w / w.sum()
					
					
					#TRAIN
					Prop_hipo_simple_train=Y_out_ens
					Prop_hipo_final_train.append(Y_out_ens*alphabag)	
					#print hipo_final_train
					error_simple_train=sum(Prop_hipo_simple_train==Y_final)/float(len(Y_final))
					Prop_error_final_train=sum(numpy.sign(array(Prop_hipo_final_train).sum(axis=0))==Y_final)/float(len(Y_final))
					resultados3.append(iteracion)
					#print ' Prop Train Hipo: ',error_simple_train,'Prop Train Final: ',Prop_error_final_train, ' Ronda: ',iteracion	
					
					medidas_prop_train=performance(numpy.sign(array(Prop_hipo_final_train).sum(axis=0)),Y_final)
					for med in medidas_prop_train:
						resultados2.append(med)
					del medidas_prop_train

					Y_ens_test=zeros(len(Y_test))
					variable=0
					itera=0
					for model in modelos_ens3:
						#print Xnuevo[mod]	
						salida=model.predict(X_test)*variableiter[itera]		
						ponderacion=float(pond_bag[variable])	
						Y_ens_test+=salida*ponderacion*invertir
						#Y_ens_test+=salida*invertir
						itera+=1
						variable+=1
					###### FIN WHILE

					
					Y_out_ens_test=numpy.sign(array(Y_ens_test))
					
					#TEST
					Prop_hipo_simple_test=Y_out_ens_test
					Prop_hipo_final_test.append(Y_out_ens_test*alphabag)	
					error_simple_test=sum(Prop_hipo_simple_test==Y_test)/float(len(Y_test))
					Prop_error_final_test=sum(numpy.sign(array(Prop_hipo_final_test).sum(axis=0))==Y_test)/float(len(Y_test))
					
					#print ' Prop Test Hipo: ',error_simple_test,'Prop Test Final: ',Prop_error_final_test, ' Ronda: ',iteracion	
					medidas_prop_test=performance(numpy.sign(array(Prop_hipo_final_test).sum(axis=0)),Y_test)
					for med in medidas_prop_test:
						resultados3.append(med)
					del medidas_prop_test
					resultados3.append(finaltime3)
		
					resultados3=map(str,resultados3)
					archivo3.write('&'.join(resultados3)+'\n')
					########################################DECISION TREE###############################################################

					archivo1.close()
					archivo2.close()
					archivo3.close()
					del resultados1
					del datosnuev1
					del resultados2
					del datosnuev2
					del resultados3
					del datosnuev3
					
					 	
					job_server1.destroy()
					job_server2.destroy()
					iteracion+=1
		
			del permutacion
			del datos
			del entre	
			del test
			del hipotesis
			del hipo_final_train
			del hipo_final_test
			del Prop_hipo_final_train
			del Prop_hipo_final_test
	


if __name__=="__main__":
	main(argv)	
