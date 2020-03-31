#Random sector sampling when applied in conjunction with data preprocessing and data visualization.
#A tool for inferencing hidden data patterns and intelligence from suspected storage drive.

#Include Packages
import random, subprocess #,itertools
import math
import sys, errno
import re
import os
import sys
import pandas
import scipy
import csv
from sets import Set
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
#from time import time
import random
import string
import argparse
import commands
import statistics
import hashlib
import itertools
from itertools import groupby
from itertools import cycle
from collections import Counter
from sklearn import metrics
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
###### K-MEANS ######
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
##### DBSCAN #######
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

################################## Output Result Parameter ######################################
AlgoName='ML_DF'
Dataset=AlgoName+'-Dataset.csv'
ResultFileName=AlgoName+'-FinalResult.csv'

StorageID 		= ''			# "/dev/sdb1".
Num_of_samples		= None
ETime			= None
Sector_Samples 		= []			# Store the sector. information
Create_Dataset		= []			# Create Dataset
Dataset			= None
Sector_Size		= 512			# 512 Byte sector size.
Total_Sectors		= None			# Total number of available sectors.
Start			= 0			# Drive start sector
labels = ['Sector','Entropy','ASCIIScore','D_Density','H_Distance','Unique','Common','Type','Class','Category']

numpy.set_printoptions(threshold=sys.maxsize)

class Individual(Exception): 			#Stores the Chromosome and its Fitness collectively
    def __init__(self,Sector_Number,Entropy,ASCIIScore,DataDensity,HammingDistance,UniqueB,CommonB,Type,ClassN,Category):
        self.Sector_Number   	= Sector_Number
	self.Entropy 		= Entropy
	self.ASCIIScore		= ASCIIScore
        self.DataDensity      	= DataDensity
	self.HammingDisance	= HammingDistance
        self.UniqueB      	= UniqueB
        self.CommonB      	= CommonB
        self.Type	      	= Type
        self.ClassN      	= ClassN
        self.Category      	= Category

def KM(Dataset):
	nclu=3     # Number of Clusters
	sns.set()  # for plot styling
	AS_Ent=[[Dataset.ASCIIScore[i],Dataset.Entropy[i]] for i in range(len(Dataset.Sector))]
	Sec_AS=[[Dataset.Sector[i],Dataset.ASCIIScore[i]] for i in range(len(Dataset.Sector))]
	Sec_Ent=[[Dataset.Sector[i],Dataset.Entropy[i]] for i in range(len(Dataset.Sector))]
	X1=np.array(AS_Ent)
	X2=np.array(Sec_AS)
	X3=np.array(Sec_Ent)
	plt.scatter(X1[:, 0], X1[:, 1], s=50);
	plt.show()
	kmeans = KMeans(n_clusters=nclu)
	kmeans.fit(X1)
	y_kmeans = kmeans.predict(X1)
	plt.scatter(X1[:, 0], X1[:, 1], c=y_kmeans, s=50, cmap='viridis')
	np.savetxt('Cluster_KMeans.csv', X1, delimiter=',', fmt=['%f' , '%f'], header='AS,EN', comments='')
	centers = kmeans.cluster_centers_
	plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5);
	np.savetxt('Cluster_KMeans_Centroid.csv', centers, delimiter=',', fmt=['%f' , '%f'], header='AS,EN', comments='')
	plt.show()
	###############################
	k_means_labels = kmeans.labels_
	# print(X1)
	################## File Writing ###########

	with open("K_Means.csv", mode="w") as file:
		f = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
		f.writerow(["Value-1", "Value-2", "Cluster"])
		for k in range(nclu):
			my_members = k_means_labels == k
			if k == 0:
				plt.scatter(X3[my_members, 0], X3[my_members, 1], marker="o", c='green', s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i, j, "0"])
			elif k == 1:
				# c2=X2[my_members].tolist()
				plt.scatter(X3[my_members, 0], X3[my_members, 1], marker="+", c='red', s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i, j, "1"])
			elif k == 2:
				# c2=X2[my_members].tolist()
				plt.scatter(X3[my_members, 0], X3[my_members, 1], marker="D", c='yellow', s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i, j, "2"])
			elif k == 3:
				# c2=X2[my_members].tolist()
				# print "Cluster-2",c2
				plt.scatter(X3[my_members, 0], X3[my_members, 1], marker="*", c='cyan', s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i, j, "3"])
			else:
			        for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i, j, "122"])


	#plt.legend(("Null sectors","Encrypted information","Plain text","Compressed information"),scatterpoints=1,fontsize=12)
	plt.xlabel('Drive Sectors',fontsize=15)
	plt.ylabel('Entropy Values',fontsize=15)
	# Limits for the Y axis
	plt.ylim(-0.2,8)
	plt.xlim(0)
	plt.grid(True)
	# Create names
	plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=20)
	plt.savefig("SecvsEnt.eps", dpi=300, bbox_inches='tight')
	plt.show()

	with open("K_Means_1.csv", mode="w") as file:
		f = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
		f.writerow(["Value-1", "Value-2", "Cluster"])
		for k in range(nclu):
			my_members = k_means_labels== k
			if k == 0:
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="o", c='green', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "0"])
			elif k == 1:
				# c2=X2[my_members].tolist()
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="+", c='red', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "1"])
			elif k == 2:
				# c2=X2[my_members].tolist()
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="D", c='yellow', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "2"])
			elif k == 3:
				# c2=X2[my_members].tolist()
				# print "Cluster-2",c2
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="*", c='cyan', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "3"])
			else:
			        for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "122"])

	#plt.legend(("Null sectors","Encrypted/Compressed information","Plain text"),scatterpoints=1,fontsize=12)
	plt.xlabel('Drive Sectors',fontsize=15)
	plt.ylabel('ASCII Score',fontsize=15)
	# Limits for the Y axis
	plt.ylim(-0.1,1.2)
	plt.xlim(0)
	plt.grid(True)
	plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=17)
	plt.savefig('SecvsAS.eps', dpi=300, bbox_inches='tight')
	plt.show()
	#f.close()
	actuals=[[Dataset.Category[i]] for i in range(len(Dataset.Sector))]
	matrix = confusion_matrix(y_kmeans,np.array(actuals))
	report = classification_report(y_kmeans, np.array(actuals))
	print "K-MEANS Report: \n",matrix,'\n',report


def DBS(Dataset):
	sns.set()  # for plot styling
	AS_Ent=[[Dataset.ASCIIScore[i],Dataset.Entropy[i]] for i in range(len(Dataset.Sector))]
	Sec_AS=[[Dataset.Sector[i],Dataset.ASCIIScore[i]] for i in range(len(Dataset.Sector))]
	Sec_Ent=[[Dataset.Sector[i],Dataset.Entropy[i]] for i in range(len(Dataset.Sector))]
	X1=np.array(AS_Ent)
	X2=np.array(Sec_AS)
	X3=np.array(Sec_Ent)
	plt.scatter(X1[:, 0], X1[:, 1], s=50);
	plt.show()
	X1 = StandardScaler().fit_transform(X1)
	db = DBSCAN(eps=0.5, min_samples=10).fit(X1)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	print('Estimated number of clusters: %d' % n_clusters_)
	unique_labels = set(labels)
	cl_1_a = []
	cl_1_b = []
	cl_a = []
	cl_b = []
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
        	if k == -1:
            		col = 'k'
        	class_member_mask = (labels == k)
        	xy = X1[class_member_mask & core_samples_mask]
		
        	plt.plot(xy[:, 0], xy[:, 1], 'o',markerfacecolor=col,markeredgecolor='k', markersize=14)
        	cl_a.append(xy[:,0])
		cl_b.append(xy[:,1])
		#np.savetxt('Cluster_DBSCAN1.csv', xy, delimiter=',', fmt=['%f' , '%f'], header='AS,EN', comments='')
        	
		xy = X1[class_member_mask & ~core_samples_mask]
        	cl_1_a.append(xy[:,0])
		cl_1_b.append(xy[:,1])
		plt.plot(xy[:, 0], xy[:, 1], 'o',markerfacecolor=col,markeredgecolor='k', markersize=6)
        	#np.savetxt('Cluster_DBSCAN2.csv', xy, delimiter=',', fmt=['%f' , '%f'], header='AS,EN', comments='')
	

	
	with open('Cluster_DBSCAN1.csv', 'w') as f:
    		writer = csv.writer(f)
		writer.writerows(zip(["AS"], ["EN"]))
    		writer.writerows(zip(cl_a[0], cl_b[0]))
	
	
	
	with open('Cluster_DBSCAN2.csv', 'w') as f:
    		writer = csv.writer(f)
		writer.writerows(zip(["AS"], ["EN"]))
    		writer.writerows(zip(cl_1_a[0], cl_1_b[0]))


	plt.title('Estimated number of clusters wrt '+' : %d' % n_clusters_)
	plt.show()
	###############################
	dbscan_labels = labels
	################## File Writing ###########

	with open("DBSCAN.csv", mode = "w") as file:
		f = csv.writer(file, delimiter= ",", quotechar='"', quoting = csv.QUOTE_MINIMAL)
		f.writerow( ["Value-1","Value-2","Cluster"] )
		for k in range(n_clusters_):
    			my_members = dbscan_labels == k
			if k==0:
				plt.scatter(X3[my_members, 0], X3[my_members, 1],marker="o", c='green',s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i,j,"0"])
			elif k==1:
    				#c2=X2[my_members].tolist()
				plt.scatter(X3[my_members, 0], X3[my_members, 1],marker="+", c='red',s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i,j,"1"])
			elif k==2:
    				#c2=X2[my_members].tolist()
				plt.scatter(X3[my_members, 0], X3[my_members, 1],marker="D", c='yellow',s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i,j,"2"])
			elif k==3:
    				#c2=X2[my_members].tolist()
				#print "Cluster-2",c2
				plt.scatter(X3[my_members, 0], X3[my_members, 1],marker="*", c='cyan',s=100);
				for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i,j,"3"])
			else:
			        for i, j in zip(X3[my_members, 0], X3[my_members, 1]):
					f.writerow([i, j, "122"])

	#plt.title('Significant region identification using K-means clustering approach')
	#plt.legend(("Null sectors","Encrypted information","Plain text","Compressed information"),scatterpoints=1,fontsize=12)
	plt.xlabel('Drive Sectors',fontsize=15)
	plt.ylabel('Entropy Values',fontsize=15)
	# Limits for the Y axis
	plt.ylim(-0.2,8)
	plt.xlim(0)
	plt.grid(True)
	# Create names
	plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=20)
	plt.savefig("SecvsEnt.eps", dpi=300, bbox_inches='tight')
	plt.show()
	#f.close()
	################## File Writing ###########

	with open("DBSCAN_1.csv", mode="w") as file:
		f = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
		f.writerow(["Value-1", "Value-2", "Cluster"])
		for k in range(n_clusters_):
			my_members = dbscan_labels== k
			if k == 0:
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="o", c='green', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "0"])
			elif k == 1:
				# c2=X2[my_members].tolist()
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="+", c='red', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "1"])
			elif k == 2:
				# c2=X2[my_members].tolist()
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="D", c='yellow', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "2"])
			elif k == 3:
				# c2=X2[my_members].tolist()
				# print "Cluster-2",c2
				plt.scatter(X2[my_members, 0], X2[my_members, 1], marker="*", c='cyan', s=100);
				for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "3"])

			else:
			        for i, j in zip(X2[my_members, 0], X2[my_members, 1]):
					f.writerow([i, j, "122"])


	#plt.legend(("Null sectors","Compressed information","Plain text","Encrypted information"),scatterpoints=1,fontsize=12)
	plt.xlabel('Drive Sectors',fontsize=15)
	plt.ylabel('ASCII Score',fontsize=15)
	# Limits for the Y axis
	plt.ylim(-0.1,1.2)
	plt.xlim(0)
	plt.grid(True)
	plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=17)
	plt.savefig('SecvsAS.eps', dpi=300, bbox_inches='tight')
	plt.show()
	#f.close()
	actuals=[[Dataset.Category[i]] for i in range(len(Dataset.Sector))]
	matrix1 = accuracy_score(labels,np.array(actuals))
	matrix2 = accuracy_score(labels,np.array(actuals), normalize=False)
	#report = classification_report(y_kmeans, np.array(actuals))
	print matrix1,'\n',matrix2

def encode(input_string):
    return [(len(list(g)), k) for k,g in groupby(input_string)]

def decode(lst):
    return ''.join(c * n for n,c in lst)

def ASCIISCORE(data):
	byteArr = map(ord, data)
	fileSize = len(byteArr)
	ctr = 0
# calculate the frequency of each byte value in the file
	if sum(byteArr)>0:
		for byte in byteArr:
        		if byte >= 0 and byte<128:
            			ctr += 1
    		return (float(ctr) / fileSize)
	else:
		return float(0)


def entropy(data):
	byteArr = map(ord, data)
#	print byteArr
	fileSize = len(byteArr)
# calculate the frequency of each byte value in the file
	freqList = []
	for b in range(256):
    		ctr = 0
    		for byte in byteArr:
        		if byte == b:
            			ctr += 1
    		freqList.append(float(ctr) / fileSize)
# Shannon entropy
	ent = 0.0
	for freq in freqList:
    		if freq > 0:
        		ent = ent + freq * math.log(freq, 2)
	ent = -ent
	if ent!=0:
    		return ent
	else:
    		return 0

a=1

def datadensity(data):
	den=0
	if data==('00'*Sector_Size):
		dataden = 0
	elif not (data==('00'*Sector_Size)):
		data1=data
		data2=data1.split('00')
		for i in range(len(data2)):
			den=den+len(data2[i])/2
		dataden=float(den)/float(Sector_Size)
	return dataden

def HamDistance(data):
	den=0
	if data==('00'*Sector_Size):
		dataden = 0
	elif not (data==('00'*Sector_Size)):
		data1=data
		data2=data1.split('00')
		for i in range(len(data2)):
			den=den+len(data2[i])/2
	return den

def decideclass(data):
	if data==('00'*Sector_Size):
		dclass='Null'
	elif not(data==('00'*Sector_Size)):
		dclass='Non-Null'
	return dclass

def uniquebytes(data):
    si = iter(data)
    stList=map(''.join, itertools.izip(si, si))
    alphabet = list(Set(stList))
    return float(len(alphabet)/float(Sector_Size))

def main():
	global Sector_Samples
	global Create_Dataset
	drive = file(StorageID,'rb')
	RanGen(drive)
	drive.close()
	Datasetcreation(Create_Dataset)
	KM(Dataset)		# K-Mean Algorithm
	DBS(Dataset)		# DBSCAN Algorithm
###################### Random Sample Generation #####################
def RanGen(drive):
    global Sector_Samples
    global Create_Dataset
    Sector_Samples=[]
    starttime=time.time()
    count=0#Num_of_samples
    for i in range(0,Total_Sectors):# Zero to Total_sectors
	sectornumber = random.randint(0,Total_Sectors)
	SectorData = ExtractSector(drive,sectornumber)
	count=count+1
	########## FEATURE SELECTION #################
	Entropy	= entropy(SectorData)
	ASCIIScore = ASCIISCORE(SectorData)
	d1 = SectorData.encode('hex')
	DataDensity = datadensity(d1)
	HammingDisance = HamDistance(d1)
	UniqueB = uniquebytes(d1)
	CommonB = (1-UniqueB)
	ClassN  = decideclass(d1)
	Category=0
	if ClassN=='Non-Null':
		if (Entropy > 0  and Entropy <= 4.8) and (ASCIIScore >= 0.6  and ASCIIScore <= 1.0):
			Type='Plain text'
			Category=1
		elif (Entropy >4.8  and Entropy <= 8.0) and (ASCIIScore > 0  and ASCIIScore < 0.6):
			Type='Encrypted/Compressed'
			Category=2
		#elif (Entropy > 7.0  and Entropy <= 8.0) and (ASCIIScore > 0  and ASCIIScore < 0.6):
		#	Type='Encrypted'
		#	Category=3
	else:
		Type='Null'
	Create_Dataset.append([sectornumber,Entropy,ASCIIScore,DataDensity,HammingDisance,UniqueB,CommonB,Type,ClassN,Category])
        Sector_Samples.append(Individual(sectornumber,Entropy,ASCIIScore,DataDensity,HammingDisance,UniqueB,CommonB,Type,ClassN,Category))
	endtime=time.time()
	if (endtime-starttime)>=ETime or count>=Num_of_samples or count>=Total_Sectors:
		print endtime-starttime,' sec. time has been elapsed. \n','Required samples have been collected.\n Now awaiting analysis to be done.'
		break;

def ExtractSector(drive,sectornumber):
    data = None
    if long(sectornumber)<long(Total_Sectors):
	drive.seek(Start)
	sectr=long(sectornumber)*long(Sector_Size)
        drive.seek(sectr)
        data=drive.read(Sector_Size)
	#H = hsh(data)
#	if H in	MD5HashOfTargetFile and H != 'bf619eac0cdf3f68d496ea9344137e8b':# fitnessvalue != 0 or H!='bf619eac0cdf3f68d496ea9344137e8b':
        return data

def Datasetcreation(data_set):
	global Dataset
	#['Sector','Entropy','ASCIIScore','D_Density','H_Distance','Unique','Common','Type','Class']
	df=pd.DataFrame.from_records(data_set,columns=labels)
	Dataset = df
	print type(df),df.head(),df.info(),df.describe()
	#for i in range(len(df.ASCIIScore)):
	#	if df.ASCIIScore[i]!=0 and df.Entropy[i]!=0:
	#		print df.Entropy[i], df.ASCIIScore[i]
	#sns.boxplot(x="Class", y="H_Distance",  data=df)
	#plt.show()
	#sns.swarmplot(x="Class", y="H_Distance",  data=df)
	#plt.show()
	#sns.swarmplot(x="Sector", y="ASCIIScore",  data=df)
	#plt.show()
	#sns.swarmplot(x="Class", y="Entropy",hue = "Type",  data=df)
	#plt.show()
	#sns.swarmplot(x="Class", y="Entropy", data=df)
	#plt.show()
	#sns.swarmplot(x="Class", y="Sector",hue = "Type",  data=df)
	#plt.show()
	#sns.swarmplot(x="Sector", y="Category",  data=df)
	#plt.show()
	#sns.swarmplot(x="Sector", y="Class",  data=df)
	#plt.show()
	#sns.swarmplot(x="Type", y="Sector", hue='Class',  data=df)
	#plt.show()
	#sns.swarmplot(x="Unique", y="Entropy",hue = "Class",  data=df)
	#plt.show()
	#g = sns.FacetGrid(df,col='Class',row='Type')
	#plt.show()
	#g.map(sns.regplot,'Entropy',"Sector")
	#plt.show()


if __name__ == '__main__':
	################################# Parsing the arguments ###############################################
	parser = argparse.ArgumentParser(description='Preprocessing and Analytics of Suspected Storage Media.')
	parser.add_argument('-d', '--drive', required=True,help='The device ID of the drive. Eg. /dev/sda1')
	parser.add_argument('-o', '--outputDir', required=True,help='The directory to which DE results should be saved')
	parser.add_argument('-S', '--sectors', required=True, type=int, help='The number of total sectors present in suspected media.')
	parser.add_argument('-T', '--etime', required=True, type=int, help='Time (min.) under which examination should be completed necessarily.')
	parser.add_argument('-N', '--samples', required=False, type=int, default=10000, help='The size of tye random samples. Default value is 10,000.')
	args = parser.parse_args()
	if not os.access(args.drive, os.W_OK):
		sys.exit("Unable to locate the storage drive %s" % (args.drive,) )
	if not os.path.isdir(args.outputDir) or not os.access(args.outputDir, os.W_OK):
		sys.exit("Unable to write to output directory %s" % (args.outputDir,) )
	if args.samples < 1:
		sys.exit('Number of samples must be greater than 1000.')
	if args.etime < 1:
		sys.exit('The value of time (in min.) must be greater than 0.')
	if not args.samples:
		sys.exit('Please provide valid quantity of available sectors. Please check the device ID.')
        ############################ PARSING END #######################################

        ################################# Setting-Up Arguments  ###############################################
        StorageID 	= args.drive
	AlgoName 	= args.outputDir+AlgoName+'-'
	Num_of_samples	= args.samples
	ETime		= args.etime
	Total_Sectors	= args.sectors
        ResultFileName 	= args.outputDir+ResultFileName#os.path.join(args.outputDir,ResultFileName)
	main()
