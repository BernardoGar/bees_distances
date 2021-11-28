#import matplotlib.pyplot as plt
import numpy as np
import utm
import pandas as pd
import math
#from matplotlib.pyplot import figure
#from matplotlib import cm as CM
#from matplotlib import mlab as ML
import pymysql 
import time
import random
import requests
import similaritymeasures
try: 
	from mpl_toolkits.basemap import Basemap
	basemap_exists=True
except: 
	print("no basemap")
	basemap_exists=False
	

from scipy.stats.mstats import gmean


def get_task(ip_address, force=[]):
	connection = get_connection()

	with connection.cursor() as cursor:
		sql = "LOCK TABLE tasks_abejas WRITE;" 
		cursor.execute(sql)
	connection.commit()
	slows=["montecarlobees"]
	alts=["0"]+force
	
	with connection.cursor() as cursor:
	
		sql = "select * from tasks_abejas where (task_completed=0 and task_responsible=\"\") or (task_id in ("+", ".join(alts)+")) order by RAND() limit 1000;" 
		cursor.execute(sql)
		result = cursor.fetchall()
		if result:
			flag=0
			indx=0
			nl=[]
			for r in result:
				if r["task_name"] in slows:
					flag=1
					if len(nl)<3:
						nl.append(r)
			if flag==1: result=nl
			
			tids=[str(res["task_id"]) for res in result]
			sql = "update tasks_abejas set task_responsible=\""+str(ip_address)+"\" where task_id in ("+", ".join(tids)+")"
			cursor.execute(sql)
				
				
		else:
			sql = "select * from tasks_abejas where task_completed=0 order by RAND() limit 1;" 
			cursor.execute(sql)
			result = cursor.fetchone()
			result=[result]
			if not result:
				result=False
	connection.commit()
	
	with connection.cursor() as cursor:
		sql = "UNLOCK TABLES;" 
		cursor.execute(sql)
		
	connection.commit()
	connection.close()
	return result
	
	
def do_insertions(inserts):
	k={}
	insert_statements=[]
	for ins in inserts:
		indx=ins.find("values ")+len("values ")
		if indx<len("values")+1: 
			insert_statements.append(ins)
			print("No se encontró values")
		else:
			if ins[:indx] in k:
				k[ins[:indx]]+=", "+ins[indx:]
			else:
				k[ins[:indx]]=ins
	for key in k:
		insert_statements.append(k[key])
	conn=get_connection()
	with conn.cursor() as cursor:
		for sql in insert_statements:
			print(sql)
			cursor.execute(sql)
	conn.commit()
	conn.close()
  

def gen_lookups(tasks):
	nd={"gen_distancia_par":[{"db":"abejas_curvas", "condition":("task_id","task_id1")},
	{"db":"abejas_curvas", "condition":("task_id","task_id2")}],
		"gen_distancia_contra_orig":[{"db":"abejas_curvas", "condition":("task_id","task_id")}]}
	lookups={}
	#not important definitions
	d2=2
	diccionario_distfisica_v_distgenetica=2
	results={}
	for tas in tasks:
		params=eval(tas["task_params"])
		if tas["task_name"] in nd:
			for lookup in nd[tas["task_name"]]:
				condition="("+lookup["condition"][0]+"="+str(params[lookup["condition"][1]])+")"
				if lookup["db"] in lookups:
					lookups[lookup["db"]]+=" or "+condition
				else: 
					lookups[lookup["db"]]="select * from "+lookup["db"]+" where "+condition
	
	conn=get_connection()
	for lookup in lookups:
		with conn.cursor() as cursor:
			sql=lookups[lookup]
			cursor.execute(sql)
			result = cursor.fetchall()
	conn.close()
	for tas in tasks:
		results[tas["task_id"]]={}
		params=eval(tas["task_params"])
		if tas["task_name"] in nd:
			for lookup in nd[tas["task_name"]]:
				condition="("+lookup["condition"][0]+"="+str(params[lookup["condition"][1]])+")"
				results[tas["task_id"]]["select * from "+lookup["db"]+" where "+condition]=[]
				for r in result:
					if r[lookup["condition"][0]]==params[lookup["condition"][1]]:
						results[tas["task_id"]]["select * from "+lookup["db"]+" where "+condition].append(r)
	return results
	
  
def gen_distancia_par(task_id1, task_id2, tid, lookedup_info):
	num_data = np.zeros((50, 2))
	exp_data = np.zeros((50, 2))
	num_data[:, 0] = list(range(50))
	exp_data[:, 0] = list(range(50))
	
	if task_id2<task_id1:
		a=task_id2
		task_id2=task_id1
		task_id1=a
		
	
	sql1="select * from abejas_curvas where (task_id="+str(task_id1)+")"
	sql2="select * from abejas_curvas where (task_id="+str(task_id2)+")"
	result=[]
	result+=lookedup_info[sql1]
	result+=lookedup_info[sql2]
	nums=[]
	for r1 in result:
		for r2 in result:
			if r1["task_id"]<r2["task_id"] and r1["k"]==r2["k"]: 
				num_data[:, 1] = [r1["Int_"+str(u)] for u in range(50)]
				exp_data[:, 1] = [r2["Int_"+str(u)] for u in range(50)]
				simi=similaritymeasures.area_between_two_curves(exp_data, num_data)
				nums.append(simi)
	if nums==[]: return "Nothing"
	res=gmean(nums)
	sql="insert into distancias_geom (ca, cb, distancia, tid) values ("+str(task_id1)+", "+str(task_id2)+", "+str(res)+", "+str(tid)+")"
	return {"message":res, "insert":sql}
	
	
def searchfam(i, familias):
	if i.find("SYD8")>0: i=i.replace("SYD8", "")
	for k in familias:
		if i in k["integrantes"]: 
			return k
	return {"Familia":"none", "c1":0, "c2":0}

#Function to get bootstrap metrics
def bootit(lista):
	proms=[]
	if len(lista)<1000:
		for s in range(500):
			nl=random.choices(lista, k=len(lista))
			proms.append(np.mean(nl))
		return np.percentile(proms, 5), np.percentile(proms, 95)
	else:
		error_std=np.std(lista)/(len(lista)**0.5)
		return np.mean(lista)-1.96*error_std, np.mean(lista)+1.96*error_std
	

	
	
def distancia_fisica(a, b, information):
	x1=information[a]["utm_xy"][0]
	y1=information[a]["utm_xy"][1]
	x2=information[b]["utm_xy"][0]
	y2=information[b]["utm_xy"][1]
	return euclidean_dist((x1, y1), (x2, y2))

def distancia_np(a, b, information):
	a=information[a]["distance_np"]
	b=information[b]["distance_np"]
	return (a+b)/2.0



#Funciones de verosimilitud de panal
def euclidean_dist(a, b):
	d=0
	for i, j in zip(a, b):
		d+=(i-j)**2
	return d**0.5

def verif_atr(veros, atr):
	v2=veros
	flag=True
	ind=0
	for at in atr:
		if ind<len(atr)-1:
			if at not in v2: 
				flag=False
				v2[at]={}
			v2=v2[at]
			ind+=1
		else:
			if at not in v2: 
				flag=False
				v2[at]=0
	return veros, flag

def gradient(c1, c2, a):
	b=1-a
	tupi1=(tuple(int(c1[1:][i:i+2], 16)/255.0 for i in (0, 2, 4)))
	tupi2=(tuple(int(c2[1:][i:i+2], 16)/255.0 for i in (0, 2, 4)))
	return (tupi1[0]*a+tupi2[0]*b, tupi1[1]*a+tupi2[1]*b, tupi1[2]*a+tupi2[2]*b)

def set_val(veros, atr, res):
	ind=0
	v2=veros
	for at in atr:
		if ind<len(atr)-1:
			v2=v2[at]
		else: 
			v2[at]=res
		ind=ind+1
	return veros

def place(veros, atr):
	v2=veros
	for at in atr:
		v2=v2[at]
	return v2


def get_verosimilitud(p0, p1, s0, s1, v, lam1, lam2):
	p=(p0,p1)
	s=(s0,s1)
	prob=lam1*math.exp(-lam2*euclidean_dist(p,s))
	return v*prob+(1-v)*(1-prob)

def get_veros(lams1, lams2, lam1, lam2, nvisit, visit, xr, yr, fams):
	Z=[]
	index=len(xr)*len(fams)*lams2.index(lam2)+len(xr)*len(fams)*len(lams2)*lams1.index(lam1)
	for ix, iy in zip(xr, yr):
		z=1
		for familia in fams:
			if fams[familia]==1: z=z*visit[index]
			else: z=z*nvisit[index]
			index+=1
		Z.append(z**(1/len(fams)))
	return Z


def distancia_genetica1(a, b, distribution):
	distancia=0
	for alele in distribution:
		if a[alele]==b[alele]: distancia=distancia+(1/distribution[alele][a[alele]])
		
	return distancia

def distancia_genetica2(a, b, distribution):
	distancia=0
	for alele in distribution:
		if a[alele]==b[alele]: distancia+=1/float(len(distribution))
		
	return distancia

def distancia_genetica3(a, b, distribution):
	distancia=0
	tot=0
	tot2=0
	for alele in distribution:
		tot+=(1/distribution[alele][a[alele]])
		tot+=(1/distribution[alele][b[alele]])
		if a[alele]==b[alele]: tot2+=2*(1/distribution[alele][b[alele]])
	return tot2/float(tot)


def distancia_genetica4(a, b, distribution, N_mam=269):
	P_A1_a1=1
	P_A2_a2=1
	P_A2_a2_hermano_a2=1
	for alele in distribution:
		P_A1_a1=P_A1_a1*(distribution[alele][a[alele]])
		P_A2_a2=P_A2_a2*(distribution[alele][b[alele]])
		# La probabilidad abajo es: si son iguales, es 1/2 (si vino de la misma columna alelo) + (1/2)*prob_del_alelo si vino de distinta columna, pero si 
		# son distintos son 1/2 * probabilidad de ese alelo. 
		P_A2_a2_hermano_a2=P_A2_a2_hermano_a2*((1/2)*int(a[alele]==b[alele])+(1/2)*distribution[alele][b[alele]])
	P_A1_A2_distinta_mama=P_A1_a1*P_A2_a2
	P_A1_A2_misma_mama=P_A1_a1*P_A2_a2_hermano_a2
	return (1/(1+(N_mam-1)*P_A1_A2_distinta_mama/P_A1_A2_misma_mama))
	
			
def to_xy(point):
	r=6371000
	phi_0 = 151.16067
	cos_phi_0 = math.cos(math.radians(phi_0))
	lam = point[0]
	phi = point[1]
	return (r * math.radians(lam) * cos_phi_0, r * math.radians(phi))

def get_coords_2017():

	coordenadas2017={ "Vegetable university":(-33.88923, 151.19259), "Marrickville 2":(-33.90468, 151.16067),
			 'Bondi Junction': (-33.88877, 151.24145) , 'University': (-33.88514, 151.18837), 'Surrey Hills':(-33.88972, 151.20965),
			 'Newtown': (-33.894944, 151.187083), 'Anandale': (-33.882977, 151.173809),
				 "Jannali":(-34.0212, 151.06618), 'Caringbah':(-34.06084, 151.11107),
				 'Marrickville':(-33.89828, 151.16525), "Leichhardt":(-33.884889, 151.156028),
				 "Paddington":(-33.881917, 151.180611), "Tanya":(-33.88425, 151.23336)
			}
	return coordenadas2017
	
	
def get_coords_QL():

	coordenadasQL={"Beerwah1":(-26.840181, 152.953008),
			   "Saharafarms":(-26.896434, 152.939769), 
			   "Mololahvalley":(-26.754253, 152.987354)}
	return coordenadasQL

def get_loc_coords_dnp():
	locations={"1 Belmore st. North parramatta": "Parramatta", "1 Oberon Cresc": "Gordon", 
		   "20 Philip Rd Mona Vale": "Palmbeach", "25 Strickland Ave.": "Strickland", 
		   "38 Park Ave Ashfield": "Asheville", "4 Goodlet St. Surry Hills": "Surry Hills", 
		   "430 Mona Vale Rd.": "Stives", "67-71 Eton St Sutherland": "Sutherland", 
		   "7th Fourth Avenue Jannali": "Jannali", "8 Treatts Rd": "Lindfield", 
		   "92 Griffiths St. balgowlah": "Balgowlah ", "beauford Ave Ros parents": "Caringbah", 
		   "Kate´s place Unit 6 2-4 Wood st. Manley": "Manley", "Ros House Woodland street": "Marrickville", 
		   "Strickland Avenue 25": "Strickland", "UNISYD": "Camperdown", " 7 Cary St Leichhardt": "Leichhardt", 
		   "131 Nelson Street, Anandale": "Anandale", "Bondi Junction": "Bondi Junction", 
		   "Francisco (Surrey Hills)": "Surrey Hills", "Jannali": "Jannali", "Newtown Amelie": "Newtown", 
		   "Paddington": "Paddington", "Ros' House": "Marrickville", "Ros's Brothers": "Marrickville 2", 
		   "Ros's Parents": "Caringbah", "Ros´": "Marrickville sp", "Tanya's House": "Tanya", 
		   "University": "University", "Vegetable Garden UNISYD": "Vegetable university", 
		   "Mololahvalley":"Mololahvalley", "Beerwah2":"Beerwah2", 'Beerwah1':'Beerwah1', 'Saharafarms':'Saharafarms',
		  'Beerwah3':'Beerwah3'}

	coordenadas={"Marrickville":(-33.90468, 151.16067), "Surry Hills":(-33.88972, 151.20965),
			 "Camperdown":(-33.88514, 151.18837), "Asheville":(-33.89428, 151.12279),
			 "Balgowlah ":(-33.79207, 151.26724),"Manley":(-33.80352, 151.28617),"Jannali":(-34.01215, 151.07181),
			 "Sutherland":(-34.03502, 151.05768), "Stives": (-33.70606, 151.18046),"Palmbeach":(-33.66858, 151.31378),
			"Lindfield":(-33.77102, 151.16314),"Gordon":(-33.75781, 151.15081),"Caringbah":(-34.06095, 151.11111), 
			 "Parramatta":(-33.800766, 151.009321), "Strickland":(-33.778420, 151.171036)
			}

	distance_national_park={
	"Marrickville":17.5,"Surry Hills":20, "Camperdown":20, "Asheville":17.5,"Balgowlah ":11.3,"Manley":13.5,
			"Jannali":3.3,"Sutherland":5, "Stives": 0.1,"Palmbeach":3.4,"Lindfield":7.3,"Gordon":6.5,"Caringbah":1, 
	"Parramatta":10, "Strickland":7.3}
	
	return locations, coordenadas, distance_national_park
	
	
def get_connection():
	return pymysql.connect(host='ec2-18-222-185-142.us-east-2.compute.amazonaws.com',
							 user='root',
							 password='password',
							 db='dbname',
							 charset='utf8mb4',
							 cursorclass=pymysql.cursors.DictCursor)
							 
							 
def gen_distancias(seed, xs_min, ys_min, r1, r2, mu, sigma, pi2, loc_familia,  list_cor, min_dist_2):
	locs=[]
	distancias=[]
	random.seed(seed+5676)
	loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
	for mem in range(1000):
		vlucht=-1
		vlucht=random.normalvariate(mu, sigma)
		if vlucht>0:
			angle=random.random()*pi2
			loc=(loc_familia[0]+math.cos(angle)*vlucht, loc_familia[1]+math.sin(angle)*vlucht)
			for k in list_cor:
				if abs(loc[0]-k[0])<min_dist_2:
					if abs(loc[1]-k[1])<min_dist_2:
						if euclidean_dist(loc, k)<min_dist_2:
							locs.append(k)
	for i in range(len(locs)-1):
		for j in range(i+1, len(locs)):
			distancia=euclidean_dist(locs[i], locs[j])
			distancias.append(distancia)
	return distancias

def rsel(distribution):
	nd={}
	for k in distribution:
		s=random.random()
		lkeys=list(distribution[k].keys())
		indi=0
		tot=distribution[k][lkeys[indi]]
		while tot<s:
			indi+=1
			tot=tot+distribution[k][lkeys[indi]]
			
			
		nd[k]=lkeys[indi]
	return nd
			




def gen_distancias_lognormal(seed, xs_min, ys_min, r1, r2, mu, sigma, pi2, loc_familia,  list_cor, min_dist_2):
	locs=[]
	distancias=[]
	random.seed(seed+5676)
	water=True
	while water==True:
		loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
		res=utm.to_latlon(1000*loc_familia[0]+medx, 1000*loc_familia[1]+medy, 56, "H")
		url="https://api.onwater.io/api/v1/results/"+str(res[0])+","+str(res[1])+"?access_token=zYQHVfPa7cSXhz6Masy6"
		if url not in dic_proba:
			s={"error":0}
			while "error" in s:
				s=requests.get(url).json()
			dic_proba[url]=s["water"]
		water=dic_proba[url]
	for mem in range(1000):
		vlucht=-1
		vlucht=random.lognormvariate(mu, sigma)
		if vlucht>0:
			angle=random.random()*pi2
			loc=(loc_familia[0]+math.cos(angle)*vlucht, loc_familia[1]+math.sin(angle)*vlucht)
			for k in list_cor:
				if abs(loc[0]-k[0])<min_dist_2:
					if abs(loc[1]-k[1])<min_dist_2:
						if euclidean_dist(loc, k)<min_dist_2:
							locs.append(k)
	for i in range(len(locs)-1):
		for j in range(i+1, len(locs)):
			distancia=euclidean_dist(locs[i], locs[j])
			distancias.append(distancia)
	return distancias


def gen_distancias_beta(seed, xs_min, ys_min, r1, r2, alpha, beta, pi2, loc_familia,  list_cor, min_dist_2):
	locs=[]
	distancias=[]
	random.seed(seed+5676)
	loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
	for mem in range(1000):
		vlucht=-1
		vlucht=random.betavariate(alpha, beta)*50
		if vlucht>0:
			angle=random.random()*pi2
			loc=(loc_familia[0]+math.cos(angle)*vlucht, loc_familia[1]+math.sin(angle)*vlucht)
			for k in list_cor:
				if abs(loc[0]-k[0])<min_dist_2:
					if abs(loc[1]-k[1])<min_dist_2:
						if euclidean_dist(loc, k)<min_dist_2:
							locs.append(k)
	for i in range(len(locs)-1):
		for j in range(i+1, len(locs)):
			distancia=euclidean_dist(locs[i], locs[j])
			distancias.append(distancia)
	return distancias


def gen_distancias_expo(seed, xs_min, ys_min, r1, r2, lam, pi2, loc_familia,  list_cor, min_dist_2):
	locs=[]
	distancias=[]
	random.seed(seed+5676)
	loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
	for mem in range(1000):
		vlucht=-1
		vlucht=random.expovariate(lam)
		if vlucht>0:
			angle=random.random()*pi2
			loc=(loc_familia[0]+math.cos(angle)*vlucht, loc_familia[1]+math.sin(angle)*vlucht)
			for k in list_cor:
				if abs(loc[0]-k[0])<min_dist_2:
					if abs(loc[1]-k[1])<min_dist_2:
						if euclidean_dist(loc, k)<min_dist_2:
							locs.append(k)
	for i in range(len(locs)-1):
		for j in range(i+1, len(locs)):
			distancia=euclidean_dist(locs[i], locs[j])
			distancias.append(distancia)
	return distancias

from contextlib import contextmanager
from multiprocessing import Pool
from functools import partial

@contextmanager
def terminating(thing):
	try:
		yield thing
	finally:
		thing.terminate()
		
		
def gen_distancia_contra_orig(task_id, diccionario_distfisica_v_distgenetica, tid, lookedup_info):
	num_data = np.zeros((50, 2))
	exp_data = np.zeros((50, 2))
	num_data[:, 0] = list(range(50))
	exp_data[:, 0] = list(range(50))


	sql="select * from abejas_curvas where (task_id="+str(task_id)+")"
	result=lookedup_info[sql]
	print("len(res)="+str(len(result)))
	nums=[]
	for res in result:
		num_data[:, 1] = [res["Int_"+str(u)] for u in range(50)]
		exp_data[:, 1] = diccionario_distfisica_v_distgenetica[res["k"]][1]
		simi=similaritymeasures.area_between_two_curves(exp_data, num_data)
		nums.append(simi)
	if nums==[]: return {"message":"nothing"}
	res=gmean(nums)
	sql="insert into distancias_geom (ca, cb, distancia, tid) values (0, "+str(task_id)+", "+str(res)+", "+str(tid)+")"
	return {"message":res, "insert":sql}
	
def consult_loc(loc_familia, medx, medy, dic_water, letter):
	round_loc_familia=(round(loc_familia[0]/.3)*.3, round(loc_familia[1]/.3)*.3)
	orig=utm.to_latlon(1000*round_loc_familia[0]+medx, 1000*round_loc_familia[1]+medy, 56, letter)
	res=(round(orig[1], 3), round(orig[0], 4))
	url="https://api.onwater.io/api/v1/results/"+str(res[1])+","+str(res[0])+"?access_token=zYQHVfPa7cSXhz6Masy6"
	if res not in dic_water[letter]:
		if "bm" in dic_water:
			xpt, ypt = dic_water["bm"]( res[0], res[1] ) # convert to projection map
			land=dic_water["bm"].is_land(xpt, ypt)
		else:
			s={"error":0}
			while "error" in s:
				s=requests.get(url).json()
			land=bool(not s["water"])
		
		
	
		dic_water[letter][res]=(not land)
		upload_loc(res, int((not land)), letter)
		print(len(dic_water[letter]))
	water=dic_water[letter][res]
	return water
	
	
def batch_upload(locs_familia, medx, medy, dic_water, letter):
	to_upload=[]
	for loc_familia in locs_familia:
		round_loc_familia=(round(loc_familia[0]/.3)*.3, round(loc_familia[1]/.3)*.3)
		orig=utm.to_latlon(1000*round_loc_familia[0]+medx, 1000*round_loc_familia[1]+medy, 56, letter)
		res=(round(orig[1], 3), round(orig[0], 4))
		url="https://api.onwater.io/api/v1/results/"+str(res[1])+","+str(res[0])+"?access_token=zYQHVfPa7cSXhz6Masy6"
		if res not in dic_water[letter]:
			#s={"error":0}
			#while "error" in s:
			#	s=requests.get(url).json()
			#dic_water[res]=s["water"]
			xpt, ypt = dic_water["bm"]( res[0], res[1] ) # convert to projection map
			land=dic_water["bm"].is_land(xpt, ypt)
			to_upload.append((res, int((not land))))
			dic_water[letter][res]=(not land)
	if len(to_upload)>0:
		connection=get_connection()
		instruction="insert into onwater_results (longitud, latitud, onwater, letter) values "
		res, water = to_upload[0]
		instruction+=" ("+str(res[0])+", "+str(res[1])+", "+str(water)+", \""+str(letter)+"\") "
		for res, water in to_upload[1:]:
			instruction+=", ("+str(res[0])+", "+str(res[1])+", "+str(water)+", \""+str(letter)+"\") "
		with connection.cursor() as cursor:
			res=cursor.execute(instruction)
		connection.commit()
		connection.close()
	
	print(len(dic_water[letter]))
	#We return False because we need to run this more and more
	if (len(to_upload)/len(locs_familia))>0.01: return False
	return True
		

def upload_loc(res, onw, letter):
	connection=get_connection()
	with connection.cursor() as cursor:
		res=cursor.execute("insert into onwater_results (longitud, latitud, onwater, letter) values ("+str(res[0])+", "+str(res[1])+", "+str(onw)+", \""+str(letter)+"\")")
	connection.commit()
	connection.close()

def get_d2():
	connection=get_connection()
	with connection.cursor() as cursor:
		cursor.execute("select * from onwater_results where letter=\"J\"")
		res=cursor.fetchall()
	connection.close()
	d2j={}
	for s in res:
		d2j[(round(s["longitud"], 3), round(s["latitud"], 4))]=s["onwater"]
		
	connection=get_connection()
	with connection.cursor() as cursor:
		cursor.execute("select * from onwater_results where letter=\"H\"")
		res=cursor.fetchall()
	connection.close()
	d2h={}
	for s in res:
		d2h[(round(s["longitud"], 3), round(s["latitud"], 4))]=s["onwater"]
		
	return {"J":d2j, "H":d2h}


def gen_distancias_proba(seed, xs_min, ys_min, r1, r2, randomfun, params, num_abejas, pi2,  list_cor, min_dist_2, distribution, medx, medy, d2, year):
	locs=[]
	distancias=[]
	
	rmom1=rsel(distribution)
	rmom2=rsel(distribution)
	letter="H"
	if year=="QL": letter="J"
	
	random.seed(seed+5676)
	land=False
	errors=0
	corrects=0
	while corrects<10:
		loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
		land=(not consult_loc(loc_familia, medx, medy, d2, letter))
		corrects+=land
	
	if corrects<3 and "bm" in d2:
		errors=0
		locs_familia=[]
		for k in range(3000):
			loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
			locs_familia.append(loc_familia)
		while (not batch_upload(locs_familia, medx, medy, d2, letter)): 
			locs_familia=[]
			for k in range(3000):
				loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
				locs_familia.append(loc_familia)
			
	random.seed(seed+5676)
	land=False
	while land==False:
		loc_familia=(xs_min+random.random()*r1, ys_min+random.random()*r2)
		land=(not consult_loc(loc_familia, medx, medy, d2, letter))
		
	
	for mem in range(num_abejas):
		vlucht=-1
		while vlucht<0:
			vlucht=randomfun(**params)
			if vlucht>0:
				angle=random.random()*pi2
				loc=(loc_familia[0]+math.cos(angle)*vlucht, loc_familia[1]+math.sin(angle)*vlucht)
				if loc[0]<xs_min or loc[0]>xs_min+r1 or loc[1]<ys_min or loc[1]>ys_min+r2:
					land=False
				else: 
					land=(not consult_loc(loc, medx, medy, d2, letter))
				if land==False: vlucht=-1
		flag=1
		for k in list_cor:
			if abs(loc[0]-k[0])<min_dist_2:
				if abs(loc[1]-k[1])<min_dist_2:
					if euclidean_dist(loc, k)<min_dist_2:
						if flag==0:
							print("Error error "+str(loc))
						flag=0
						rmem={}
						for al in rmom1:
							if random.random()<=0.5:
								rmem[al]=rmom1[al]
							else:
								rmem[al]=rmom2[al]
						locs.append((k, rmem, seed))
	return locs, loc_familia


  
def done_task(tid, res):
	connection = get_connection()
	sql = "update tasks_abejas set task_completed="+str(int(time.time()))+", progress=\""+str(res)+"\" where task_id="+ str(tid) 
	with connection.cursor() as cursor:
		cursor.execute(sql)
	connection.commit()
	connection.close()
	
	

def done_tasks(tids):
	connection = get_connection()
	where= "where task_id=0 "
	for t in tids:
		where+=" or task_id="+str(t)
	sql = "update tasks_abejas set task_completed="+str(int(time.time()))+" "+where
	with connection.cursor() as cursor:
		cursor.execute(sql)
	connection.commit()
	connection.close()
	
	  
	  
def get_geodata(coordenadas, margin):
	
	xaxis  = [utm.from_latlon(coordenadas[l][0],coordenadas[l][1])[0]  for l in coordenadas]
	yaxis =  [utm.from_latlon(coordenadas[l][0],coordenadas[l][1])[1] for l in coordenadas]
	medy  =  np.mean(yaxis)
	medx  =  np.mean(xaxis)

	xaxis  = [to_xy((coordenadas[l][0],coordenadas[l][1]))[0]  for l in coordenadas]
	yaxis = [to_xy((coordenadas[l][0],coordenadas[l][1]))[1] for l in coordenadas]
	medy2  =  np.mean(yaxis)
	medx2  =  np.mean(xaxis)

	information={}
	
	
		
	for k in coordenadas:
		if True:
			information[k]={}
			information[k]["coordenadas"]=coordenadas[k]
			utm_val=utm.from_latlon(coordenadas[k][0], coordenadas[k][1])
			information[k]["utm_xy"]=((utm_val[0]-medx)/1000, (utm_val[1]-medy)/1000)
			xy_val=to_xy(coordenadas[k])
			information[k]["alt_xy"]=((xy_val[0]-medx2)/1000, (xy_val[1]-medy2)/1000)
	
	
	list_cor=list(pd.DataFrame(information ).loc["utm_xy"])
	utm_xs=[i[0] for i in list_cor]
	utm_ys=[i[1] for i in list_cor]
	xs_max=np.max(utm_xs)+margin
	xs_min=np.min(utm_xs)-margin
	ys_max=np.max(utm_ys)+margin
	ys_min=np.min(utm_ys)-margin

	min_dist=20
	for k in range(len(list_cor)-1):
		for j in range(k+1, len(list_cor)):
			a=euclidean_dist(list_cor[k], list_cor[j])
			if a<min_dist: 
				min_dist=a
	
	
	start=time.time()
	min_dist_2=min_dist/2
	pi2=2*math.pi
	r1=(xs_max-xs_min)
	r2=(ys_max-ys_min)
	
	
		
	distancias=[]
	distancias_cert=[]
	i=0
	flag=0
	seed=0
	random_from_flow=1
	nv=(1/random_from_flow)
	return xs_min, ys_min, r1, r2, medx, medy
		


def montecarlobees(num_abejas, bootstrap, randomfun, params, d2, tid, urban=True, rural=True, weird_coords=True, N_mam=269, year=2018, min_dist=20, distance_cert=True, nmam_lim=True,
seedlim=0, margin=20):
	starttime=time.time()
	
	#get basic information and variables
	if year==2018:
		if rural and urban: 
			f=open("distribution.txt", "r")
			dict=f.read()
			f.close()
			distribution=eval(dict)
			tot_abejas=2118
		if rural and (not urban):
			f=open("distribution_rural.txt", "r")
			dict=f.read()
			f.close()
			distribution=eval(dict)
			tot_abejas=1197
		if urban and (not rural):
			f=open("distribution_urban.txt", "r")
			dict=f.read()
			f.close()
			distribution=eval(dict)
			tot_abejas=729
		other_coords=['Caringbah', 'Parramatta', 'Strickland']
		urban_coords=["Marrickville", "Surry Hills", "Camperdown", "Asheville", "Balgowlah "]
		rural_coords=["Jannali", "Sutherland", "Manley", "Lindfield", "Gordon", "Palmbeach", "Stives"]
		locations, coordenadas, distance_national_park=get_loc_coords_dnp()
		
	if year=="kuringbai":
		urban_coords=[]
		rural_coords=[]
		locations, coordenadas, distance_national_park=get_loc_coords_dnp()
		other_coords=["Lindfield", "Gordon", "Stives"]
		f=open("distribution_kuringbai.txt", "r")
		dict=f.read()
		f.close()
		distribution=eval(dict)
		tot_abejas=480
		
	if year==2017:
		urban_coords=[]
		rural_coords=[]
		coordenadas=get_coords_2017()
		other_coords=[i for i in coordenadas]
		f=open("distribution_2017.txt", "r")
		dict=f.read()
		f.close()
		distribution=eval(dict)
		tot_abejas=1342
		
	if year=="QL":
		urban_coords=[]
		rural_coords=[]
		coordenadas=get_coords_QL()
		other_coords=[i for i in coordenadas]
		f=open("distribution_QL.txt", "r")
		dict=f.read()
		f.close()
		distribution=eval(dict)
		tot_abejas=1522
		   
	xaxis  = [utm.from_latlon(coordenadas[l][0],coordenadas[l][1])[0]  for l in coordenadas]
	yaxis =  [utm.from_latlon(coordenadas[l][0],coordenadas[l][1])[1] for l in coordenadas]
	medy  =  np.mean(yaxis)
	medx  =  np.mean(xaxis)

	xaxis  = [to_xy((coordenadas[l][0],coordenadas[l][1]))[0]  for l in coordenadas]
	yaxis = [to_xy((coordenadas[l][0],coordenadas[l][1]))[1] for l in coordenadas]
	medy2  =  np.mean(yaxis)
	medx2  =  np.mean(xaxis)

	information={}
	
	
		
	for k in coordenadas:
		if k not in urban_coords+rural_coords+other_coords: 
			if year!="kuringbai": print(7/0)
			else: print(k+" excluded")
		if ((urban==True) and (k in urban_coords)) or ((rural==True) and (k in rural_coords)) or (((rural==True) and (urban==True)) and k in (rural_coords+urban_coords+other_coords)):
			information[k]={}
			information[k]["coordenadas"]=coordenadas[k]
			utm_val=utm.from_latlon(coordenadas[k][0], coordenadas[k][1])
			information[k]["utm_xy"]=((utm_val[0]-medx)/1000, (utm_val[1]-medy)/1000)
			xy_val=to_xy(coordenadas[k])
			information[k]["alt_xy"]=((xy_val[0]-medx2)/1000, (xy_val[1]-medy2)/1000)
	
	
	list_cor=list(pd.DataFrame(information ).loc["utm_xy"])
	list_cor_keys=list(pd.DataFrame(information).transpose().index)
	utm_xs=[i[0] for i in list_cor]
	utm_ys=[i[1] for i in list_cor]
	xs_max=np.max(utm_xs)+margin
	xs_min=np.min(utm_xs)-margin
	ys_max=np.max(utm_ys)+margin
	ys_min=np.min(utm_ys)-margin
	print("Min dist "+str(min_dist))
	for k in range(len(list_cor)-1):
		for j in range(k+1, len(list_cor)):
			a=euclidean_dist(list_cor[k], list_cor[j])
			if a<min_dist: 
				min_dist=a
	print("Min dist "+str(min_dist))
	
	start=time.time()
	min_dist_2=min_dist/2
	pi2=2*math.pi
	r1=(xs_max-xs_min)
	r2=(ys_max-ys_min)
	
	try: 
		s=randomfun(**params)
	except: 
		return "Invalid function"
		
	distancias=[]
	distancias_cert=[]
	i=0
	flag=0
	seed=0
	random_from_flow=1
	nv=(1/random_from_flow)
	xs_min, ys_min, r1, r2, medx, medy = get_geodata(coordenadas, margin)
	all_locs=[]
	nseed_lim=False
	if not nmam_lim: nseed_lim=True
	while (nmam_lim and flag < N_mam) or (nseed_lim and seed<seedlim):
		a=[]
		while (len(a)==0 and nmam_lim) or (len(a)==0 and nseed_lim and seed<seedlim):
			print(str(flag)+"/"+str(N_mam)+" ("+str(seed)+", "+str(round(nv))+")", end="									  \r")
			a, loc_fam = gen_distancias_proba(5676+bootstrap*100000+seed, xs_min=xs_min, ys_min=ys_min, r1=r1, r2=r2, 
				randomfun=randomfun, params=params,num_abejas=num_abejas, pi2=pi2, list_cor=list_cor, 
				min_dist_2=min_dist_2, distribution=distribution, medx=medx, medy=medy , d2=d2, year=year)
			#all_locs.append(loc_fam)
			seed+=1
		if len(a)>0:
			flag+=1
			if distance_cert:
				distancias_cert.append(a[0])
				a=a[1:]
			if len(a)>0:
				if random_from_flow==1:
					distancias+= a
				else:
					random.seed(seed)
					for p in a:
						if random.random()<random_from_flow:
							distancias.append(p)
			
				if len(distancias)>20000:
					random.seed(seed)
					random_from_flow=random_from_flow*(tot_abejas/len(distancias))
					distancias=random.sample(distancias, tot_abejas)
					nv=(1/random_from_flow)
		
	random.seed(seed)
	if len(distancias)>tot_abejas-len(distancias_cert):
		distancias=random.sample(distancias, tot_abejas-len(distancias_cert))
	
	distancias+=distancias_cert
	sitios_l=sorted([k for k in coordenadas])
	nl=[]
	for p in distancias: 
		sit=list_cor_keys[list_cor.index(p[0])]
		nl.append({"Familia":p[2], "Sitio":sitios_l.index(sit)})
	df_fams=pd.DataFrame(nl)
	df_fams["cons"]=1
	df_fams=df_fams.groupby(["Familia", "Sitio"]).sum().reset_index()
	
	conn=get_connection()
	with conn.cursor() as cursor:
		for row in df_fams.iterrows():
			sql="Insert into familias_abejas_conteo (task_id, familia_id, sitio_id, conteo, total ) values ("+str(tid)+", "+str(row[1]["Familia"])+", "+str(row[1]["Sitio"])+", "+str(row[1]["cons"])+", "+str(seed)+")"
			cursor.execute(sql)
	conn.commit()
	conn.close()
	
	
	flag=0
	dfisis=[]
	dgenis=[]
	#Now we set the distribution to whatever we have gotten
	old_dis=distribution
	aleles=["Al1", "Al2", "Al3", "Al4", "Al5", "Al6", "Al7"]
	distribution={}
	for alele in aleles:
		distribution[alele]={}
		for loc, abeja, fam in distancias:
			if abeja[alele] not in distribution[alele]: distribution[alele][abeja[alele]]=0
			distribution[alele][abeja[alele]]+=1/float(len(distancias))
	i=0
	for  loc1, abeja1, fami in distancias:
		j=0
		for  loc2, abeja2, famo in distancias:
			if j>i:
				dgen=distancia_genetica4(abeja1, abeja2, distribution, N_mam=N_mam)
				dfis=euclidean_dist(loc1, loc2)
				dfisis.append(dfis)
				dgenis.append(dgen)
			j=j+1
		i=i+1

	distankias=pd.DataFrame({"Similitud_genetica_4":dgenis, "Distancia":dfisis})

	ma1=5
	ma2=5
	dict_res={}
	for k in range(0, 101, 10):
		#print(k, end="   \r")
		cropped=distankias[(distankias.Similitud_genetica_4>= (k-ma1)/100.0) & 
								 (distankias.Similitud_genetica_4< (k+ma1)/100.0)]
		leni=len(cropped)
		xs=[]
		for s in range(0, 50):
			#cropped["d2"]=[i+random.normalvariate(0,1) for i in cropped["Distancia"]]
			if leni>0:
				l1=len(cropped[(cropped.Distancia < (s))])/leni
			else: l1=1
			xs.append(l1)
	
		dict_res[k]=xs

	conn=get_connection()
	with conn.cursor() as cursor:
		for k in range(0, 101, 10):
			sql="Insert into abejas_curvas (task_id, k, Int_0,  Int_1,  Int_2,  Int_3,  Int_4,  Int_5,  Int_6,  Int_7,  Int_8,  Int_9,  Int_10,  Int_11,  Int_12,  Int_13,  Int_14,  Int_15,  Int_16,  Int_17,  Int_18,  Int_19,  Int_20,  Int_21,  Int_22,  Int_23,  Int_24,  Int_25,  Int_26,  Int_27,  Int_28,  Int_29,  Int_30,  Int_31,  Int_32,  Int_33,  Int_34,  Int_35,  Int_36,  Int_37,  Int_38,  Int_39,  Int_40,  Int_41,  Int_42,  Int_43,  Int_44,  Int_45,  Int_46,  Int_47,  Int_48,  Int_49) values "+(
			"("+str(tid))+", "+str(k)+", "+(", ").join([str(y) for y in dict_res[k]])+")"
			cursor.execute(sql)
	conn.commit()
	conn.close()
		
	
	tot=len(distankias)
	conn=get_connection()
	with conn.cursor() as cursor:
		sql="Insert into abejas_planos (task_id, gen_dis, fis_dis, mean ) values "
		vals_to_add=[]
		for k in range(10):
			for j in range(10):
				cropped=distankias[(distankias.Similitud_genetica_4>= (j)/10.0) & 
								 (distankias.Similitud_genetica_4< (j+1)/10.0) & 
								 (distankias.Distancia>= (k*3)) & 
								 ( distankias.Distancia< ((k+1)*3) ) ]
				vals_to_add.append("("+str(tid)+", "+str(j)+", "+str(k)+", "+str(round(len(cropped)/float(tot), 15))+")")
		sql=sql+", ".join(vals_to_add)+", "+"("+str(tid)+", "+str(-1)+", "+str(-1)+", "+str(float(tot))+")"
		cursor.execute(sql)
	conn.commit()
	conn.close() 
	
	return {"message":"Completed in "+str(int(round((time.time()-starttime)/60)))+" minutes"}
	
	
	