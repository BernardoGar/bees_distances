import matplotlib.pyplot as plt
import numpy as np
import utm
import pandas as pd
import math
from matplotlib.pyplot import figure
from matplotlib import cm as CM
from matplotlib import mlab as ML

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


def distancia_genetica4(a, b, distribution):
    P_A1_a1=1
    P_A2_a2=1
    N=269
    P_A2_a2_hermano_a2=1
    for alele in distribution:
        P_A1_a1=P_A1_a1*(distribution[alele][a[alele]])
        P_A2_a2=P_A2_a2*(distribution[alele][b[alele]])
        P_A2_a2_hermano_a2=P_A2_a2_hermano_a2*((1/2)*(int(a[alele]==b[alele]))+int(a[alele]!=b[alele])*(1/2)*(distribution[alele][b[alele]])/(1-distribution[alele][a[alele]]))
    P_A1_A2_distinta_mama=P_A1_a1*P_A2_a2
    P_A1_A2_misma_mama=P_A1_a1*P_A2_a2_hermano_a2
    return (1/(1+(N-1)*P_A1_A2_distinta_mama/P_A1_A2_misma_mama))
    
            
def to_xy(point):
    r=6371000
    phi_0 = 151.16067
    cos_phi_0 = math.cos(math.radians(phi_0))
    lam = point[0]
    phi = point[1]
    return (r * math.radians(lam) * cos_phi_0, r * math.radians(phi))



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