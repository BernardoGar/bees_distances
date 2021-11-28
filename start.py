import random
import requests
from urllib.request import urlopen
from libreria import *
from multiprocessing import Manager
import time
import sys

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if "-f" in opts:
    print("Forcing tasks: "+str(args))
    
    

external_ip = urlopen('https://ident.me').read().decode('utf8')

d2=get_d2()

if basemap_exists:
	bm = Basemap(resolution='f', llcrnrlat=-35.499399,
		llcrnrlon=147.016745,
		urcrnrlat=-23.341214,
		urcrnrlon=158.101763)   # default: projection='cyl'
	d2["bm"]=bm



f=open("diccionario_distfisica_v_distgenetica.txt", "r")
diccionario_distfisica_v_distgenetica=eval(f.read())
f.close()


match=0
banned=0
while match==0:
    match=1
    todos=get_task(external_ip, args)
    a=time.time()
    if todos!=False:
        ress=[]
        lookedup_info=gen_lookups(todos)
        for todo in todos:
            print("Starting task: "+str(todo))
            try: 
            	ress.append(eval(todo["task_name"])(**eval(todo["task_params"]), tid=todo["task_id"], lookedup_info=lookedup_info[todo["task_id"]]))
            except:
            	ress.append(eval(todo["task_name"])(**eval(todo["task_params"]), tid=todo["task_id"]))
            match=0
        inserts=[]
        for r in ress:
            if "insert" in r:
                inserts.append(r["insert"])
        do_insertions(inserts)
        done_tasks([todo["task_id"] for todo in todos])
    b=time.time()
    if round((b-a)/60)>1: print("Ended task in  "+str(int(round((b-a)/60)))+" minutes")
    else: print("Ended task in  "+str(int(round((b-a))))+" seconds")
    

