# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:50:34 2024

@author: Aaron
"""

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path, reconstruct_path
graph = [
[0, 1, 2, 0, 0],
[0, 0, 0, 1, 0],
[0, 0, 0, 0, 2],
[0, 0, 10, 0, 5],
[0, 0, 0, 0, 0]
]

graph = [
[0, 1, 0, 1, 0],
[0, 0, 2, 0, 0],
[0, 0, 0, 3, 0],
[0, 0, 0, 0, 4],
[0, 0, 0, 0, 0]
] 
graph = csr_matrix(graph)
print(graph)

dist_matrix, predecessors = shortest_path(csgraph=graph, indices=1, return_predecessors=True)
dist_matrix

predecessors
cstree = reconstruct_path(csgraph=graph, predecessors=predecessors, directed=False)
cstree

############# build a distance matrix from minima peaks
nodes = []
for i,r in enumerate(alltangs):
    for x in r:
        nodes.append([i,x])
        if i>0:
            nodes.append([i,x-360])
nodes = np.array(nodes)
graph = np.zeros((len(nodes),len(nodes)))
#go through each node and measure the distance to the other nodes it is near
for i,n in enumerate(nodes[:-1]):
    nxtfr = nodes[np.where(nodes[:,0]==n[0]+1)]
    if n[1]>0:
        temp = nxtfr[:,1].copy()
        temp[temp<0] += 360
        graph[i,np.where(nodes[:,0]==n[0]+1)] = abs(n[1]-nxtfr[:,1])
        print('first',n[1],nxtfr[:,1],abs(n[1]-nxtfr[:,1]))
    else:
        graph[i,np.where(nodes[:,0]==n[0]+1)] = abs(n[1]+abs(nxtfr[:,1]))
        print(n[1], nxtfr[:,1], abs(n[1]+abs(nxtfr[:,1])))
        
        
  for i,n in enumerate(nodes[:-len(np.where(nodes[:,0]==len(alltangs)-1)[0])]):
      nxtfr = nodes[np.where(nodes[:,0]==n[0]+1)]
      nxtfrind = np.where(nodes[:,0]==n[0]+1)[0]
      for j,o in enumerate(nxtfr):
          if n[1]>0:
              if o[1]>0:
                  graph[i,nxtfrind[j]] = abs(n[1]-o[1])
              else:
                  graph[i,nxtfrind[j]] = abs(abs(n[1]-o[1])-360)
          else:
              if o[1]>0:
                  graph[i,nxtfrind[j]] = abs(abs(n[1]+o[1])-360)
              else:
                  graph[i,nxtfrind[j]] = abs(n[1]-o[1])
                  
                  
############# distances add up to 360
for i,n in enumerate(nodes[:-len(np.where(nodes[:,0]==len(alltangs)-1)[0])]):
    nxtfr = nodes[np.where(nodes[:,0]==n[0]+1)]
    nxtfrind = np.where(nodes[:,0]==n[0]+1)[0]
    for j,o in enumerate(nxtfr):
        if n[1]>0:
            if o[1]>0:
                graph[i,nxtfrind[j]] = abs(n[1]-o[1])
            else:
                graph[i,nxtfrind[j]] = abs(abs(n[1]-(o[1]+360))-360)
        else:
            if o[1]>0:
                graph[i,nxtfrind[j]] = abs(abs((n[1]+360)-o[1])-360)
            else:
                graph[i,nxtfrind[j]] = abs(n[1]-o[1])   
                
                
                
     
      
#### chose a predecessor number and find the path that leads to it
p = 467
path = []
while p != -9999:
    p = predecessors[p]
    path.append(p)
path = path[:-1]
plt.plot(np.diff(nodes[path][:,1]))


new = nodes[path][:,1]
new[new>180] -=360
new[new<-180] += 360
plt.plot(new)


















dist_matrix, predecessors = shortest_path(csgraph=graph, indices=1, return_predecessors=True)
dist_matrix

predecessors
cstree = reconstruct_path(csgraph=graph, predecessors=predecessors, directed=False)
cstree

############# build a distance matrix from minima peaks
nodes = []
for i,r in enumerate(alltangs):
    for x in r:
        nodes.append([i,x])
        if i>0:
            nodes.append([i,x-360])
nodes = np.array(nodes)
graph = np.zeros((len(nodes),len(nodes)))
############# distances add up to 360
for i,n in enumerate(nodes[:-len(np.where(nodes[:,0]==len(alltangs)-1)[0])]):
    nxtfr = nodes[np.where(nodes[:,0]==n[0]+1)]
    nxtfrind = np.where(nodes[:,0]==n[0]+1)[0]
    for j,o in enumerate(nxtfr):
        if n[1]>0:
            if o[1]>0:
                graph[i,nxtfrind[j]] = abs(n[1]-o[1])
            else:
                graph[i,nxtfrind[j]] = abs(abs(n[1]-(o[1]+360))-360)
        else:
            if o[1]>0:
                graph[i,nxtfrind[j]] = abs(abs((n[1]+360)-o[1])-360)
            else:
                graph[i,nxtfrind[j]] = abs(n[1]-o[1])   
                
#for each "seed" get distances to the endpoints and find the 
for l in range(len(np.where(nodes[:,0]==0)[0])):
    dist_matrix, predecessors = shortest_path(csgraph=graph, indices=l, return_predecessors=True)
                
                
     
      
#### chose a predecessor number and find the path that leads to it
p = 581
path = []
while p != -9999:
    p = predecessors[p]
    path.append(p)
path = path[:-1]
plt.plot(np.diff(nodes[path][:,1]))


new = nodes[path][:,1]
new[new>180] -=360
new[new<-180] += 360
plt.plot(new)






####### differential look-ahead?
allmins = []

for y, wp in enumerate(alltangs):
    #remove duplicates and sort
    wp = np.sort(np.array(list(set(wp))))
    if bool(wp.size == 0):
        allmins.append(allmins[-1])
    elif y == 0:
        # allmins.append(wp[np.argmin(abs(wp))])
        allmins.append(wp[0])
    else:
        choice = np.zeros((len(wp),2))
        lookforward = 30
        for iw, w in enumerate(wp):
            ############# build a distance matrix from minima peaks
            nodes = [[-1,w]]
            for i,r in enumerate(alltangs[y+1:int(y+1+min(lookforward,abs(len(alltangs)-(y+1))))]):
                for x in r:
                    nodes.append([i,x])
                    nodes.append([i,x-360])
            nodes = np.array(nodes)
            nodes[:,0] = nodes[:,0]+1
            graph = np.zeros((len(nodes),len(nodes)))
            ############# distances add up to 360
            lookforward = min(lookforward, np.max(nodes[:,0]))
            for i,n in enumerate(nodes[:-len(np.where(nodes[:,0]==lookforward)[0])]):
                nxtfr = nodes[np.where(nodes[:,0]==n[0]+1)]
                nxtfrind = np.where(nodes[:,0]==n[0]+1)[0]
                for j,o in enumerate(nxtfr):
                    if n[1]>0:
                        if o[1]>0:
                            graph[i,nxtfrind[j]] = abs(n[1]-o[1])
                        else:
                            graph[i,nxtfrind[j]] = abs(abs(n[1]-(o[1]+360))-360)
                    else:
                        if o[1]>0:
                            graph[i,nxtfrind[j]] = abs(abs((n[1]+360)-o[1])-360)
                        else:
                            graph[i,nxtfrind[j]] = abs(n[1]-o[1])
            dist_matrix = shortest_path(csgraph=graph, indices=0)
            # print(wp,nodes, np.min(dist_matrix[np.where(nodes[:,0]==lookforward)[0]]))
            #add mindist
            choice[iw,:] = [w,np.min(dist_matrix[np.where(nodes[:,0]==lookforward)[0]])]
            
            
        print(choice)
        allmins.append(choice[np.argmin(choice[:,1]),0])
        
        
        
        
      
####### differential look-ahead?
allmins = []

for y, wp in enumerate(alltangs):
    #remove duplicates and sort
    wp = np.sort(np.array(list(set(wp))))
    if bool(wp.size == 0):
        allmins.append(allmins[-1])
    elif y == 0:
        # allmins.append(wp[np.argmin(abs(wp))])
        allmins.append(wp[0])
    else:
        choice = np.zeros((len(wp),2))
        lookforward = 5

        ############# build a distance matrix from minima peaks
        nodes = [[-1,allmins[-1]]]
        for i,r in enumerate(alltangs[y:int(y+min(lookforward,abs(len(alltangs)-y)))]):
            for x in r:
                nodes.append([i,x])
                nodes.append([i,x-360])
        nodes = np.array(nodes)
        nodes[:,0] = nodes[:,0]+1
        graph = np.zeros((len(nodes),len(nodes)))
        ############# distances add up to 360
        lookforward = min(lookforward, np.max(nodes[:,0]))
        for i,n in enumerate(nodes[:-len(np.where(nodes[:,0]==lookforward)[0])]):
            nxtfr = nodes[np.where(nodes[:,0]==n[0]+1)]
            nxtfrind = np.where(nodes[:,0]==n[0]+1)[0]
            for j,o in enumerate(nxtfr):
                if n[1]>0:
                    if o[1]>0:
                        graph[i,nxtfrind[j]] = abs(n[1]-o[1])
                    else:
                        graph[i,nxtfrind[j]] = abs(abs(n[1]-(o[1]+360))-360)
                else:
                    if o[1]>0:
                        graph[i,nxtfrind[j]] = abs(abs((n[1]+360)-o[1])-360)
                    else:
                        graph[i,nxtfrind[j]] = abs(n[1]-o[1])
        dist_matrix, predecessors = shortest_path(csgraph=graph, indices=0, return_predecessors=True)
        # print(wp,nodes, np.min(dist_matrix[np.where(nodes[:,0]==lookforward)[0]]))
        #add mindist
        # print(nodes[np.where(dist_matrix == np.min(dist_matrix[np.where(nodes[:,0]==lookforward)[0]]))[0],1][0])
        allmins.append(nodes[np.where(dist_matrix == np.min(dist_matrix[np.where(nodes[:,0]==lookforward)[0]]))[0],1][0])
            
            
        # print(choice)
        # allmins.append(choice[np.argmin(choice[:,1]),0])