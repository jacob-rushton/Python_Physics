#This code is designed to rediscover the Curie temperature by trying to find
#the lowest energy state of a system of 50x50 electrons at several different 
#temperatures and then creating animations because animations are neat.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

plt.clf()
electrons=np.random.choice([-1,1],size=(50,50)) #50x50 array
def hamiltonian(a,J=1):
    l=np.roll(a,1,axis=1) ##Creating rolled arrays to sum close pairs simultaneously
    r=np.roll(a,-1,axis=1)
    t=np.roll(a,1,axis=0)
    b=np.roll(a,-1,axis=0)
    return -1*np.sum(a*(l+r+t+b))

#using mod function; if (i+-1) or (j+-1)=50, resets it to 0
#Only need to modify energy by the 5 close pairs affected, so I made this function
def modsum(a,i,j):
    p1=(a[i,j]*a[(i+1)%50,j]+a[i,j]*a[(i-1)%50,j]+a[i,j]*a[i,(j+1)%50]+a[i,j]*a[i,(j-1)%50])
    p2=(a[(i+1)%50,j]*a[(i+2)%50,j]+a[(i+1)%50,j]*a[i,j]+a[(i+1)%50,j]*a[(i+1)%50,(j+1)%50]+a[(i+1)%50,j]*a[i,(j-1)%50])
    p3=(a[(i-1)%50,j]*a[i,j]+a[(i-1)%50,j]*a[(i-2)%50,j]+a[(i-1)%50,j]*a[(i-1)%50,(j+1)%50]+a[(i-1)%50,j]*a[(i-1)%50,(j-1)%50])
    p4=(a[i,(j+1)%50]*a[(i+1)%50,(j+1)%50]+a[i,(j+1)%50]*a[(i-1)%50,(j+1)%50]+a[i,(j+1)%50]*a[i,(j+2)%50]+a[i,(j+1)%50]*a[i,j])
    p5=(a[i,(j-1)%50]*a[(i+1)%50,(j-1)%50]+a[i,(j-1)%50]*a[(i-1)%50,(j-1)%50]+a[i,(j-1)%50]*a[i,j]+a[i,(j-1)%50]*a[i,(j-2)%50])
    q1=((-1)*a[i,j]*a[(i+1)%50,j]+(-1)*a[i,j]*a[(i-1)%50,j]+(-1)*a[i,j]*a[i,(j+1)%50]+(-1)*a[i,j]*a[i,(j-1)%50])
    q2=(a[(i+1)%50,j]*a[(i+2)%50,j]+a[(i+1)%50,j]*(-1)*a[i,j]+a[(i+1)%50,j]*a[(i+1)%50,(j+1)%50]+a[(i+1)%50,j]*a[i,(j-1)%50])
    q3=(a[(i-1)%50,j]*(-1)*a[i,j]+a[(i-1)%50,j]*a[(i-2)%50,j]+a[(i-1)%50,j]*a[(i-1)%50,(j+1)%50]+a[(i-1)%50,j]*a[(i-1)%50,(j-1)%50])
    q4=(a[i,(j+1)%50]*a[(i+1)%50,(j+1)%50]+a[i,(j+1)%50]*a[(i-1)%50,(j+1)%50]+a[i,(j+1)%50]*a[i,(j+2)%50]+a[i,(j+1)%50]*(-1)*a[i,j])
    q5=(a[i,(j-1)%50]*a[(i+1)%50,(j-1)%50]+a[i,(j-1)%50]*a[(i-1)%50,(j-1)%50]+a[i,(j-1)%50]*(-1)*a[i,j]+a[i,(j-1)%50]*a[i,(j-2)%50])
    sum_p=(-1)*(p1+p2+p3+p4+p5)
    sum_q=(-1)*(q1+q2+q3+q4+q5)
    return sum_p,sum_q  #sum_p is the original sum of the close pairs
#sum_q is sum with a[i,j]'s spin flipped


def convergence(a,T,stepmax=600000):
    step=1
    c=np.copy(a)
    E=hamiltonian(c)
    while step<=stepmax:
        i=np.random.randint(0,high=50)
        j=np.random.randint(0,high=50)
        sum_p,sum_q=modsum(c,i,j)
        if sum_q<sum_p:  #this is a simplification of E-sum_p+sum_q<E
            c[i,j]=(-1)*c[i,j]
            E=E-sum_p+sum_q
        else:
            p=np.exp(-((sum_q-sum_p)/T)) #simplified ((E-sum_p+sum_q)-E)
            k=np.random.random()  #random number to compare to probability
            if p>=k: #meaning k is within probability
                c[i,j]=(-1)*c[i,j]
                E=E-sum_p+sum_q
        step=step+1
    return c   #c is final accepted array iteration
        

#I decided to write a 2nd function stricly for animation. This one requires 
#more array creation and is thus less optimized than the above version.
#it also returns a list of all the iterations instead of just one final array
def convergence_b(a,T,stepmax=600000):
    step=1
    b=[a]
    c=np.copy(a)
    E=hamiltonian(a)
    while step<=stepmax:
        i=np.random.randint(0,high=50)
        j=np.random.randint(0,high=50)
        sum_p,sum_q=modsum(c,i,j)
        if sum_q<sum_p:  #this is a simplification of E-sum_p+sum_q<E
            d=np.copy(c) #Need new array so all elements in list aren't all the same
            d[i,j]=(-1)*d[i,j]
            b.append(d)
            c=d #Updating c to keep list elements unique w/o using extra memory           
        else:
            p=np.exp(-((sum_q-sum_p)/T)) #simplified ((E-sum_p+sum_q)-E)
            k=np.random.random()  #as above
            if p>=k: 
                d=np.copy(c)
                d[i,j]=(-1)*d[i,j]
                b.append(d)
                c=d
                E=E-sum_p+sum_q
        step=step+1
    return b #b is a list of all accepted array iterations

Temp=[0.01,0.1,1.,2.,3.,4.,5.,10.,100.] #temps for graph
AniTemp=[0.1,2.5,100] #temps for animations



def Tcurie(a,Temp,):
    tests=np.empty((5,9)) # 2D array 
    for j in range(9):
        for i in range(5):
            b=convergence(a,Temp[j])
            tests[i,j]=np.sum(b) #now has 5 moment trials in each column
    moments=np.array([np.max(tests[:,0]),np.max(tests[:,1]),np.max(tests[:,2]),np.max(tests[:,3]),np.max(tests[:,4]),np.max(tests[:,5]),np.max(tests[:,6]),np.max(tests[:,7]),np.max(tests[:,8])])
    plt.plot(Temp,moments,'g') #Now I start labelling stuff
    plt.title('Average Magnetic Moment as a Function of Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetic Moment')
    plt.savefig('Tcurie.pdf')
    plt.show() #Enjoy the preview
    
    
#Creating the animations
fig=plt.figure()
states01=[] #empty list to append wanted frames to
states25=[]
states100=[]
lats01=convergence_b(electrons,0.1)
lats25=convergence_b(electrons,2.5)
lats100=convergence_b(electrons,100)

for n in lats01[0::500]:  #slicing to get desired frames
    states01.append((plt.imshow(n,cmap='OrRd'),)) 
movie01=ani.ArtistAnimation(fig,states01,repeat=False)
movie01.save('temp_0.1.mp4')
plt.cla()

for n in lats25[0::500]:  #slicing to get desired frames
    states25.append((plt.imshow(n,cmap='OrRd'),)) 
movie25=ani.ArtistAnimation(fig,states25,repeat=False)
movie25.save('temp_2.5.mp4')
plt.cla()

for n in lats100[0::500]:  #slicing to get desired frames
        states100.append((plt.imshow(n,cmap='OrRd'),)) 
movie100=ani.ArtistAnimation(fig,states100,repeat=False)
movie100.save('temp_100.mp4')
plt.cla()

def update_imshow(num,array):
    plt.imshow(array[num])


states=[]
fig=plt.figure()
for T in AniTemp:
    lats=convergence_b(electrons,T)
            
        

    
        

        