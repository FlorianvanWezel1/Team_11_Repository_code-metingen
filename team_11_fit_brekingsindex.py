import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

df =pd.read_csv("Project/team_11_brekingsindex_bepalen/team_11_metingen.csv") # Uitlezen van de csv file
print(df)


i = np.radians(df['hoek_graden'].values) #Omreken gegeven hoek in graden naar radialen
N = df['aantal_franjes'].values # Aantal gemeten franjes oproepen
onzekerheid = df['onzekerheid_N'].values #Gegeven onzekerheid oproepen

def werking (i,i_0,n):
    #constante
    d = 0.003
    l = 532*10**-9
   # return((2*d/l)*(np.sqrt(n**2 - np.sin(i)**2)-np.cos(i)+(1-n))) Orginele formule
    return ((d/l)*((n-1)/n)*(i-i_0)**2) # Verder afgeleiden formule voor kleine hoek en een correctie voor de onzekerheid van de beginpositie van het plaatje 


popt, pcov = curve_fit(werking, i, N, p0=[0,1.5], sigma=onzekerheid, absolute_sigma=True) # Curvefit
n_fit = popt[1] # De gefitte brekingsindex
n_onzekerheid = np.sqrt(pcov[1,1]) # Onzekerheid vasn de gefitte n
print(f"Brekingsindex n = {n_fit:.4f} ± {n_onzekerheid:.4f}")

#Grafiek maken
plt.errorbar(i,N,yerr=onzekerheid,fmt='o')
plt.plot(i,werking(i,*popt))
plt.xlabel('hoek van inval (rad)')
plt.ylabel('aantal franjes (-)')
plt.text(0.2,400,s="brekingsindex n = 1.5064 ± 0.0398")
plt.savefig("team_11_fit_brekingsindex_plot.png", dpi=300, bbox_inches="tight")
plt.show()
