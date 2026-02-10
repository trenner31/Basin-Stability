import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize, TwoSlopeNorm
from functools import partial
from scipy.integrate import odeint
from scipy.optimize import brentq
from multiprocessing import Pool
import os
from tqdm import tqdm
"""
First Model, Notation is the same as in the paper

Menck, P., Heitzig, J., Marwan, N. et al. 
How basin stability complements the linear-stability paradigm. 
Nature Phys 9, 89–92 (2013). 
https://doi.org/10.1038/nphys2516

"""

def forest_cover_derivative(C_crit, x, r, C, A):
    """
    Returns the derivative dC/dt of the Tree cover T with respect to time t. 
    
    :param C_crit: Critical Forest Cover
    :param x: Extinction Rate
    :param r: Saturation Rate
    :param C: Forest Cover
    :param A: Aridity
    
    """
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = C > C_crit
    result[~mask] = -x * C[~mask]
    result[mask] = r * (1 - C[mask]) * C[mask] - x * C[mask]

    return result

def forest_cover(C_crit, x, r, m, b, C, A):
    """
    Returns the State a Start condition will converge to -1 for Savanna and +1 for Forrest
    
    :param x: Extinction Rate
    :param r: Saturation Rate
    :param m: Slope of critical cover as a function of aridity
    :param b: y-intercept of critical cover as a function of aridity
    :param C: Forest Cover
    :param A: Aridity
    
    """
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = (1-x/r > calc_C_crit(A,m,b)) & (C>calc_C_crit(A,m,b))
    result[~mask] = -1
    result[mask] = 1

    return result

def calc_C_crit(A,m,b):
    """
    Returns the critical forest cover threshold as a function of aridity
    
    :param A: Aridity
    :param m: Slope
    :param b: y-intercept
    """
    return np.maximum(A*m+b,0)

def calc_C_F(x,r):
    """
    Returns the Forest Equilibrium value
    
    :param x: Extinction Rate
    :param r: Saturation Rate
    """
    return(1-x/r)

def calc_A_crit(x,r,m,b):
    """
    Calculates the critical aridity above which no forest state exists
    
    :param x: Extinction Rate
    :param r: Saturation Rate
    :param m: Slope of critical cover as a function of aridity
    :param b: y-intercept of critical cover as a function of aridity
    """
    return (calc_C_F(x,r)-b)/m

def calc_U(x,r,C_crit,C):
    """
    Calculates a potential U derived according to the forest cover derivative.
    The slope gives the speed and direction of forest cover developement.
    
    :param x: Extinction Rate
    :param r: Saturation Rate
    :param C_crit: Critical Forest Cover
    :param C: Forest Cover
    """
    return x/2*C**2+ (C>C_crit)*(r/3 * (C**3-C_crit**3) - r/2 *(C**2-C_crit**2))

def calc_linear_stability(x,r,C_crit,C):
    """
    Returns the value dF(C)/dC derivative of dC/dt with C_F=1-x/r
    Result is eigenvalue, because of r>x>0 is eigenvalue<0 ->stability, always comes back to initial state
    Result is how fast it goes back use positive value for x-r ->abs(x-r)
    
    :param x: Extinction Rate
    :param r: Saturation Rate
    :param C_crit: Critical Forest Cover
    :param C: Forest Cover
    """

    return(C>C_crit)*(abs(x-r))

def calc_basin_stability(A,A_crit,C_crit):
    """
    Returns the volume of the Forest State which is C_crit to 1 (100% coverage),
    A<A_crit because basin exists only if forest is stable
    
    :param A: Aridity
    :param A_crit: Critical Aridity
    :param C_crit: Critical Forest Cover
    """

    return(A<A_crit)*(1-C_crit)
    
def dCdt(C,t,A):
    """
    Helper Function that applies the linear function for the critical value to the forest_cover_derivative
    Time variable as input is needed for ode solvers but unused
    
    :param C: Forest Cover
    :param t: Time
    :param A: Aridity
    """
    return forest_cover_derivative(C_crit=calc_C_crit(A=A,m=m,b=b),x=x,r=r,C=C,A=A)

def make_plot_noarrows(A,C,func_cover,func_crit,C_F,cmap=ListedColormap(['lightgray','grey']),xlabel=None,ylabel=None,title=None,figsize=None,output_file=None):
    """
    Reproduces plot from initial paper that shows the regions that converge to the forest state and savanna state
    
    :param A: Aridity
    :param C: Forest Cover
    :param func_cover: Function that describes the state an A,C combination converges to
    :param func_crit: Function that describes dependence of the critical forest cover on aridity
    :param C_F: Forest cover equilibrium state
    """    
    AA, CC = np.meshgrid(A, C)
    C_crit=func_crit(A)
    # Calculates the critical aridity and the index corresponding to it
    idx_crit = np.argmin(abs(C_crit-C_F))
    A_crit = A[idx_crit]
    
    State=func_cover(C_crit=C_crit,C=CC,A=AA)

    fig, ax = plt.subplots(figsize=figsize)
    
    #Colors basins of attraction
    ax.pcolormesh(AA, CC,State, shading='auto', cmap=cmap)
    
    # Ploting the critical cover function
    ax.plot(A, C_crit, color='white', linewidth=2)
    text_idx = int(idx_crit * 0.5) 
    text_x = A[text_idx]
    text_y = C_crit[text_idx]
    ax.text(text_x, text_y + 0.02, r'$C_{crit}(A)$', 
            color='white', 
            fontsize=12, 
            fontweight='bold', 
            rotation=28, 
            ha='center', 
            va='bottom')

    # Other helpful lines
    ax.plot([0,A_crit],[C_F,C_F],color='white')
    ax.plot([A_crit,A_crit],[0,1],linestyle='--',color='white')
    ax.plot([0,np.max(A)],[C_F,C_F],linestyle='--',color='white')
    ax.scatter([A_crit],[C_F],color='white', zorder=5)
    
    xticks =  [A_crit]
    xlabels = [ r"$A_{crit}$"]
    ax.set_xticks(xticks, xlabels)
    yticks = list(ax.get_yticks()) + [C_F]
    ylabels = [*ax.get_yticklabels(), r"$C_F=$"+format(C_F,".1f")]
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    ax.set_ylim(0,1)

    # Legend
    legend_handles = [
        matplotlib.patches.Patch(color=cmap.colors[0], label="Savanna State"),
        matplotlib.patches.Patch(color=cmap.colors[1], label="Forest State")
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f"Plot gespeichert unter: {output_file}")
    plt.close()
    return fig,ax

def make_plot_arrows(A,C,func_cover,func_cover_derivative,func_crit,C_F,resolution_sparse=20,cmap=ListedColormap(['lightgray','grey']),xlabel=None,ylabel=None,title=None,figsize=None,output_file=None):
    """
    Adds arrows to initial plot to show speed of the change of forest cover
    
    :param A: Aridity
    :param C: Forest Cover
    :param func_cover: Function that describes the state an A,C combination converges to
    :param func_crit: Function that describes dependence of the critical forest cover on aridity
    :param C_F: Forest cover equilibrium state
    :param resolution_sparse: Number of points with arrows in each direction
    """      
    fig,ax=make_plot_noarrows(figsize=figsize,
                    A=A, C=C,
                    func_cover=func_cover,
                    func_crit=func_crit,
                    C_F=C_F,
                    cmap=cmap,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,)
    
    A_sparse=np.linspace(np.min(A),np.max(A),resolution_sparse)
    C_sparse=np.linspace(np.min(C),np.max(C),resolution_sparse)
    C_crit_sparse=func_crit(A_sparse)
    AA_sparse, CC_sparse = np.meshgrid(A_sparse, C_sparse)
    Direction_sparse=func_cover_derivative(C_crit=C_crit_sparse,C=CC_sparse,A=AA_sparse)

    #This shows the arrows
    ax.quiver(AA_sparse,CC_sparse, 0*np.ones_like(AA_sparse), Direction_sparse)

    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f"Plot gespeichert unter: {output_file}")
    plt.close()
    return fig,ax

def make_plot_potential(A_range,C,func_crit,func_potential,A_crit,C_F,colors,xlabel=None,ylabel=None,title=None,figsize=None,output_file=None):
    """
    Creates a plot of the shapes of the potential associated with the model depending on the aridity
    
    :param A_range: Array of aridities
    :param C: Forest Cover
    :param func_crit: Function that describes dependence of the critical forest cover on aridity
    :param func_potential: Function that describes the potential
    :param A_crit: Critical Aridity
    :param C_F: Forest Cover Equlibrium
    :param colors: Array of colors
    """
    fig, ax = plt.subplots(figsize=figsize)

    for A_int,i,col in zip(A_range,range(len(A_range)),colors):
        C_crit_int=func_crit(A_int)
        ax.plot(C,func_potential(C_crit=C_crit_int,C=C)+i*3e-3, label=r'A='+format(A_int/A_crit, ".1f")+r'$A_{crit}$', color=col)
        ax.scatter(C_crit_int,func_potential(C_crit=C_crit_int,C=C_crit_int)+i*3e-3, color=col)
    ax.axvline(C_F, linestyle='--', color='black')
    ax.axvline(0, linestyle='--',color='black')

    xticks = list(ax.get_xticks()) + [0.0, C_F]
    xlabels = [*ax.get_xticklabels(), r"$C_S=$"+format(0.0,".2f"), r"$C_F=$"+format(C_F,".2f")]
 
    ax.set_xticks(xticks, xlabels, rotation=35)
    ax.set_xlim(-3e-2,1)

    circle_marker = matplotlib.lines.Line2D([], [], marker='o',color='black',markersize=6,linestyle='None',label=r'$C_{crit}(A)$')
    handles,_x = ax.get_legend_handles_labels()
    handles.append(circle_marker)
    ax.legend(handles=handles, loc='upper left')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f"Plot gespeichert unter: {output_file}")
    plt.close()
    return fig,ax

def make_plot_basin_vs_linear_stability(A, basin_stability, linear_stability, A_crit,output_file=None):
    """
    Compares the measurement of linear stability and basin size in one blot
    
    :param A: Aridity
    :param basin_stability: Volume of basin of attraction of forest state
    :param linear_stability: Linear stability measure in depence of A
    :param A_crit: Critical Aridity
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    #Basin Stability
    color = 'tab:green'
    ax1.set_xlabel('Aridity (A)',)
    ax1.set_ylabel('Basin Stability (Safety Volume)')
    ax1.plot(A, basin_stability, color=color, label='Basin Stability')
    ax1.set_ylim(-0.05, 1.05)

    #Linear Stability
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Linear Stability (Recovery Speed)')
    ax2.plot(A, linear_stability, color=color, linestyle='--', label='Linear Stability')
    max_val = np.max(linear_stability)
    top_limit = max(1.05, max_val * 1.1) 
    ax2.set_ylim(-0.05, top_limit)

    ax1.axvline(x=A_crit, color='red', linestyle=':', linewidth=2, label=r'$A_{crit}$ (Tipping Point)')

    plt.title('Linear vs. Basin Stability', fontsize=16)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f"Plot gespeichert unter: {output_file}")
    plt.close()
    return fig, (ax1, ax2)

def make_plot_trajectories(A_range,C0_range,func_C_crit,C_F,t=np.linspace(0,10,5),figsize=None,output_file=None):
    """
    Docstring for make_plot_trajectories
    
    :param A_range: Array of Aridities
    :param C0_range: Array of Forest Covers
    :param func_C_crit: Function that gives the critical cover in dependence of aridity
    :param C_F: Forest Cover Equilibrium
    :param t: time
    """
    fig,axs = plt.subplots(len(A_range),figsize=figsize, sharex=True)
    for A,ax in zip(A_range, axs):
        dcdt_fixedA = partial(dCdt,A=A)
        center = func_C_crit(A)*(func_C_crit(A)<C_F)+(func_C_crit(A)>C_F)
        norm = normation(vmin=0.0,vcenter=center,vmax=1.0)
        cmap3 = cmap2(vmin=0.0,vcenter=center,vmax=1.0)
        for C0 in C0_range:
            C_solved = odeint(dcdt_fixedA, C0, t,)
            ax.plot(t,C_solved,label=rf"$C_0=${C0:.1f}",color=cmap3(norm(C0)),)
        ax.set_ylabel('C(t)')
    axs[-1].set_xlabel('t')
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f"Plot gespeichert unter: {output_file}")
    plt.close()
    return fig,axs

def cmap2(vmin, vmax, vcenter):
    """
    Centered colormap around given value
    """
    if vcenter == vmin:
        return LinearSegmentedColormap.from_list("green_yellow", colors2[:-1]).reversed()
    elif vcenter == vmax:
        return LinearSegmentedColormap.from_list("yellow_red", colors2[1:]).reversed()
    else:
        return LinearSegmentedColormap.from_list("green_yellow_red", colors2).reversed()

def normation(vmin,vcenter,vmax):
    """
    Helper function for colormaps
    """
    if vcenter == vmin or vcenter == vmax:
        return Normalize(vmin=vmin, vmax=vmax)
    else:
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

def expansion_rate(P,r_m=0.3,h_P=0.5):
    """
    Expansion rate of tree cover in dependence of precipitation.
    Described via a saturating function.
    
    :param P: Precipitation
    :param r_m: Maximum expansion rate of tree cover
    :param h_P: The precipitation where the expansion rate is reduced by a half of its maximum
    """
    return P / (h_P+P) * r_m

"""
Second Model, Notation follows the paper

van Nes, E.H., Hirota, M., Holmgren, M. and Scheffer, M. (2014), 
Tipping points in tropical tree cover: linking theory to data. 
Glob Change Biol, 20: 1016-1021. 
https://doi.org/10.1111/gcb.12398
"""

def egbert_tree_cover_derivative(T,P,K=90,m_A=0.15,h_A=10,m_f=0.11,h_f=64,p=7):
    """
    Returns the derivative dT/dt of the Tree cover T with respect to time t.
    
    :param T: Tree cover
    :param P: Precipitation
    :param K: Maximum Tree Covewr
    :param m_A: Maximum loss rate for increased mortality at low tree covers
    :param h_A: Tree cover below which there is an increased mortality due to an Allee effect
    :param m_f: Maximum loss rate due to fire
    :param h_f: Tree cover below which the fire mortality increases steeply
    :param p: Exponent in Hill function for fire effect
    """
    T=np.asarray(T)
    P=np.asarray(P)
    dTdt = expansion_rate(P) * T * (1 - T/K) - m_A * T * h_A / (T+h_A) - m_f * T * h_f**p / (h_f**p+T**p)
    return dTdt

def F_prime_numeric(T,P,K=90,m_A=0.15,h_A=10,m_f=0.11,h_f=64,p=7, h=1e-8):
    """
    Calculates the derivative of dT/dt with respect to T
    Used to find wether roots are stabkle or unstables
    """
    return (egbert_tree_cover_derivative(T=T+h,P=P) - egbert_tree_cover_derivative(T=T-h,P=P)) / (2*h)

def egbert_precipitation_cover_derivative(T,P,r_p=0.01,P_d=3.5,b=0.3,K=90):
    """
    Returns the derivative dP/dt of the Precipitation P with respect to time t.
    
    :param T: Tree Cover
    :param P: Precipiation
    :param r_p: Maximum rate toward equilibrium for precipitation
    :param P_d: Mean annual precipitation over desert
    :param b: The effect strength of vegetation cover on precipitation
    :param K: Maximum tree cover
    """
    T=np.asarray(T)
    P=np.asarray(P)
    dPdt = r_p * ((P_d + b*T/K) - P)
    return dPdt

def odes(y, t):
    """
    Returns both dP/dt and dT/dt 
    
    :param y: Consists of P,T
    :param t: Time (unused)
    """
    p_i,t_i = y
    dPdt = egbert_precipitation_cover_derivative(T=t_i,P=p_i)
    dTdt = egbert_tree_cover_derivative(T=t_i,P=p_i)
    return [dPdt, dTdt]

def findroots(P):
    """
    Returns all stable and unstable equilibria of the forest cover
    
    :param P: Forest Cover
    """
    func=partial(egbert_tree_cover_derivative,P=P)
    interval_count=int(1e2)
    grid=np.linspace(0.0,100.0,interval_count+1)
    stableroots=[]
    unstableroots=[]

    if F_prime_numeric(T=0.0,P=P)<0: stableroots.append(0.0)
    else: unstableroots.append(0.0)

    for i in range(interval_count):
        interval=grid[i:i+2]
        try:
            root = brentq(func,*interval) 
            if (root not in stableroots) and (root not in unstableroots):
                if F_prime_numeric(T=root,P=P)<0: stableroots.append(root)
                else: unstableroots.append(root)   
        except ValueError:
                continue    
    return np.repeat(P,len(stableroots)),stableroots,np.repeat(P,len(unstableroots)),unstableroots

def worker_task(args, odes, gaps):
    """
    Helper function for the simulation from a single starting value
    
    :param args: i,P0,T0 with index i and initial precipitation P0 and initial tree cover T0
    :param odes: Ode to develop tree cover and precipitation
    :param gaps: Gaps in between stable states
    """
    i, P0, T0 = args
    
    # We integrate long enough to converge to a stable state
    sol = odeint(func=odes, y0=[P0, T0], t=np.linspace(0, 1000, 100))
    final_T = sol[-1, 1]
    
    # Code 1: No Tree, Code 2: Savanna, Code 3: Forest
    if final_T < gaps[0]:
        code = 1
    elif final_T < gaps[1]:
        code = 2
    else:
        code = 3
        
    return (i, P0, T0, code)

def compute_ode_gaps_meshgrid_parallel_progress(P_grid, T_grid, odes, pstable, stable, gaps, eps=1e-6):
    """
    Parallelizes the calculation for the whole grid of starting values
    
    :param P_grid: Grid of initial precipitations
    :param T_grid: Grid of initial tree covers
    :param odes: Ode to develop tree cover and precipitation
    :param gaps: Gaps in between stable states
    """
    results = np.zeros(P_grid.shape, dtype=int)
    
    flat = [[P_grid[i,j], T_grid[i,j]]
             for i in range(P_grid.shape[0])
             for j in range(P_grid.shape[1])] 
    flat_results = np.zeros(len(flat), dtype=int)
    tasks = [(i,*flat[i]) for i in range(len(flat))]

    func = partial(worker_task, odes=odes, gaps=gaps)


    with Pool(os.cpu_count()) as pool:
        # Use imap_unordered for faster results and tqdm for progress
        for _, (i,p0,t0,code) in enumerate(tqdm(pool.imap_unordered(func, tasks), total=len(tasks), desc="Volume Estimation")):
            flat_results[i] = code
    results=flat_results.reshape(P_grid.shape)
    return results

def compute_values_in_gaps(arr,n_gaps):
    """
    Takes the n_gaps largest gaps of the array and returns the median value of each gap
    
    :param arr: 1 dimensional Array
    :param n_gaps: Number of gaps to be considered
    """
    
    sort=np.sort(arr)
    diff=np.diff(sort)
    largest_gap_indices = np.argsort(diff)[-n_gaps:]  # indices of the two largest gaps
    values_in_gaps = np.sort([(sort[i] + sort[i+1])/2 for i in largest_gap_indices])

    return values_in_gaps

def make_plot_egbert(P,T,func_cover_derivative,func_precipitation_derivative,xlabel=None,ylabel=None,title=None,figsize=None,output_file=None):
    """
    Creates the same plot as for the initial model with all calculations needed in this model
    Adds the bifurcation plot with color coding for stable/unstable regions
    
    :param P: Precipitation
    :param T: Tree Cover
    :param func_cover_derivative: Tree Cover derivative wrt time
    :param func_precipitation_derivative: Precipitation derivative wrt time
    """
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)    
    
    #Creates all used meshgrids
    PP, TT = np.meshgrid(P, T)
    refinment=30
    PP_fine, TT_fine = np.meshgrid(np.linspace(P[0],P[-1],len(P)*refinment), np.linspace(T[0],T[-1],len(T)*refinment))
   
    #Calulation of stable/unstable equilibria
    inputs=np.linspace(P[0],P[-1],len(P)*refinment)  
    with Pool(os.cpu_count()) as pool:  # automatically uses all cores
        results = tqdm(pool.map(findroots, inputs), total=len(inputs), desc="Finding Stable States")
    pstable,stable,punstable,unstable=[],[],[],[]
    for result in results:
        ps,s,pu,u=result
        pstable.extend(ps)
        stable.extend(s)
        punstable.extend(pu)
        unstable.extend(u)
    stable=np.array(stable)
    pstable=np.array(pstable)

    #Plot of basins of attraction with color
    values_in_gaps=compute_values_in_gaps(arr=stable,n_gaps=2)
    eps = 1e-1
    Basin=compute_ode_gaps_meshgrid_parallel_progress(PP_fine,TT_fine,odes=odes,pstable=pstable,stable=stable,gaps=values_in_gaps, eps=eps)
    colors = ['black', "#bc2d05", "#ff9900", "#006635"]   
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = [0,1,2,3,4]  
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.pcolormesh(PP_fine, TT_fine, Basin, cmap=cmap, norm=norm, shading='auto', alpha=0.8)
    
    #Arrows to show direction and speed of convergence
    Direction_T=func_cover_derivative(T=TT,P=PP)
    Direction_P=func_precipitation_derivative(T=TT,P=PP)
    ax.quiver(PP,TT, Direction_P/4.5, Direction_T/100,color='white', alpha=1.0, pivot='mid',)       
    
    ax.set_xlabel(r'Precipitation $(mm d^{-1})$')
    ax.set_ylabel(r'Tree Cover')

    ax.scatter(pstable,stable,color='black',s=20,zorder=100)
    ax.scatter(punstable,unstable,color='white',s=20,zorder=101)

    
    legend_handles = [
        matplotlib.patches.Patch(color=cmap.colors[1], label="No Tree State"),
        matplotlib.patches.Patch(color=cmap.colors[2], label="Savanna State"),
        matplotlib.patches.Patch(color=cmap.colors[3], label="Forest State")
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.8)

    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f"Plot gespeichert unter: {output_file}") 
    
    return fig,ax

def make_plot_monte_carlo(N_samples, bounds_P, bounds_T, odes, gaps, output_file=None, figsize=(12, 6)):
    """
    Führt die MC-Simulation aus und erstellt den Plot (Karte + Statistik).
    
    :param N_samples: Number of initial positions to be sampled
    :param bounds_P: Range in which initial precipitations can be chosen
    :param bounds_T: Range in which initial tree covers can be chosen
    :param odes: Function that returns the time derivatives of precipitation and tree cover
    :param gaps: Values that inbetween the stable value ranges for tree cover
    :param output_file: File to save plot to
    :param figsize: Size of final plot
    
    """
    print(f"Starte Monte Carlo Simulation mit {N_samples} Punkten...")

    # Generate starting values
    random_P = np.random.uniform(bounds_P[0], bounds_P[1], N_samples)
    random_T = np.random.uniform(bounds_T[0], bounds_T[1], N_samples)
    
    # Calculate results
    tasks = [(i, random_P[i], random_T[i]) for i in range(N_samples)]
    func = partial(worker_task, odes=odes, gaps=gaps)
    results = []
    with Pool(os.cpu_count()) as pool:
        # tqdm zeigt einen Ladebalken
        for (_,*res) in tqdm(pool.imap_unordered(func, tasks), total=N_samples, desc="MC Sampling"):
            results.append(res)
            
    # Unpack results
    res_P = np.array([r[0] for r in results])
    res_T = np.array([r[1] for r in results])
    codes = np.array([r[2] for r in results])
    
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    mask_notree = (codes == 1)
    mask_savanna = (codes == 2)
    mask_forest = (codes == 3)
    
    # Color coded basins of attraction
    ax_map.scatter(res_P[mask_notree], res_T[mask_notree], c='black', s=10, alpha=0.5, label='No Tree')
    ax_map.scatter(res_P[mask_savanna], res_T[mask_savanna], c='orange', s=10, alpha=0.5, label='Savanna')
    ax_map.scatter(res_P[mask_forest], res_T[mask_forest], c='forestgreen', s=10, alpha=0.5, label='Forest')
    
    ax_map.set_xlabel(r'Precipitation $P$ ($mm d^{-1}$)')
    ax_map.set_ylabel(r'Tree Cover $T$')
    ax_map.set_title(f'Monte Carlo Estimation (N={N_samples})')
    ax_map.set_xlim(bounds_P)
    ax_map.set_ylim(bounds_T)
    ax_map.legend(loc='upper left')

    # Statistical results
    N = len(codes)
    fractions = [np.sum(mask_notree)/N, np.sum(mask_savanna)/N, np.sum(mask_forest)/N]
    
    errors = [np.sqrt(f*(1-f)/N) for f in fractions]
    
    labels = ['No Tree', 'Savanna', 'Forest']
    colors = ['black', 'orange', 'forestgreen']
    
    bars = ax_bar.bar(labels, fractions, yerr=errors, capsize=5, color=colors, alpha=0.8)
    
    for bar, frac in zip(bars, fractions):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                    f"{frac*100:.1f}%", ha='center', fontweight='bold')
    
    ax_bar.set_ylim(0, 1.1) 
    ax_bar.set_ylabel(r'Basin Stability ($S_B$)')
    ax_bar.set_title('Quantified Resilience')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=300)
        print(f"Plot gespeichert unter: {output_file}")
    plt.close()
    return fig

# -------------------------------
# Parameters
# -------------------------------
resolution = int(1e3)
resolution_sparse=20

# First Model
x = 0.4
r = 1.5
m=0.8
b=-0.1

A_min, A_max = 0, 1.3
C_min, C_max = 0, 1.0

A = np.linspace(A_min, A_max, resolution)
C = np.linspace(C_min, C_max, resolution)


A_range=np.linspace(0,calc_A_crit(x,r,m,b)*1.2,7)
A_crit_value=calc_A_crit(x, r, m, b)

# Second Model
P_min, P_max = 0.5, 5.0
T_min, T_max = 0, 100.2

P = np.linspace(P_min, P_max, resolution_sparse*2)
T = np.linspace(T_min, T_max, resolution_sparse*2) 


# -------------------------------
# Plot settings
# -------------------------------
figsize = (8, 6)
cmap = ListedColormap(["#ff9900","#006635"])
colors=[
    "#115a00",
    "#26a200",
    "#0fb300",
    "#44ff00",
    "#eaff00",
    "#ff9900",
    "#bc2d05",
    ]
colors2=[
    "#115a00",
    "#eaff00",
    "#bc2d05",
    ]

if __name__ == '__main__':   
    make_plot_noarrows(
        A=A, C=C,
        func_cover=partial(forest_cover,x=x,r=r,m=m,b=b),
        func_crit=partial(calc_C_crit,m=m,b=b),
        C_F=calc_C_F(x=x,r=r),
        cmap=cmap,
        xlabel='Aridity (A)',
        ylabel='Forest Cover (C)',
        title='Forest-Savanna Model',
        figsize=figsize,
        output_file="results/noarrows.png"
    )

    make_plot_arrows(
        A=A, C=C,
        func_cover=partial(forest_cover,x=x,r=r,m=m,b=b),
        func_cover_derivative=partial(forest_cover_derivative, x=x, r=r),
        func_crit=partial(calc_C_crit,m=m,b=b),
        C_F=calc_C_F(x=x,r=r),
        cmap=cmap,
        xlabel='Aridity (A)',
        ylabel='Forest Cover (C)',
        title='Forest-Savanna Model',
        figsize=figsize,
        output_file="results/arrows.png"
    )

    make_plot_potential(
        A_range=A_range,
        C=C,
        colors=colors,
        func_crit=partial(calc_C_crit,m=m,b=b),
        func_potential=partial(calc_U,x=x,r=r),
        A_crit=calc_A_crit(x=x,r=r,m=m,b=b),
        C_F=calc_C_F(x=x,r=r),
        xlabel='Forest Cover (C)',
        ylabel='Potential (U)',
        title='Forest-Savanna Model',
        figsize=figsize,
        output_file="results/potential.png"
    )

    make_plot_basin_vs_linear_stability(
        A=A,
        #C_F=calc_C_F(x=x,r=r),
        A_crit=calc_A_crit(x=x,r=r,m=m,b=b),
        basin_stability=calc_basin_stability(A=A, A_crit=calc_A_crit(x=x,r=r,m=m,b=b), C_crit=calc_C_crit(A=A, m=m, b=b)),
        linear_stability=calc_linear_stability(x=x, r=r, C_crit=calc_C_crit(A=A, m=m, b=b), C=calc_C_F(x=x,r=r)),
        #colors=colors,
        #title=title,
        #figsize=figsize,
        output_file="results/comparison.png"
    )

    make_plot_trajectories(
        A_range=np.linspace(A_min,A_max,5),
        C0_range=np.linspace(0,1,11),
        func_C_crit=partial(calc_C_crit,m=m,b=b),
        C_F=calc_C_F(x=x,r=r),
        t=np.linspace(0, 7, 200),
        figsize=(6,12),
        output_file="results/trajectories.png"
    )

    make_plot_egbert(
        P=P, T=T,
        func_cover_derivative=egbert_tree_cover_derivative,
        func_precipitation_derivative=egbert_precipitation_cover_derivative,
        figsize=(10,8),
        output_file='results/grid_scan.png',
    )
    
    mc_gaps = [11.5, 61.7] 
    make_plot_monte_carlo(
        N_samples=int(1e4),           # Anzahl der Punkte (je mehr, desto genauer)
        bounds_P=(0.5, 5.0),      # Bereich für Niederschlag
        bounds_T=(0, 100),        # Bereich für Baumdichte
        odes=odes,                # Deine existierende odes-Funktion
        gaps=mc_gaps,             # Die Schwellenwerte
        output_file="results/monte_carlo_resilience.png"
    )
    
