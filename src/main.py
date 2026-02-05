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

def forest_cover_derivative(C_crit, x, r, C, A):
    """Compute derivative of forest cover."""
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = C > C_crit
    result[~mask] = -x * C[~mask]
    result[mask] = r * (1 - C[mask]) * C[mask] - x * C[mask]

    return result

def forest_cover(C_crit, x, r, m, b, C, A):
    """Forest cover state function."""
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)

    mask = (1-x/r > calc_C_crit(A,m,b)) & (C>calc_C_crit(A,m,b))
    result[~mask] = -1
    result[mask] = 1

    return result

def calc_C_crit(A,m,b):
    """Critical forest cover threshold."""
    return np.maximum(A*m+b,0)

def calc_C_F(x,r):
    return(1-x/r)

def calc_A_crit(x,r,m,b):
    return (calc_C_F(x,r)-b)/m

def calc_U(x,r,C_crit,C):
    return x/2*C**2+ (C>C_crit)*(r/3 * (C**3-C_crit**3) - r/2 *(C**2-C_crit**2))

def calc_linear_stability(x,r,C_crit,C):
    """dF(C)/dC derivative of dC/dt with C_F=1-x/r, result is eigenvalue, because of r>x>0 is eigenvalue<0 ->stability, always comes back to initial state, result is how fast it goes back
    use positive value for x-r ->abs(x-r)"""
    return(C>C_crit)*(abs(x-r))

def calc_basin_stability(A,A_crit,C_crit):
    """basin=stability volume c_crit to 1 (100% coverage), A<A_crit because basin exists only if forest is stable"""
    return(A<A_crit)*(1-C_crit)
    
def dcdt(C,t,A):
    return forest_cover_derivative(C_crit=calc_C_crit(A=A,m=m,b=b),x=x,r=r,C=C,A=A)

def make_plot_noarrows(A,C,func_cover,func_crit,C_F,cmap=ListedColormap(['lightgray','grey']),xlabel=None,ylabel=None,title=None,figsize=None):
        
    AA, CC = np.meshgrid(A, C)
    C_crit=func_crit(A)
    # Berechne den Tipping Point Index
    idx_crit = np.argmin(abs(C_crit-C_F))
    A_crit = A[idx_crit]
    
    State=func_cover(C_crit=C_crit,C=CC,A=AA)

    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(AA, CC,State, shading='auto', cmap=cmap)
    
    # 1. Die Linie Plotten (ohne Label für Legende)
    ax.plot(A, C_crit, color='white', linewidth=2)
    
    # 2. Text DIREKT auf die Linie schreiben
    # Wir suchen einen Punkt ca. in der Mitte zwischen 0 und A_crit
    text_idx = int(idx_crit * 0.5) 
    text_x = A[text_idx]
    text_y = C_crit[text_idx]
    
    # Text einfügen: rotation passt den Textwinkel an die Linie an (ggf. anpassen)
    ax.text(text_x, text_y + 0.02, r'$C_{crit}(A)$', 
            color='white', 
            fontsize=12, 
            fontweight='bold', 
            rotation=28, # Winkel anpassen, falls es bei dir zu steil/flach ist
            ha='center', 
            va='bottom')

    # Hilfslinien
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

    # Legende nur noch für die Zustände (Farben)
    legend_handles = [
        matplotlib.patches.Patch(color=cmap.colors[0], label="Savanna State"),
        matplotlib.patches.Patch(color=cmap.colors[1], label="Forest State")
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.8)
    fig.tight_layout()
    
    return fig,ax

def make_plot_arrows(A,C,func_cover,func_cover_derivative,func_crit,C_F,resolution_sparse=20,cmap=ListedColormap(['lightgray','grey']),xlabel=None,ylabel=None,title=None,figsize=None):
        
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
    ax.quiver(AA_sparse,CC_sparse, 0*np.ones_like(AA_sparse), Direction_sparse)

    return fig,ax

def make_plot_potential(A_range,C,func_crit,func_potential,A_crit,C_F,colors,xlabel=None,ylabel=None,title=None,figsize=None,):

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

    return fig,ax

def make_plot_basin_vs_linear_stability(A, basin_stability, linear_stability, A_crit):
    """comparison of basin and linear stability with 2 y-axes"""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Axis 1 (left): Basin Stability (green)
    color = 'tab:green'
    ax1.set_xlabel('Aridity (A)',)
    ax1.set_ylabel('Basin Stability (Safety Volume)')
    ax1.plot(A, basin_stability, color=color, label='Basin Stability')
    # ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)

    # Axis 2 (right): Linear Stability (blue)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Linear Stability (Recovery Speed)')
    ax2.plot(A, linear_stability, color=color, linestyle='--', label='Linear Stability')
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim(-0.05, 1.05) 
    max_val = np.max(linear_stability)
    top_limit = max(1.05, max_val * 1.1) 
    ax2.set_ylim(-0.05, top_limit)

    # Markierungen
    ax1.axvline(x=A_crit, color='red', linestyle=':', linewidth=2, label=r'$A_{crit}$ (Tipping Point)')
    # ax1.text(A_crit, -0.05, r'$A_{crit}$ (Tipping Point)', color='red', ha='center', va='top', 
    #       transform=ax1.get_xaxis_transform(), fontsize=12)

    # # --- DYNAMISCHE TEXT-PLATZIERUNG ---
    # # 1. Linear Stability Text (Blau)
    # y_linear = linear_stability[0]
    # ax2.text(A[int(len(A)*0.3)], y_linear + 0.05, 
    #          'Recovery Speed constant', 
    #          color='tab:blue', fontweight='bold', fontsize=12, ha='left')

    # # 2. Basin Stability Text (Grün) - DEUTLICH TIEFER
    # # Wir nehmen einen Punkt ziemlich am Anfang (15% des Weges)
    # idx_pos = np.argmin(np.abs(A - (A_crit * 0.15))) 
    # x_basin = A[idx_pos]
    # y_basin = basin_stability[idx_pos]
    
    # # Das schiebt den Text weiter in den "freien Raum" unter der Kurve
    # ax1.text(x_basin, y_basin - 0.4, 
    #          'Safety Volume decreases!', 
    #          color='tab:green', fontweight='bold', fontsize=12, 
    #          ha='left', va='top') 

    # Titel und Layout
    plt.title('Linear vs. Basin Stability', fontsize=16)
    # ax1.grid(True, linestyle='--', alpha=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,)
    fig.tight_layout()
    
    return fig, (ax1, ax2)

def make_plot_trajectories(A_range,C0_range,func_C_crit,C_F,t=np.linspace(0,10,5),figsize=None):
    fig,axs = plt.subplots(len(A_range),figsize=figsize, sharex=True)
    for A,ax in zip(A_range, axs):
        dcdt_fixedA = partial(dcdt,A=A)
        center = func_C_crit(A)*(func_C_crit(A)<C_F)+(func_C_crit(A)>C_F)
        norm = normation(vmin=0.0,vcenter=center,vmax=1.0)
        cmap3 = cmap2(vmin=0.0,vcenter=center,vmax=1.0)
        for C0 in C0_range:
            C_solved = odeint(dcdt_fixedA, C0, t,)
            ax.plot(t,C_solved,label=rf"$C_0=${C0:.1f}",color=cmap3(norm(C0)),)
        ax.set_ylabel('C(t)')
    axs[-1].set_xlabel('t')
    fig.tight_layout()
    return fig,axs


# -------------------------------
# Parameters
# -------------------------------
resolution = int(1e3)
resolution_sparse=20
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

def cmap2(vmin, vmax, vcenter):
    if vcenter == vmin:
        return LinearSegmentedColormap.from_list("green_yellow", colors2[:-1]).reversed()
    elif vcenter == vmax:
        return LinearSegmentedColormap.from_list("yellow_red", colors2[1:]).reversed()
    else:
        return LinearSegmentedColormap.from_list("green_yellow_red", colors2).reversed()

def normation(vmin,vcenter,vmax):
    if vcenter == vmin or vcenter == vmax:
        return Normalize(vmin=vmin, vmax=vmax)
    else:
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


output_file = "results/noarrows.png"
output_file_arrows = "results/arrows.png"
output_file_potential = "results/potential.png"
output_file_comparison = "results/comparison.png"
output_file_trajectories = "results/trajectories.png"

xlabel = "Aridity (A)"
ylabel = "Forest Cover (C)"
title = "Forest-Savanna Model"






# -------------------------------
# Plotting
# -------------------------------
fig,_=make_plot_noarrows(A=A, C=C,
                         func_cover=partial(forest_cover,x=x,r=r,m=m,b=b),
                         func_crit=partial(calc_C_crit,m=m,b=b),
                         C_F=calc_C_F(x=x,r=r),
                         cmap=cmap,
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=title,
                         figsize=figsize,
                        )
fig.savefig(output_file, dpi=300)
plt.close()

fig,_=make_plot_arrows(A=A, C=C,
                       func_cover=partial(forest_cover,x=x,r=r,m=m,b=b),
                       func_cover_derivative=partial(forest_cover_derivative, x=x, r=r),
                       func_crit=partial(calc_C_crit,m=m,b=b),
                       C_F=calc_C_F(x=x,r=r),
                       cmap=cmap,
                       xlabel=xlabel,
                       ylabel=ylabel,
                       title=title,
                       figsize=figsize,
                       )
fig.savefig(output_file_arrows, dpi=300)
plt.close()

fig,_ = make_plot_potential(A_range=A_range,
                            C=C,
                            colors=colors,
                            func_crit=partial(calc_C_crit,m=m,b=b),
                            func_potential=partial(calc_U,x=x,r=r),
                            A_crit=calc_A_crit(x=x,r=r,m=m,b=b),
                            C_F=calc_C_F(x=x,r=r),
                            xlabel='Forest Cover (C)',
                            ylabel='Potential (U)',
                            title=title,
                            figsize=figsize,
                            )
fig.savefig(output_file_potential, dpi=300)
plt.close()

fig,_ = make_plot_basin_vs_linear_stability(A=A,
                                            #C_F=calc_C_F(x=x,r=r),
                                            A_crit=calc_A_crit(x=x,r=r,m=m,b=b),
                                            basin_stability=calc_basin_stability(A=A, A_crit=calc_A_crit(x=x,r=r,m=m,b=b), C_crit=calc_C_crit(A=A, m=m, b=b)),
                                            linear_stability=calc_linear_stability(x=x, r=r, C_crit=calc_C_crit(A=A, m=m, b=b), C=calc_C_F(x=x,r=r)),
                                            #colors=colors,
                                            #title=title,
                                            #figsize=figsize,
                                            )
fig.savefig(output_file_comparison, dpi=300)
plt.close()

fig,_ = make_plot_trajectories(A_range=np.linspace(A_min,A_max,5),
                               C0_range=np.linspace(0,1,11),
                               func_C_crit=partial(calc_C_crit,m=m,b=b),
                               C_F=calc_C_F(x=x,r=r),
                               t=np.linspace(0, 7, 200),
                               figsize=(6,12),
                               )
fig.savefig(output_file_trajectories, dpi=300)
plt.close()




def r(P,r_m=0.3,h_P=0.5):
    return P / (h_P+P) * r_m


def egbert_tree_cover_derivative(T,P,K=90,m_A=0.15,h_A=10,m_f=0.11,h_f=64,p=7):
    T=np.asarray(T)
    P=np.asarray(P)
    # dTdt = r(P) * T * (1 - T/K) - m_A * T * h_A / (T+h_A)
    dTdt = r(P) * T * (1 - T/K) - m_A * T * h_A / (T+h_A) - m_f * T * h_f**p / (h_f**p+T**p)
    return dTdt

def egbert_tree_cover_mixed_derivative(T,P,K=90,m_A=0.15,h_A=10,m_f=0.11,h_f=64,p=7):
    T=np.asarray(T)
    P=np.asarray(P)
    # dTdt = r(P) * T * (1 - T/K) - m_A * T * h_A / (T+h_A)
    dTdt = (r(P) * (1 - T/K) 
            - m_A * h_A / (T+h_A) * (1 - T/(T+h_A))
            - m_f  * h_f**p / (h_f**p+T**p)) * (1 - p*T**p/(h_f**p+T**p))
    return dTdt

def F_prime_numeric(T,P,K=90,m_A=0.15,h_A=10,m_f=0.11,h_f=64,p=7, h=1e-8):
    return (egbert_tree_cover_derivative(T=T+h,P=P) - egbert_tree_cover_derivative(T=T-h,P=P)) / (2*h)

def egbert_precipitation_cover_derivative(T,P,r_p=0.01,P_d=3.5,b=0.3,K=90):
    T=np.asarray(T)
    P=np.asarray(P)
    dPdt = r_p * ((P_d + b*T/K) - P)
    return dPdt

def odes(y, t):
    p_i,t_i = y
    dPdt = egbert_precipitation_cover_derivative(T=t_i,P=p_i)
    dTdt = egbert_tree_cover_derivative(T=t_i,P=p_i)
    return [dPdt, dTdt]

def remove_epsilon_from_interval(intervals, c, eps):
    """
    Remove the ball [c-eps, c+eps] from interval [a,b].
    Returns a list of resulting intervals (1 or 2 intervals).
    """
    intervals_out = []
    for interval in intervals:
        a,b=interval
        left = max(a, c - eps)
        right = min(b, c + eps)

        

        # Left part
        if left > a:
            intervals_out.append((a, left))

        # Right part
        if right < b:
            intervals_out.append((right, b))

    return intervals_out

def oldfindroots(p,func_cover_derivative):
    width=0.001
    intervalls=[(0.0,1.0)]
    roots=[]
    intervalls=remove_epsilon_from_interval(intervalls,roots[-1],width)
    
    while intervalls:
        interval=intervalls.pop(0)
        try:
            root = brentq(func_cover_derivative,*interval)
            if root not in roots:
                roots.append(root)
                intervalls.extend(remove_epsilon_from_interval([interval],roots[-1],width))
            print(intervalls)
        except ValueError:
            continue     
    return (np.repeat(p,len(roots)),roots)

def findroots(p):
    func=partial(egbert_tree_cover_derivative,P=p)
    interval_count=int(1e2)
    grid=np.linspace(0.0,100.0,interval_count+1)
    stableroots=[]
    unstableroots=[]

    if F_prime_numeric(T=0.0,P=p)<0: stableroots.append(0.0)
    else: unstableroots.append(0.0)

    for i in range(interval_count):
        interval=grid[i:i+2]
        try:
            root = brentq(func,*interval)
            if (root not in stableroots) and (root not in unstableroots):
                if F_prime_numeric(T=root,P=p)<0: stableroots.append(root)
                else: unstableroots.append(root)   
        except ValueError:
                continue    
    return np.repeat(p,len(stableroots)),stableroots,np.repeat(p,len(unstableroots)),unstableroots

def compute_ode_gaps_meshgrid(P_grid, T_grid, odes, pstable, stable, gaps, eps=1e-6):
    results = np.empty(P_grid.shape, dtype=int)
    for i in range(P_grid.shape[0]):
        for j in range(P_grid.shape[1]):
            p0 = P_grid[i,j]
            t0 = T_grid[i,j]
            sol = odeint(func=odes, y0=[p0,t0], t=np.linspace(0,10000,100))
            mask = (pstable >= sol[-1, 0] - eps) & (pstable <= sol[-1, 0] + eps)
            if np.any(np.abs(stable[mask] - sol[-1, 1]) <= eps):
                if sol[-1,1]<gaps[0]:
                    results[i,j]=1
                elif sol[-1,1]<gaps[1]:
                    results[i,j]=2
                else:
                    results[i,j]=3
                    
            else: results[i,j]=0
    return results

def worker_task(args, odes, gaps):
    """
    Hilfsfunktion für einen einzelnen Simulations-Schritt (für Parallelisierung).
    Nutzt einfache Schwellenwerte (Gaps) zur Klassifizierung.
    """
    i, P0, T0 = args
    
    # Simulation: Wir integrieren lange genug (t=1000), um das Gleichgewicht zu finden
    sol = odeint(func=odes, y0=[P0, T0], t=np.linspace(0, 1000, 100))
    final_T = sol[-1, 1]
    
    # Klassifizierung anhand der Gaps
    # Code 1: No Tree, Code 2: Savanna, Code 3: Forest
    if final_T < gaps[0]:
        code = 1
    elif final_T < gaps[1]:
        code = 2
    else:
        code = 3
        
    return (i, P0, T0, code)

# def compute_ode_gaps_meshgrid_parallel(P_grid, T_grid, odes, pstable, stable, gaps, eps=1e-6):
#     results = np.zeros(P_grid.shape, dtype=int)
    
#     # Prepare tasks
#     tasks = [(i, j, P_grid[i,j], T_grid[i,j])
#              for i in range(P_grid.shape[0])
#              for j in range(P_grid.shape[1])]
    
#     tasks = [(i, random_P[i], random_T[i]) for i in range(N_samples)]

#     # Partial to pass extra args
#     func = partial(worker_task, odes=odes, gaps=gaps)

#     # Run in parallel
#     with Pool(os.cpu_count()) as pool:
#         outputs = pool.map(func, tasks)

#     # Fill results
#     for i, j, code in outputs:
#         results[i,j] = code

#     return results

def compute_ode_gaps_meshgrid_parallel_progress(P_grid, T_grid, odes, pstable, stable, gaps, eps=1e-6):
    results = np.zeros(P_grid.shape, dtype=int)
    
    flat = [[P_grid[i,j], T_grid[i,j]]
             for i in range(P_grid.shape[0])
             for j in range(P_grid.shape[1])] 
    flat_results = np.zeros(len(flat), dtype=int)
    tasks = [(i,*flat[i]) for i in range(len(flat))]
    print(tasks[0])

    func = partial(worker_task, odes=odes, gaps=gaps)
    with Pool(os.cpu_count()) as pool:
        # Use imap_unordered for faster results and tqdm for progress
        for _, (i,p0,t0,code) in enumerate(tqdm(pool.imap_unordered(func, tasks), total=len(tasks))):
            flat_results[i] = code
    results=flat_results.reshape(P_grid.shape)
    return results

def make_plot_egbert(P,T,func_cover_derivative,func_precipitation_derivative,xlabel=None,ylabel=None,title=None,figsize=None):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)    
    
    PP, TT = np.meshgrid(P, T)
    Direction_T=func_cover_derivative(T=TT,P=PP)
    Direction_P=func_precipitation_derivative(T=TT,P=PP)
    # print(func_cover_derivative(T=60,P=0.6),func_precipitation_derivative(T=60,P=0.6))
    refinment=30
    PP_fine, TT_fine = np.meshgrid(np.linspace(P[0],P[-1],len(P)*refinment), np.linspace(T[0],T[-1],len(T)*refinment))
    Direction_T_fine=func_cover_derivative(T=TT_fine,P=PP_fine)

    # ax.pcolormesh(PP_fine, TT_fine, np.sign(Direction_T_fine), shading='auto', cmap='coolwarm_r', alpha=1)
    
    # ax.quiver(PP,TT, Direction_P, Direction_T*0,color='blue', alpha=0.3)
    # ax.quiver(PP,TT, Direction_P*0, (Direction_T),color='white', alpha=1.0)
    inputs=np.linspace(P[0],P[-1],len(P)*refinment)
    with Pool(os.cpu_count()) as pool:  # automatically uses all cores
        results = pool.map(findroots, inputs)
    pstable,stable,punstable,unstable=[],[],[],[]
    for result in results:
        ps,s,pu,u=result
        pstable.extend(ps)
        stable.extend(s)
        punstable.extend(pu)
        unstable.extend(u)
    
    stable = np.array(stable)
    pstable = np.array(pstable)
    stable_sort=np.sort(stable)
    stable_diff=np.diff(stable_sort)
    largest_gap_indices = np.argsort(stable_diff)[-2:]  # indices of the two largest gaps
    values_in_gaps = np.sort([(stable_sort[i] + stable_sort[i+1])/2 for i in largest_gap_indices])
    eps = 1e-1
    Basin=compute_ode_gaps_meshgrid_parallel_progress(PP_fine,TT_fine,odes=odes,pstable=pstable,stable=stable,gaps=values_in_gaps, eps=eps)
    colors = ['black', "#fa7970", "#faa356", "#77bdfb"]   # 'fa7970','77bdfb','7ce38b','cea5fb','faa356'
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = [0,1,2,3,4]  
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.pcolormesh(PP_fine, TT_fine, Basin, cmap=cmap, norm=norm, shading='auto', alpha=0.8)
    
    ax.quiver(PP,TT, Direction_P/4.5, Direction_T/100,color='white', alpha=1.0, pivot='mid',)       
    
    ax.set_xlabel(r'Precipitation $(mm d^{-1})$')
    ax.set_ylabel(r'Tree Cover')
    for p0 in np.linspace(0.5,5,40):
        if (p0==0.5) or (p0==5):
            ts=np.linspace(1,100,30)
        else:
            ts=[2,100]
        # print(p0)
        for t0 in ts:
                sol = odeint(func=odes, y0=[p0,t0], t=np.linspace(0,1000,10000))
                ax.plot(sol[:,0],sol[:,1],color='white')     
            # ax.scatter(p0,t0) 

    ax.scatter(pstable,stable,color='black',s=20,zorder=100)
    ax.scatter(punstable,unstable,color='white',s=20,zorder=101)

    
    legend_handles = [
        matplotlib.patches.Patch(color=cmap.colors[1], label="No Tree State"),
        matplotlib.patches.Patch(color=cmap.colors[2], label="Savanna State"),
        matplotlib.patches.Patch(color=cmap.colors[3], label="Forest State")
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.8)
    
   
    
    return fig,ax

P_min, P_max = 0.5, 5.0
T_min, T_max = 0, 100.2

P = np.linspace(P_min, P_max, resolution_sparse*2)
T = np.linspace(T_min, T_max, resolution_sparse*2) 
if __name__ == '__main__':
    #fig.clear()
    fig,_ = make_plot_egbert(P=P, T=T,
                        func_cover_derivative=egbert_tree_cover_derivative,
                        func_precipitation_derivative=egbert_precipitation_cover_derivative,
                        figsize=(10,8),
                        )
    fig.savefig('results/grid_scan.png', dpi=300)
    plt.close()

# MONTE CARLO #
# -----------------------------------------------------------

def make_plot_monte_carlo(N_samples, bounds_P, bounds_T, odes, gaps, output_file=None, figsize=(12, 6)):
    """
    Führt die MC-Simulation aus und erstellt den Plot (Karte + Statistik).
    """
    print(f"Starte Monte Carlo Simulation mit {N_samples} Punkten...")
    
    # 1. Zufällige Startpunkte generieren
    # random_P = np.random.normal(loc=(bounds_P[0]+bounds_P[1])/2, scale=0.8, size=N_samples)
    # random_T = np.random.normal(loc=(bounds_T[0]+bounds_T[1])/2, scale=15, size=N_samples)

    random_P = np.random.uniform(bounds_P[0], bounds_P[1], N_samples)
    random_T = np.random.uniform(bounds_T[0], bounds_T[1], N_samples)
    
    tasks = [(i, random_P[i], random_T[i]) for i in range(N_samples)]
    
    # 2. Parallel rechnen
    # Wir übergeben 'odes' und 'gaps' mit partial
    func = partial(worker_task, odes=odes, gaps=gaps)
    
    results = []
    # Nutzt alle verfügbaren Kerne
    with Pool(os.cpu_count()) as pool:
        # tqdm zeigt einen Ladebalken, falls gewünscht
        for (_,*res) in tqdm(pool.imap_unordered(func, tasks), total=N_samples, desc="MC Sampling"):
            results.append(res)
            
    # Ergebnisse entpacken
    res_P = np.array([r[0] for r in results])
    res_T = np.array([r[1] for r in results])
    codes = np.array([r[2] for r in results])
    
    # 3. Plotten
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    
    # Masken für die Farben
    mask_notree = (codes == 1)
    mask_savanna = (codes == 2)
    mask_forest = (codes == 3)
    
    # A) Linker Plot: Die Scatter-Karte
    # alpha verringert, damit man Überlappungen sieht
    ax_map.scatter(res_P[mask_notree], res_T[mask_notree], c='black', s=10, alpha=0.5, label='No Tree')
    ax_map.scatter(res_P[mask_savanna], res_T[mask_savanna], c='orange', s=10, alpha=0.5, label='Savanna')
    ax_map.scatter(res_P[mask_forest], res_T[mask_forest], c='forestgreen', s=10, alpha=0.5, label='Forest')
    
    ax_map.set_xlabel(r'Precipitation $P$ ($mm d^{-1}$)')
    ax_map.set_ylabel(r'Tree Cover $T$')
    ax_map.set_title(f'Monte Carlo Estimation (N={N_samples})')
    ax_map.set_xlim(bounds_P)
    ax_map.set_ylim(bounds_T)
    ax_map.legend(loc='upper left')

    # B) Rechter Plot: Die Statistik (Balken)
    N = len(codes)
    fractions = [np.sum(mask_notree)/N, np.sum(mask_savanna)/N, np.sum(mask_forest)/N]
    
    # Standardfehler: sqrt(p*(1-p)/N)
    errors = [np.sqrt(f*(1-f)/N) for f in fractions]
    
    labels = ['No Tree', 'Savanna', 'Forest']
    colors = ['black', 'orange', 'forestgreen']
    
    bars = ax_bar.bar(labels, fractions, yerr=errors, capsize=5, color=colors, alpha=0.8)
    
    # Prozentzahlen beschriften
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
    
    return fig
if __name__ == '__main__':
      
    mc_gaps = [11.5, 61.7] 
    
    
    make_plot_monte_carlo(
        N_samples=int(1e4),           # Anzahl der Punkte (je mehr, desto genauer)
        bounds_P=(0.5, 5.0),      # Bereich für Niederschlag
        bounds_T=(0, 100),        # Bereich für Baumdichte
        odes=odes,                # Deine existierende odes-Funktion
        gaps=mc_gaps,             # Die Schwellenwerte
        output_file="results/monte_carlo_resilience.png"
    )
    
