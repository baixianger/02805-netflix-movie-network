import os
import powerlaw
import numpy as np
import pandas as pd 
import networkx as nx
from networkx.algorithms import bipartite
import igraph as ig
from functools import reduce
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import requests
from imdb import Cinemagoer
from imdb import IMDbDataAccessError
from bs4 import BeautifulSoup
import matplotlib.animation as animation
from IPython import display
from tqdm.notebook import tqdm
from ndforceatlas import ndforceatlas
from fa2 import ForceAtlas2
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import permutation_test, ttest_ind, f
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


class fs:
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'
class fc:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'
class bg:
    black = '\033[40m'
    red = '\033[41m'
    green = '\033[42m'
    orange = '\033[43m'
    blue = '\033[44m'
    purple = '\033[45m'
    cyan = '\033[46m'
    lightgrey = '\033[47m'

dict2np = lambda ddict: np.array([(k, v)for k, v in ddict.items()])

def build_bipartite(movie_path, actor_path, link_path, save_path=None, tag=None):
    """
    The function is use to build the bipartite movie-actor network from given node and link list.
    For igraph tool, we need to translate the string node names to unique integers.
    """
    
    m_nodes = pd.read_parquet(movie_path)
    a_nodes = pd.read_parquet(actor_path)
    links = pd.read_parquet(link_path)
    # add bipartite attribute
    m_nodes.loc[:, 'bipartite'] = 0
    a_nodes.loc[:, 'bipartite'] = 1
    # count row in each file
    mn = len(m_nodes)
    an = len(a_nodes)
    
    # Initialize a nework and add the two patite, movies and actors
    bp = ig.Graph(directed=False)
    print(f'{fc.red}>>>>{fs.reset} Adding {mn} movies and {an} actors to network.')
    bp.add_vertices(len(m_nodes), m_nodes.to_dict('list'))
    bp.add_vertices(len(a_nodes), a_nodes.to_dict('list'))

    # translate tconst~nconst link pair to mid~aid pair
    mid = pd.DataFrame(np.zeros((mn, 2)), columns=['tconst', 'mid']) 
    mid.tconst = bp.vs[:mn]['tconst']
    mid.mid = [i for i in range(mn)]
    aid = pd.DataFrame(np.zeros((an, 2)), columns=['nconst', 'aid']) 
    aid.nconst = bp.vs[mn:]['nconst']
    aid.aid = [i + mn for i in range(an)]
    lid = links.merge(mid, left_on='tconst', right_on='tconst', how='left')
    lid = lid.merge(aid, left_on='nconst', right_on='nconst', how='left').loc[:,('mid','aid')]

    # Add links
    ln = len(lid)
    print(f'{fc.red}>>>>{fs.reset} Adding {ln} movie-actor links to network.')
    bp.add_edges(lid.values)

    # Get the projected movie network and actor network
    ## [ref](https://igraph.readthedocs.io/en/stable/api/igraph.Graph.html#bipartite_projection)
    print(f'{fc.red}>>>>{fs.reset} Projecting bipartite nework to movie nodes and actor nodes.')
    mg, ag = bp.bipartite_projection(types='bipartite')
    
    # Delete useless attribute in each graph
    attrs = ['bipartite', 'birthYear', 'deathYear', 'nconst', 'primaryName']
    print(f'{fc.red}>>>>{fs.reset} Cleanning useless attributes for Movie Network.')
    print(f'\t\t{attrs}')
    for attr in attrs:
        del(mg.vs[attr])
    attrs = ['bipartite', 'description', 'genres', 'averageRating', 
             'numVotes', 'primaryTitle', 'runtimeMinutes', 'startYear', 'tconst']
    print(f'{fc.red}>>>>{fs.reset} Cleanning useless attributes for Actor Network.')
    print(f'\t\t{attrs}')
    for attr in attrs:
        try:
            del(ag.vs[attr])
        except KeyError:
            print(f'\t\tThere is no {attr} attribute, pass')
        
    # Get GCC
    print(f'{fc.red}>>>>{fs.reset} Getting the GCC of Movie Network and Actor Network.')
    mgcc = mg.connected_components().giant()
    agcc = ag.connected_components().giant()
    print(f'\t\tMovie: (N{len(mg.vs)}, L{len(mg.es)})-->(N{len(mgcc.vs)}, L{len(mgcc.es)})')
    print(f'\t\tActor: (N{len(ag.vs)}, L{len(ag.es)})-->(N{len(agcc.vs)}, L{len(agcc.es)})')

    # Save checkpoints
    if save_path and tag:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)

        movie_path = os.path.join(save_path, f'{tag}.movie.ig.pickle')
        actor_path = os.path.join(save_path, f'{tag}.actor.ig.pickle')
        print(f'{fc.red}>>>>{fs.reset} Saving to {movie_path} and {actor_path}')
        ig.Graph.write_pickle(mg, movie_path)
        ig.Graph.write_pickle(ag, actor_path)
    print(f'{fc.red}>>>>{fs.reset} Summary:\n{mg.summary()}\n{ag.summary()}')
    print(f'{fc.red}>>>>{fs.reset} Done !')
    
    return mg, ag, mgcc, agcc

def build_bipartite_networkx(movie_path, actor_path, link_path, save_path=None, tag=None):
    '''Build network by networkx'''    
    m_nodes = pd.read_parquet(movie_path)
    a_nodes = pd.read_parquet(actor_path)
    links = pd.read_parquet(link_path)
    
    B = nx.Graph()
    B.add_nodes_from(m_nodes.tconst.values, bipartite=0) # Add the node attribute "bipartite"
    B.add_nodes_from(a_nodes.nconst.values, bipartite=1)
    B.add_edges_from(links.loc[:, ['tconst', 'nconst']].values)

    print(nx.is_connected(B))

    M = bipartite.projected_graph(B, m_nodes.tconst.values)
    A = bipartite.projected_graph(B, a_nodes.nconst.values)
    
    # Save checkpoints
    if save_path and tag:
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)

        movie_path = os.path.join(save_path, f'{tag}.movie.ig.pickle')
        actor_path = os.path.join(save_path, f'{tag}.actor.ig.pickle')
        print(f'>>>> Saving to {movie_path} and {actor_path}')
        nx.write_gpickle(M, movie_path)
        nx.write_gpickle(A, actor_path)

    # # how to load checkpoint
    # M = nx.read_gpickle('./imdb/network/imdb.movie.gpickle')
    return M, A

money2int = lambda str: int(''.join(re.findall('[\d]+', str)))
def add_money_attribute(g, df, key, func=money2int):    
    '''
    Add boxoffice or budject values to graph.
    g: graph    df: dataframe   key: new attribute key    func: convert string with numbers to a int type
    '''
    for lab, row in df.iterrows():
        if not pd.isna(row[key]):
            try: 
                g.vs.find(tconst=lab)[key] = money2int(row[key])
            except ValueError:
                continue

def add_numerical_attribute(g, df, key):  
    '''Add numerical value to graph: g-graph, df-dataframe, key-attribute_name'''
    for lab, row in df.iteritems():
        if not pd.isna(row):
            try: 
                g.vs.find(tconst=lab)[key] = row
            except ValueError:
                continue

def get_first_last(g, key='startYear', title='primaryTitle'):
    '''Get the first movie and last movies in given network'''
    first_year = min(g.vs[key])
    last_year = max(g.vs[key])
    first_movie = g.vs.select(startYear_eq = first_year)[title]
    last_movie = g.vs.select(startYear_eq = last_year)[title]
    print(f'The first movie in our network is {fc.blue}{first_movie[0]}{fs.reset} \
in {fc.blue}{first_year}{fs.reset}, and the newest is {fc.blue}{last_movie[0]}{fs.reset}, \
released in {fc.blue}{last_year}{fs.reset}')
    return first_year, last_year

def draw_powerlaws(dists, N, L, tag, s=1):
    '''Generate a powerlaw distribution and plot its log-log distribution'''
    n = len(dists)
    fig, axes = plt.subplots(1, n-1, figsize=(4 * (n-1), 5), dpi=100)
    for i in range(1, n):
        dist = dists[i]
        axes[i-1].scatter(dist[:, 0], dist[:, 1] / N, s=s, label='Power Law')
        axes[i-1].scatter(dists[0][:, 0], dists[0][:, 1] / N, s=1, label=f'{tag}')
        axes[i-1].set_title(f'Power law with $\gamma$={i+1}')
        axes[i-1].set_xlabel('$k$')
        axes[i-1].set_ylabel('$p_k$')
        axes[i-1].set_xscale('log')
        axes[i-1].set_yscale('log')
        axes[i-1].grid(alpha=0.6)
        axes[i-1].legend()
    plt.suptitle(f'{tag} vs Power Law\n(N={N} L={L})')
    plt.tight_layout()
    plt.show()

def draw_degree_distributions_bar(gs, tags, title, s=1, xmin=0):
    assert len(gs) == len(tags)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5))
    gammas = []
    for g, tag in zip(gs, tags):
        dist = dict2np(Counter(g.degree()))
        gamma = powerlaw.Fit(g.degree(), True, xmin, verbose=False).alpha
        gammas.append(gamma)
        n = len(g.vs)
        ax1.scatter(dist[:, 0], dist[:, 1] / n, s=s, label=f'{tag}: $\gamma$={gamma:.2f}', alpha=0.6)
        bar = ax2.bar(dist[:, 0], dist[:, 1] / n, label=f'{tag}', alpha=0.5)
        ax2.scatter(dist[:, 0], dist[:, 1] / n, c=[bar.patches[0].get_facecolor()], s=0.5, alpha=0.8)
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('$p_k$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(alpha=0.6)
    ax1.legend(loc='lower left', frameon=False)
    ax2.set_xlabel('$k$')
    ax2.set_ylabel('$p_k$')
    ax2.grid(alpha=0.6)
    ax2.legend()
    plt.suptitle(f'{title} {tags[0]} Degree Distribution\n(N={len(gs[0].vs)})')
    plt.tight_layout()
    plt.show()   
    return gammas

def get_max_degree_timely(g, year):
    sub_vs = g.vs.select(startYear_le = year)
    sub_g = g.subgraph(sub_vs)
    sud_k = sub_g.degree()
    sub_gamma = powerlaw.Fit(sud_k, True, 0, verbose=False).alpha
    sub_k_max = np.max(sud_k)
    sub_n = len(sub_vs)
    sub_l = len(sub_g.es)
    return sub_gamma, sub_k_max, sub_l, sub_n

def get_ba_er_params(ns, ls):
    params_ba, params_er = [], []
    for n, l in zip(ns, ls):
        n = int(n)
        g = ig.Graph.Barabasi(n, 2) 
        ks = g.degree()
        gamma = powerlaw.Fit(ks, True, 0, verbose=False).alpha
        k_max = np.max(ks)
        l_ = len(g.es)
        params_ba.append([gamma, k_max, l_])
        
        g = ig.Graph.Erdos_Renyi(n=n, p=2*l/n/(n-1), directed=False, loops=False)
        ks = g.degree()
        gamma = powerlaw.Fit(ks, True, 0, verbose=False).alpha
        k_max = np.max(ks)
        l_ = len(g.es)
        params_er.append([gamma, k_max, l_])
    return np.array(params_ba), np.array(params_er)

def draw_gamma_maxk_len_overtime(params, years):
    fig, axes = plt.subplots(3,3, figsize=(14, 12))
    xs = years
    ns = params[0][:, -1]
    axes[0, 1].plot(years, ns, '-', label='Node Growth')
    axes[0, 1].set_xticks([1960, 1980, 2000, 2020], [1960, 1980, 2000, 2020])
    axes[0, 1].set_xlabel('$year$')
    axes[0, 1].set_ylabel('$node(t)$')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.6)
    # axes[0, 1].spines['top'].set_visible(False)
    # axes[0, 1].spines['right'].set_visible(False)
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')
    ylabels = ['$\gamma$', '$k_{max}$', '$link$']
    for i, ax in enumerate(axes[1]):
        # if i == 0:
        #     ax.hist(params[0][:, i], rwidth=0.9, alpha=0.5, label='Netflix Movie')
        #     ax.hist(params[1][:, i], rwidth=0.9, alpha=0.5, label='Barabasi-Albert')
        #     ax.hist(params[2][:, i], rwidth=0.9, alpha=0.5, label='Erdős–Rényi') 
        # else:
        ax.plot(ns, params[0][:, i], '-', label='Netflix Movie')
        ax.plot(ns, params[1][:, i], '-', label='Barabasi-Albert')
        ax.plot(ns, params[2][:, i], '-', label='Erdős–Rényi')
        ax.set_title(f'{ylabels[i]} with growing nodes')
        ax.set_xlabel('$node$')
        ax.set_xscale('log')
        ax.set_ylabel(ylabels[i])
        ax.grid(alpha=0.6)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.legend()
    ylabels = ['$\gamma(t)$', '$k_{max}(t)$', '$link(t)$']
    for i, ax in enumerate(axes[2]):
        # if i == 0:
        #     ax.hist(params[0][:, i], rwidth=0.9, alpha=0.5, label='Netflix Movie')
        #     ax.hist(params[1][:, i], rwidth=0.9, alpha=0.5, label='Barabasi-Albert')
        #     ax.hist(params[2][:, i], rwidth=0.9, alpha=0.5, label='Erdős–Rényi') 
        # else:
        ax.plot(years, params[0][:, i], '-', label='Netflix Movie')
        ax.plot(years, params[1][:, i], '-', label='Barabasi-Albert')
        ax.plot(years, params[2][:, i], '-', label='Erdős–Rényi')
        ax.set_xticks([1960, 1980, 2000, 2020], [1960, 1980, 2000, 2020])
        ax.set_title(f'{ylabels[i]} over time')
        ax.set_xlabel('$year$')
        ax.set_ylabel(ylabels[i])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.show()


def get_degree_dist_timely(g, year, key):
    if key == 'startYear':
        sub_vs = g.vs.select(startYear_le = year)
    if key == 'birthYear':
        sub_vs = g.vs.select(birthYear_le = year)
    sub_g = g.subgraph(sub_vs[:])
    sub_gamma = powerlaw.Fit(sub_g.degree(), True, 0, verbose=False).alpha
    sub_dist = dict2np(Counter(sub_g.degree()))
    return sub_dist, sub_gamma, len(sub_vs)

def draw_degree_dist_timely(g, years, key, title, s=1):
    plt.figure(figsize=(5,6))
    for year in years:
        dist, gamma, n = get_degree_dist_timely(g, year, key)
        plt.scatter(dist[:, 0], dist[:, 1] / n, s=s, label=f'${year}$ $\gamma${gamma:.2f} N{n}')
    plt.xlabel('$k$')
    plt.ylabel('$p_k$')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(alpha=0.6)
    plt.title(f'{title} Degree Distribution Yearly')
    plt.legend(frameon=False)
    plt.show()

def animate_degree_dist_yearly(g, xlim, ylim, start, end, key, tag,
                              ffmpeg_path='/opt/homebrew/Cellar/ffmpeg/5.1.2/bin/ffmpeg'):
    plt.ioff()
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path # change to your ffmpeg path

    fig, ax = plt.subplots(figsize=(5,6))
    ax.set_xlabel('$k$')
    ax.set_ylabel('$p_k$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, xlim)
    ax.set_ylim(ylim, 1)
    ax.grid(alpha=0.6)
    xdata, ydata = [], []
    ln, = ax.plot([], [], '.', ms=20, label='Release: 0000')
    txt_title = ax.set_title('')
    txt_label = plt.legend(loc='upper right')

    def update(year):
        dist, gamma, n = get_degree_dist_timely(g, year, key)
        xdata = dist[:,0]
        ydata = dist[:,1] / n
        ln.set_data(xdata, ydata)  
        txt_title.set_text(f'Evolution of Degree in {tag}\n(Y:{year} N:{n} $\gamma$:{gamma:.2f})')
        txt_label.get_texts()[0].set_text(f'Release: {year}')
        return ln,

    years = range(start, end+1, 1)
    anim = animation.FuncAnimation(fig, update, years, blit=True)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()     

def add_edge_attr(g, key, e_key):
    for e in g.es:
        src_id = e.source
        dst_id = e.target
        if g.vs[src_id][key] > g.vs[dst_id][key]:
            e[e_key] = g.vs[src_id][key]
        else:
            e[e_key] = g.vs[dst_id][key]

def draw_shortest_path_length_hist(gs, tags, title):
    fig, axes = plt.subplots(1,3, figsize=(12, 5))
    mus = []
    for ax, g, tag in zip(axes, gs, tags):
        hist = g.path_length_hist(directed=False)
        x = np.linspace(hist._min, hist._max, len(hist._bins)+1)[:-1]
        bar, mu, std = np.array(hist._bins), hist._running_mean._mean, hist._running_mean._sd
        mus.append(mu)
        ax.set_title(f'{tag}\n($\mu$:{mu:.2f} $\sigma$:{std:.2f})')
        ax.bar(x, bar / bar.sum(), color='#75fb52', edgecolor='#7b549b')
        ax.plot(x, bar / bar.sum(), c='#e8964a', linestyle='--', marker='o', mec='#75fb52')
        ax.set_xlabel('Shortest Path Length')
        ax.set_ylabel('$p_{len}$')
        ax.set_xticks(x)
        ax.grid(alpha=0.6)
    plt.suptitle(f'{title}\nShortest Path Length Distribution')
    plt.tight_layout()
    plt.show()  
    return mus

def get_summary(gs, gammas, mus, tags):
    summary = pd.DataFrame(np.ones((3,10)), columns = ['Network', 'N', 'L', '⟨k⟩', '⟨k²⟩', 'σ', 'γ', '⟨d⟩', 'Diameter', 'Density'])
    summary.Network = tags
    summary.N = [len(g.vs) for g in gs]
    summary.L = [len(g.es) for g in gs]
    summary.loc[:, '⟨k⟩']  = [np.mean(g.degree()) for g in gs]
    summary.loc[:, '⟨k²⟩'] = [np.square(g.degree()).mean() for g in gs]
    summary.loc[:, 'σ'] = [np.std(g.degree()) for g in gs]
    summary.loc[:, 'γ'] = gammas
    summary.loc[:, '⟨d⟩'] = mus
    summary.Diameter = [g.diameter(directed=False) for g in gs]
    summary.Density = [g.density() for g in gs]  
    return summary

def get_centralities(g, inplace=False):
    '''a better solution to get centrality and then set as nodes attributes'''
    if inplace == False: g = g.copy()
    g.vs['dg'] = g.degree()
    g.vs['cn'] = g.closeness(mode='in', normalized=True)
    g.vs['bt'] = g.betweenness(directed=False)
    g.vs['ev'] = g.eigenvector_centrality(directed=False)
    if inplace == False: return g

def get_top_centrality_movies(g, limit, attrs, centrality='dg'):
    '''
    get top movies under given centrality, such as degree, closeness, between and eigenvector.
    For closeness contrality, there exist null values under unconnected network.
    '''
    i = 0
    top_movies = pd.DataFrame(columns=[a for a in attrs] + [centrality])
    cv = np.array(g.vs[centrality])
    cv = cv[~np.isnan(cv)] # drop nan
    for d in sorted(set(cv), reverse=True)[:limit]: 
        if centrality == 'dg':
            vs = g.vs.select(dg_eq=d)
        if centrality == 'cn':
            vs = g.vs.select(cn_eq=d)
        if centrality == 'bt':
            vs = g.vs.select(bt_eq=d)
        if centrality == 'ev':
            vs = g.vs.select(ev_eq=d)
        for v in vs:
            top_movies.loc[i] = [v[a] for a in attrs] + [d]
            i += 1          
    return top_movies

def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

def set_fig_info(fig, axes, rows, cols, color='red', direction='left'):
    pad = 20 # in points
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=15, ha='center', va='bottom', color=color)
    if direction == 'left':
        for ax, row in zip(axes[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size=15, ha='right', va='center', rotation='vertical', color=color)
    else:
        for ax, row in zip(axes[:,-1], rows): # plot the colums annotation on the right
            w, h = get_ax_size(fig, ax)
            ax.annotate(row, xy=(1, 0.5), xytext=(w+80, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size=15, ha='right', va='center', rotation='vertical', color=color)
    return axes

def centralities_correlation(g, 
                             ctags=['Degree', 'Closeness', 'Betweenness', 'Eigenvector']):
    mask = ~np.isnan(g.vs['cn'])
    centralities = [np.array(c)[mask] for c in [g.vs['dg'], g.vs['cn'], g.vs['bt'], g.vs['ev']]]

    ax_i = [(idx[0], idx[1]-1) for idx in np.mgrid[0:4, 0:4].reshape(2, 16).T if idx[0]<idx[1]]
    c_i = [(idx[0], idx[1]) for idx in np.mgrid[0:4, 0:4].reshape(2, 16).T if idx[0]<idx[1]]
    fig, axes = plt.subplots(3,3, figsize=(15,12))
    rows = ['Degree', 'Closeness', 'Betweenness']
    cols = ['Closeness', 'Betweenness', 'Eigenvector']
    axes = set_fig_info(fig, axes, rows, cols, direction='right')
    for idx, ax in np.ndenumerate(axes):
        if idx not in ax_i:
            ax.scatter(0, 0, color='white')
            ax.axis('off')
        else:
            i, j = idx
            ax.scatter(centralities[i], centralities[j+1], s=5, alpha=0.5)
            # ax.set_title(f'{ctags[i]}-Centrality vs {ctags[j+1]}-Centrality')
            # ax.set_ylabel(f'{ctags[j+1]} Centrality')
            # ax.set_xlabel(f'{ctags[i]} Centrality')
    corrcoef = pd.DataFrame(np.corrcoef(centralities), index=ctags, columns=ctags)
    tab = axes[-1,0].table(np.round(corrcoef.values, 2), rowLabels=ctags, colLabels=ctags, loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 3)
    axes[-1,0].set_title('correlation coefficient matrix')
    plt.tight_layout()
    plt.show()
    
    return corrcoef

def attribute_with_centralities(g, attrs, 
                                ctags=['Degree', 'Closeness', 'Betweenness', 'Eigenvector'], 
                                title='Movie'):
    cn = [0 if np.isnan(c) else c for c in g.vs['cn']]
    centralities = [np.array(c) for c in [g.vs['dg'], cn, g.vs['bt'], g.vs['ev']]]
    
    n, m = 4, len(attrs)
    corrcoef = pd.DataFrame(np.zeros((n, m)), index=ctags, columns=attrs)
    fig, axes = plt.subplots(m, n, figsize=(4*n, 4*m))
    axes = set_fig_info(fig, axes, attrs, ctags)
    for i, (ax_row, attr) in enumerate(zip(axes, attrs)):
        for j, (ax, ctag, xs) in enumerate(zip(ax_row, ctags, centralities)):
            ys = np.array(g.vs[attr]).astype('float')
            # filter attribute's null value
            mask = ~np.isnan(ys) 
            xs, ys = xs[~np.isnan(ys)], ys[~np.isnan(ys)]
            corrcoef.iloc[j, i] = np.corrcoef([xs, ys])[0,1]
            ax.scatter(xs, ys, s=1, alpha=0.2)
            # ax.set_xlabel(f'{ctag} Centrality')
            # ax.set_ylabel(f'{attr.title()}')
    plt.suptitle(f'Different Centralities vs {title} Attributes\n')
    plt.tight_layout()
    plt.show()
    
    return corrcoef

def list_add(a, b):
    try:
        return list(a) + list(b)
    except TypeError:
        return a   

def get_nominal(g, attr, key):
    types = []
    for i in g.vs[attr]:
        try:
            if key in i:
                types.append(1)
            else:
                types.append(0)
        except TypeError:
            types.append(0)
    return types

ia = Cinemagoer()
def get_movie_from_imdb(tconst):
    '''
    title, cast, genre, runtimes, country, budget, box_office, rating, votes, year, director, plot_outline, plot, synopsis
    input: mid is a int type, so need to convert to a seven length string, becasue imdb id is a 7 length string.
    '''
    row = []
    # get movie info
    m = ia.get_movie(tconst[2:])
    row.append(['nm'+p.personID for p in m.get('cast')])
    if m.get('box office'): 
        row.append(m.get('box office').get('Budget'))
        row.append(m.get('box office').get('Cumulative Worldwide Gross'))
    else:
        row.append(np.nan)
        row.append(np.nan)
    
    if m.get('plot outline'):
        row.append(m.get('plot outline')) 
    else:
        row.append(None)
        
    if m.get('plot'):
        row.append(m.get('plot')[0])
    else:
        row.append(None) 
        
    if m.get('synopsis'):
        row.append(m.get('synopsis')[0])
    else:
        row.append(None)
    return row

def get_movie_keywords(tconst):
    m = ia.get_movie_keywords(tconst[2:])
    if m.get('data').get('keywords'):
        kw = m.get('data').get('keywords')
        return list(map(lambda x: x.split('-'), kw))
    else:
        return []

def get_movie_comment(tconst):
    ''' 
    [The Shawshank Redemption Reviews Page](https://www.imdb.com/title/tt0111161/reviews?ref_=tt_urv)
    [Code Ref From](https://stackoverflow.com/questions/68243944/collecting-all-movie-reviews-from-imdb-from-certain-dramas)
    '''
    start_url = f'https://www.imdb.com/title/' + tconst + '/reviews?ref_=tt_urv'
    reviews = []
    with requests.Session() as s:
        s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
        res = s.get(start_url)
        soup = BeautifulSoup(res.text,"lxml")
        divs = soup.find_all("div", {"class": "text show-more__control"})
        if len(divs) == 0:
            return [] # if we got nothing on that page, return a empty list
        for div in divs:
            review = div.get_text(strip=True) 
            reviews.append(review)
    return reviews

def get_person_from_imdb(nconst):
    '''
    name, height, birthday, country, mini bigoraphy, trivia, quotes
    input: pid is a int type, so need to convert to a seven length string, becasue imdb id is a 7 length string.
    '''
    row = []
    # get person info
    p = ia.get_person(nconst[2:])
    row.append(p.data.get('name'))
    row.append(p.data.get('height'))
    
    if p.data.get('birth date'):
        row.append(p.data.get('birth date')) 
    else:
        row.append(None)
        
    if p.data.get('birth info'):
        country = p.data.get('birth info')['birth place']
        row.append(country) 
    else:
        row.append(None)
    
    if p.data.get('mini biography'):
        row.append(p.data.get('mini biography')[0])
    else:
        row.append(None)
        
    if p.data.get('trivia'):
        row.append(p.data.get('trivia')) 
    else:
        row.append([])
    
    return row

def movie_downloader(mg, i=0, save=False):
    tconst_list = mg.vs['tconst']
    m_attr = pd.DataFrame(columns=['cast', 'budget', 'boxOffice', 'plotOutline', 'plot', 'synopsis'])
    for tconst in tqdm(tconst_list[i:]):
        try:
            m_attr.loc[tconst] = pd.Series(get_movie_from_imdb(tconst))
        except IMDbDataAccessError:
            m_attr.loc[tconst] = pd.Series(get_movie_from_imdb(tconst))
            continue
    if save==True:
        m_attr.to_csv('./netflix/download/movie.attr.csv', index=True)
    return m_attr

def keywords_downloader(mg, i=0, save=False):
    tconst_list = mg.vs['tconst']
    m_keyword = pd.DataFrame(columns=['keywords'])
    for tconst in tqdm(tconst_list[i:]):
        try:
            m_keyword.loc[tconst, 'keywords'] = get_movie_keywords(tconst)
        except IMDbDataAccessError:
            m_keyword.loc[tconst, 'keywords'] = get_movie_keywords(tconst)
            continue
    if save==True:
        m_keyword.to_csv('./netflix/download/movie.keyword.csv', index=True)
    return m_keyword

def reviews_downloader(mg, i=0, save=False):
    tconst_list = mg.vs['tconst']
    m_review = pd.DataFrame(columns=['reviews'])
    for tconst in tqdm(tconst_list[i:]):
        try:
            m_review.loc[tconst, 'reviews'] = get_movie_comment(tconst)
        except:
            m_review.loc[tconst, 'reviews'] = get_movie_comment(tconst)
            continue
    if save==True:
        m_review.to_csv('./netflix/download/movie.review.csv', index=True)
    return m_review

def actor_downloader(ag, i=0, save=False):
    a_attr = pd.DataFrame(columns=['name', 'height', 'birthday', 'country', 'bigoraphy', 'trivia'])
    nconst_list = ag.vs['nconst']
    for nconst in tqdm(nconst_list[i:]):
        try:
            a_attr.loc[nconst] = get_person_from_imdb(nconst)
        except IMDbDataAccessError:
            a_attr.loc[nconst] = get_person_from_imdb(nconst)
            continue
    if save==True:
        a_attr.to_csv('./netflix/download/actor.attr.csv', index=True)
    return a_attr

money2int = lambda str: int(''.join(re.findall('[\d]+', str)))
def add_money_attribute(g, df, key, func=money2int):    
    for lab, row in df.iterrows():
        if not pd.isna(row[key]):
            try: 
                # print(type(row[key]), row[key])
                g.vs.find(tconst=lab)[key] = money2int(row[key])
            except ValueError:
                continue

def get_layout(gcc, save='./netflix/network/movie.layout.npz'):
    # install 3d force atlas algorithm
    # !git clone https://github.com/Ankhee/ndim_forceatlas
    # !python setup.py install
    adj_matrix = np.array(list((gcc.get_adjacency())))
    # 2d force atlas algorithm
    forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True, linLogMode=False,adjustSizes=False,
                            edgeWeightInfluence=1.0, jitterTolerance=1.0, barnesHutOptimize=True,
                            barnesHutTheta=1.2, multiThreaded=False, scalingRatio=10.0, strongGravityMode=False,
                            gravity=1.0,verbose=True)

    # getting layout
    fa3_layout = ndforceatlas(adj_matrix, n_dim=3)
    kk3_layout = gcc.layout('kk', dim=3)
    fa2_layout = forceatlas2.forceatlas2_igraph_layout(gcc, pos=None, iterations=1000)
    kk2_layout = gcc.layout('kk', dim=2)

    # save to a npz file
    with open(save, 'wb') as f:
        np.savez(f, 
                fa3d=fa3_layout, 
                kk3d=kk3_layout, 
                fa2d=fa2_layout, 
                kk2d=kk2_layout)
    return fa3_layout, kk3_layout, fa2_layout, kk2_layout

def plot_3d_graph(gcc, layt, size_type, group_type):
    customdata=pd.DataFrame(columns=['ID', 'Title', 'Country', 'Genre', 'Rating', 'Community'])
    customdata.ID = gcc.vs['tconst']
    customdata.Title = gcc.vs['primaryTitle']
    customdata.Country = gcc.vs['country']
    customdata.Genre = gcc.vs['genres']
    customdata.Rating = gcc.vs['averageRating']
    customdata.Community = gcc.vs[group_type]
    group = gcc.vs[group_type]
    size = np.array(gcc.vs[size_type])
    scale = np.mean(np.array(layt).max(axis=0) - np.array(layt).min(axis=0)) / 400
    size = size/ size.std() * scale

    N = len(gcc.vs)
    L = len(gcc.es)
    C = max(group)

    Xn = [layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]# y-coordinates
    Zn = [layt[k][2] for k in range(N)]# z-coordinates

    Xe, Ye, Ze =[], [], []
    for e in gcc.es:
        Xe += [layt[e.source][0], layt[e.target][0], None] # x-coordinates of edge ends
        Ye += [layt[e.source][1], layt[e.target][1], None]
        Ze += [layt[e.source][2], layt[e.target][2], None]

    trace1 = go.Scatter3d(  x = Xe,
                            y = Ye,
                            z = Ze,
                            mode = 'lines',
                            line = dict(color='rgba(125,125,125,0.6)', width=1),
                            hoverinfo = 'none', )

    trace2 = go.Scatter3d(  x = Xn,
                            y = Yn,
                            z = Zn,
                            mode = 'markers',
                            name = 'Movie',
                            marker = dict(  symbol='circle',
                                            size=size,
                                            color=group,
                                            colorscale='Viridis',
                                            line=dict(color='rgba(50,50,50,0.8)', width=0.5)
                                            ),
                            customdata = customdata,
                            hovertemplate = "<br>".join([
                                                        "ID: %{customdata[0]}",
                                                        "Title: %{customdata[1]}",
                                                        "Country: %{customdata[2]}",
                                                        "Genre: %{customdata[3]}",
                                                        "Rating: %{customdata[4]:.1f}",
                                                        "Community:  %{customdata[5]}"
                                                        ]), )

    axis = dict(showbackground = False,
                showline = False,
                zeroline = False,
                showgrid = False,
                showticklabels = False,
                title = '' )

    layout = go.Layout( title = f"Movie Network in Netflix (3D visualization)<br>N={N}, L={L}, Community={C}",
                        width = 800, height = 600, showlegend = False,
                        scene = dict(   xaxis = dict(axis),
                                        yaxis = dict(axis),
                                        zaxis = dict(axis) ),
                        margin = dict( t=100 ),
                        hovermode = 'closest',
                        annotations = [dict(showarrow = False,
                                            text = "Data source: <a href='http://https://www.imdb.com/interfaces/'>[1] IMDB dataset</a>",
                                            xref = 'paper',
                                            yref = 'paper',
                                            x = 0,
                                            y = 0.1,
                                            xanchor = 'left',
                                            yanchor = 'bottom',
                                            font = dict(size = 14) )
                                    ],    
                    )

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def plot_2d_graph(gcc, layt, size_type, group_type, scale=4000):
    customdata=pd.DataFrame(columns=['ID', 'Title', 'Country', 'Genre', 'Rating', 'Community'])
    customdata.ID = gcc.vs['tconst']
    customdata.Title = gcc.vs['primaryTitle']
    customdata.Country = gcc.vs['country']
    customdata.Genre = gcc.vs['genres']
    customdata.Rating = gcc.vs['averageRating']
    customdata.Community = gcc.vs[group_type]
    group = gcc.vs[group_type]
    size = np.array(gcc.vs[size_type])
    scale = np.mean(np.array(layt).max(axis=0) - np.array(layt).min(axis=0)) / scale
    size = size/ size.std() * scale

    N = len(gcc.vs)
    L = len(gcc.es)
    C = max(group)

    Xn = [layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]# y-coordinates

    Xe, Ye =[], []
    for e in gcc.es:
        Xe += [layt[e.source][0], layt[e.target][0], None] # x-coordinates of edge ends
        Ye += [layt[e.source][1], layt[e.target][1], None]

    trace1 = go.Scatter(    x = Xe,
                            y = Ye,
                            mode = 'lines',
                            line = dict(color='rgba(125,125,125,0.3)', width=1),
                            hoverinfo = 'none', )

    trace2 = go.Scatter  (  x = Xn,
                            y = Yn,
                            mode = 'markers',
                            name = 'Movie',
                            marker = dict(  symbol='circle',
                                            size=size,
                                            color=group,
                                            colorscale='Viridis',
                                            line=dict(color='rgba(50,50,50,0.6)', width=0.5)
                                            ),
                            customdata = customdata,
                            hovertemplate = "<br>".join([
                                                        "ID: %{customdata[0]}",
                                                        "Title: %{customdata[1]}",
                                                        "Country: %{customdata[2]}",
                                                        "Genre: %{customdata[3]}",
                                                        "Rating: %{customdata[4]:.1f}",
                                                        "Community:  %{customdata[5]}"
                                                        ]), )

    axis = dict(
                showgrid = False, # thin lines in the background
                zeroline = False, # thick line at x=0
                visible = False,  # numbers below
                )

    layout = go.Layout( title = f"Movie Network in Netflix (2D visualization)<br>N={N}, L={L}, Community={C}",
                        width = 1000, height = 800, showlegend = False,
                        xaxis = axis,
                        yaxis = axis,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin = dict( t=100 ),
                        hovermode = 'closest',
                        annotations = [dict(showarrow = False,
                                            text = "Data source: <a href='http://https://www.imdb.com/interfaces/'>[1] IMDB dataset</a>",
                                            xref = 'paper',
                                            yref = 'paper',
                                            x = 0,
                                            y = 0.1,
                                            xanchor = 'left',
                                            yanchor = 'bottom',
                                            font = dict(size = 14) )
                                    ],    
                    )

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def plot_3d_communities(partition, layt, scale, method):
    cluster_g = partition.cluster_graph()
    m = partition.membership
    a = np.hstack([layt, [[i] for i in m]])
    volumns = sorted(Counter(m).items()) # list of tulpe
    cidx = [k for k, _ in volumns]
    cvol = [v for _, v in volumns]
    size = np.array(cvol) / np.array(cvol).std() * scale
    layt = []
    for i, vol in volumns:
        tmp = a[np.where(a[:,-1] == i)]
        layt.append([np.mean(tmp[:,0]), np.mean(tmp[:,1]), np.mean(tmp[:,2])])
    N = len(cluster_g.vs)
    L = len(cluster_g.es)
    Xn = [layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]# y-coordinates
    Zn = [layt[k][2] for k in range(N)]# z-coordinates

    Xe, Ye, Ze =[], [], []
    for e in cluster_g.es:
        Xe += [layt[e.source][0], layt[e.target][0], None] # x-coordinates of edge ends
        Ye += [layt[e.source][1], layt[e.target][1], None]
        Ze += [layt[e.source][2], layt[e.target][2], None]

    customdata = np.vstack((cidx,cvol)).T

    trace1 = go.Scatter3d(  x = Xe,
                            y = Ye,
                            z = Ze,
                            mode = 'lines',
                            line = dict(color='rgba(125,125,125,0.6)', width=4),
                            hoverinfo = 'none', )

    trace2 = go.Scatter3d(  x = Xn,
                            y = Yn,
                            z = Zn,
                            mode = 'markers',
                            name = 'Community',
                            marker = dict(  symbol='circle',
                                            size=size,
                                            color=cidx,
                                            colorscale='rainbow',
                                            line=dict(color='rgba(50,50,50,0.8)', width=1)
                                            ),
                            customdata = customdata,
                            hovertemplate = "<br>".join([
                                                        "Community: %{customdata[0]}",
                                                        "Volumn: %{customdata[1]}",
                                                        ]), )

    axis = dict(showbackground = False,
                showline = False,
                zeroline = False,
                showgrid = False,
                showticklabels = False,
                title = '' )

    layout = go.Layout( title = f"Communities in Netflix Movie Network(3D visualization)<br>N={N}, L={L}",
                        width = 800, height = 600, showlegend = False,
                        scene = dict(   xaxis = dict(axis),
                                        yaxis = dict(axis),
                                        zaxis = dict(axis) ),
                        margin = dict( t=100 ),
                        hovermode = 'closest',
                        annotations = [dict(showarrow = False,
                                            text = f"Community Detection Method: [1] {method}</a>",
                                            xref = 'paper',
                                            yref = 'paper',
                                            x = 0,
                                            y = 0.1,
                                            xanchor = 'left',
                                            yanchor = 'bottom',
                                            font = dict(size = 14) )
                                    ],    
                    )

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    fig.show()

symbol = string.punctuation
stop_words = set(stopwords.words('english'))

def to_british(tokens):
    for t in tokens:
        t = re.sub(r"(...)our$", r"\1or", t)
        t = re.sub(r"([bt])re$", r"\1er", t)
        t = re.sub(r"([iy])s(e$|ing|ation)", r"\1z\2", t)
        t = re.sub(r"ogue$", "og", t)
        yield t

wnl = nltk.WordNetLemmatizer().lemmatize
def lemmaTokenizer(text):
    text = re.sub(symbol, ' ', text)
    text = text.lower() # string type
    tokens = nltk.word_tokenize(text) # list
    tokens = map(wnl, tokens)                            # lemmatize
    # tokens = [t for t in tokens if t not in stop_words ] # delete stopwords
    # tokens = list(to_british(tokens))                    
    return tokens

def get_corpus(partition, text_series, label, limit=10, text_type='list'):
    """text_type can be list or string """
    sub_graphs = sorted(partition.subgraphs(), key=lambda g: len(g.vs), reverse=True)[:limit]
    print(f'Pareparing Corpus for {label}')
    corpus = []
    pbar = tqdm(sub_graphs)
    for g in pbar:
        community_ids = g.vs['tconst']
        community_texts = text_series[community_ids]
        community_doc = ' '.join(community_texts)
        corpus.append(community_doc)
    return corpus

def get_TFIDF(partition, text_series, label, limit=10):
    corpus = get_corpus(partition, text_series, label, limit)
    vectorizer = TfidfVectorizer(analyzer='word', stop_words = 'english' )
    X = vectorizer.fit_transform(corpus)
    keys = vectorizer.get_feature_names_out()    
    return keys, X

class Happy_Hedonometer():

    def __init__(self, delta, path):
        # delta is a parameter to tune our Hedonometer
        self.LabMT = pd.read_csv(path, sep="\t")
        LabMT_dict = dict(zip(self.LabMT['word'].values, self.LabMT['happiness_average'].values))
        self.LabMT_dict = {k:v for k, v in LabMT_dict.items() if v > 5+delta or v < 5-delta }
        self.get_hap_of_tokens = lambda freq: np.array([LabMT_dict[w]*p for w, p in freq if w in LabMT_dict.keys()])
        self.vectorizer = CountVectorizer(analyzer='word', stop_words = 'english')
    
    def __call__(self, text):
        try:
            X = self.vectorizer.fit_transform([text]) # input should be a list, not a string text
        except ValueError:
            return None # if the document only contain stop words or a null document, return None
        keys = self.vectorizer.get_feature_names_out()
        prob = X.toarray()[0] / X.toarray()[0].sum()
        freqDist = zip(keys, prob)
        hap_values = self.get_hap_of_tokens(freqDist)
        if hap_values.shape[0] == 0:
            return None
        else:
            return hap_values.sum()

def plot_hap_dist_for_attr(g, attr, limit):
    attr_types = Counter(reduce(list_add, g.vs[attr])).most_common(limit)
    _, axes = plt.subplots(limit,6, figsize=(20, 2.5*limit))
    bins = range(9)
    titles = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    for i, (key, _) in enumerate(attr_types):
        attr_list = np.array(get_nominal(g, attr, key))
        for ax, title, hap_key in zip(axes[i], titles, hap_keys):
            h = pd.Series(g.vs[hap_key])[attr_list==True]
            ax.set_title(f'{title}({key})')
            ax.hist(h.dropna(), bins, density=True, rwidth=0.8, color='#75fb52', edgecolor='#7b549b')
            ax.set_xlabel('Happy intensity')
            ax.set_ylabel('$p_{len}$')
            ax.grid(alpha=0.6)
            ax.set_xticks(bins)
    plt.suptitle(f'Sentiment value distribution for Top {limit} {attr.title()}\n\n')
    plt.tight_layout()
    plt.show() 

def plot_mean_dist_for_attr(g, attr, limit):
    attr_types = Counter(reduce(list_add, g.vs[attr])).most_common(limit)
    _, ax = plt.subplots(1,1, figsize=(5, 8))
    bins = range(9)
    legend = [k for k, v in attr_types]
    xtick = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    for i, (key, _) in enumerate(attr_types):
        attr_list = np.array(get_nominal(g, attr, key))
        means = []
        for lgd, hap_key in zip(legend, hap_keys):
            h = pd.Series(g.vs[hap_key])[attr_list==True]
            means.append(h.dropna().mean())
        ax.hist([1,2,3,4,5,6], means, label=lgd)
    ax.set_xlabel('Document Source')
    ax.set_ylabel('$Sentiment mean$')
    ax.grid(alpha=0.6)
    ax.legend()
    ax.set_xticks([1,2,3,4,5,6], xtick)
    plt.suptitle(f'Sentiment mean for Top {limit} {attr.title()}\n\n')
    plt.tight_layout()
    plt.show() 

def plot_mean_dist_for_attr_x(g, attr, limit, haps_summary, ax=None):
    attr_types = Counter(reduce(list_add, g.vs[attr])).most_common(limit)
    g_means = haps_summary.loc['mean']
    if ax == None: 
        _, ax = plt.subplots(1,1, figsize=(7, 5))
    xtick = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    ax.scatter([1,2,3,4,5,6], g_means, marker='+', s=600, c='black', label='All')
    attr_mean_dict = {}
    for key, _ in attr_types:
        attr_list = np.array(get_nominal(g, attr, key))
        means = []
        for hap_key in hap_keys:
            h = pd.Series(g.vs[hap_key])[attr_list==True]
            means.append(h.mean())
        ax.plot([1,2,3,4,5,6], means, '--o', label=key)
        attr_mean_dict[key] = means
    ax.set_xlabel('Document Source')
    ax.set_ylabel('Sentiment mean')
    ax.grid(alpha=0.6)
    ax.legend(loc=(1.04, 0))
    ax.set_xticks([1,2,3,4,5,6], xtick, rotation=45)
    if ax == None:
        plt.suptitle(f'Sentiment mean for Top {limit} {attr.title()}\n\n')
        plt.tight_layout()
        plt.show() 
    else:
        return attr_mean_dict, ax

def plot_mean_dist_for_attr_y(g, attr, limit, haps_summary):
    attr_types = Counter(reduce(list_add, g.vs[attr])).most_common(limit)
    g_means = haps_summary.loc['mean']
    _, ax = plt.subplots(1,1, figsize=(7, 5))
    labels = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    for hap_key, lab in zip(hap_keys, labels):
        h = pd.Series(g.vs[hap_key])
        means = []
        for key, _ in attr_types:
            attr_list = np.array(get_nominal(g, attr, key))
            h_ = h[attr_list==True]
            means.append(h_.mean())
        ln = ax.plot(range(limit), means, '--x', label=lab)
        ax.plot([-0.5, limit-0.5],  [g_means[lab], g_means[lab]], color=ln[0].get_color(), alpha=0.4)
    ax.set_xlabel('Document Source')
    ax.set_ylabel('Sentiment Mean')
    ax.grid(alpha=0.6)
    ax.legend(loc=(1.04, 0))
    ax.set_xlim(-0.5, limit-0.5)
    ax.set_xticks(range(limit), [k for k, _ in attr_types], rotation=45)
    plt.suptitle(f'Sentiment mean for Top {limit} {attr.title()}\n\n')
    plt.tight_layout()
    plt.show() 

def statistic_less(x, y, axis):
    """ H1: x is less than y
        H0: y is less than x
        The probability of obtaining a test statistic y less than or equal to the observed value under the null hypothesis is 
    """
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
def statistic_bigger(x, y, axis):
    """ H1: x is bigger than y
        H0: y is bigger than x
        The probability of obtaining a test statistic y bigger than or equal to the observed value under the null hypothesis is 
    """
    return np.mean(y, axis=axis) - np.mean(x, axis=axis)

def boxplot_for_perumtation(g, attr, key, title, permutation=1000, ax=None):
    if ax == None:
        _, ax = plt.subplots(1,1, figsize=(7, 5))
    xtick = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    key_list = np.array(get_nominal(g, attr, key))
    means = []
    permutatioon_means = []
    hap_statistics = []
    for hap_key in hap_keys:
        h = pd.Series(g.vs[hap_key])[key_list==True]
        means.append(h.mean())
        means_ = []
        x = h.dropna()
        y = pd.Series(g.vs[hap_key]).dropna()
        if x.mean() < y.mean():
            statistic = statistic_less
        else:
            statistic = statistic_bigger
        res = permutation_test((x, y), statistic, vectorized=True,
                      n_resamples=1000, alternative='less')
        hap_statistics.append(res.statistic)
        hap_statistics.append(res.pvalue)
        for i in range(permutation):
            h_ = np.random.choice(g.vs[hap_key], len(h)).astype('float')
            means_.append(np.nanmean(h_))
        permutatioon_means.append(means_)
    ax.plot([1,2,3,4,5,6], means, '--o', label=key)
    ax.boxplot( permutatioon_means, 
                notch=True, 
                positions=[1,2,3,4,5,6], 
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    # ax.set_xlabel('Document Source')
    # ax.set_ylabel('Sentiment mean')
    ax.grid(alpha=0.6)
    ax.legend()
    ax.set_title(title)
    ax.set_xticks([1,2,3,4,5,6], xtick, rotation=45)
    if ax == None:
        plt.tight_layout()
        plt.show() 
    else:
        return ax, hap_statistics

def cloudplot_for_perumtation(g, attr, key, title, permutation=1000, ax=None):
    if ax == None:
        _, ax = plt.subplots(1,1, figsize=(7, 5))
    xtick = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    key_list = np.array(get_nominal(g, attr, key))
    means = []
    permutatioon_means = []
    for i, hap_key in enumerate(hap_keys):
        h = pd.Series(g.vs[hap_key])[key_list==True]
        means.append(h.mean())
        means_ = []
        for _ in range(permutation):
            h_ = np.random.choice(g.vs[hap_key], len(h)).astype('float')
            means_.append(np.nanmean(h_))
        permutatioon_means.append(means_)
        # xs = np.random.normal(loc=i+1, scale=0.5, size=permutation)
        xs = (np.random.rand(permutation) - 0.5) * 0.5 + i + 1
        ax.scatter(xs, means_, s=0.02, alpha=0.7)
        ax.scatter(i+1, np.mean(means_), c='black')

    ax.plot([1,2,3,4,5,6], means, '--o', label=key)

    # ax.set_xlabel('Document Source')
    # ax.set_ylabel('Sentiment mean')
    ax.grid(alpha=0.6)
    ax.legend()
    ax.set_title(title)
    ax.set_xticks([1,2,3,4,5,6], xtick, rotation=45)
    if ax == None:
        plt.tight_layout()
        plt.show() 
    else:
        return ax

def plot_hap_dist_for_community(g, comm_key, limit):
    comm_count = Counter(g.vs[comm_key]).most_common(limit)
    _, axes = plt.subplots(limit,6, figsize=(20, 2.5*limit))
    bins = range(9)
    titles = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    for i, title in enumerate(titles):
        axes[0, i].set_title(f'{title}')
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    for i, (key, _) in enumerate(comm_count):
        member_list = np.where(np.array(g.vs[comm_key]) == key)
        axes[i, 0].set_ylabel(f'Community{i+1}')
        for ax, title, hap_key in zip(axes[i], titles, hap_keys):
            h = pd.Series(g.vs[hap_key]).iloc[member_list]
            ax.hist(h.dropna(), bins, density=True, rwidth=0.8, color='#75fb52', edgecolor='#7b549b')
            ax.set_xlabel('Happy intensity')
            ax.grid(alpha=0.6)
            ax.set_xticks(bins)
    plt.suptitle(f'Sentiment value distribution for Top {limit} Communities\n\n')
    plt.tight_layout()
    plt.show() 

def plot_mean_dist_for_community_x(g, comm_key, limit, haps_summary, ax=None):
    comm_count = Counter(g.vs[comm_key]).most_common(limit)
    g_means = haps_summary.loc['mean']
    if ax == None: 
        _, ax = plt.subplots(1,1, figsize=(7, 5))
    xtick = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    ax.scatter([1,2,3,4,5,6], g_means, marker='+', s=600, c='black', label='All')
    comm_mean_dict = {}
    for key, _ in comm_count:
        member_list = np.where(np.array(g.vs[comm_key]) == key)
        means = []
        for hap_key in hap_keys:
            h = pd.Series(g.vs[hap_key]).iloc[member_list]
            means.append(h.mean())
        comm_mean_dict[key] = means
        ax.plot([1,2,3,4,5,6], means, '--o', label=f'Community{key}')
    ax.set_xlabel('Document Source')
    ax.set_ylabel('Sentiment mean')
    ax.grid(alpha=0.6)
    ax.legend(loc=(1.04, 0))
    ax.set_xticks([1,2,3,4,5,6], xtick, rotation=45)
    if ax == None: 
        plt.suptitle(f'Sentiment mean for Top {limit} Communities\n')
        plt.tight_layout()
        plt.show() 
    return comm_mean_dict, ax

def plot_mean_dist_for_community_y(g, comm_key, limit, haps_summary):
    comm_count = Counter(g.vs[comm_key]).most_common(limit)
    g_means = haps_summary.loc['mean']
    _, ax = plt.subplots(1,1, figsize=(7, 5))
    labels = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    for hap_key, lab in zip(hap_keys, labels):
        h = pd.Series(g.vs[hap_key])
        means = []
        for key, _ in comm_count:
            member_list = np.where(np.array(g.vs[comm_key]) == key)
            h_ = h.iloc[member_list]
            means.append(h_.mean())
        ln = ax.plot(range(limit), means, '--x', label=lab)
        ax.plot([-0.5, limit-0.5],  [g_means[lab], g_means[lab]], color=ln[0].get_color(), alpha=0.4)
    ax.set_xlabel('Community')
    ax.set_ylabel('Sentiment Mean')
    ax.grid(alpha=0.6)
    ax.legend(loc=(1.04, 0))
    ax.set_xlim(-0.5, limit-0.5)
    ax.set_xticks(range(limit), [f'Community{k}' for k, _ in comm_count], rotation=45)
    plt.suptitle(f'Sentiment mean for Top {limit} Communities\n')
    plt.tight_layout()
    plt.show() 

def boxplot_for_perumtation_community(g, comm_key, comm_id, title, permutation=1000, ax=None):
    if ax == None:
        _, ax = plt.subplots(1,1, figsize=(7, 5))
    xtick = ['Movie plot outlines', 'Movie plots', 'Movie synopsis', 'Movie keywords', 'Movie reviews', 'Netflix Description']
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    comm_count = Counter(g.vs[comm_key])[comm_id]
    means = []
    permutatioon_means = []
    for hap_key in hap_keys:
        member_list = np.where(np.array(g.vs[comm_key]) == comm_id)
        h = pd.Series(g.vs[hap_key]).iloc[member_list]
        means.append(h.mean())
        means_ = []
        for i in range(permutation):
            h_ = np.random.choice(g.vs[hap_key], len(h)).astype('float')
            means_.append(np.nanmean(h_))
        permutatioon_means.append(means_)
    ax.plot([1,2,3,4,5,6], means, '--o', label=f'Community{comm_id}')
    ax.boxplot( permutatioon_means, 
                notch=True, 
                positions=[1,2,3,4,5,6], 
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                )
    # ax.set_xlabel('Document Source')
    # ax.set_ylabel('Sentiment mean')
    ax.grid(alpha=0.6)
    ax.legend()
    ax.set_title(title)
    ax.set_xticks([1,2,3,4,5,6], xtick, rotation=45)
    if ax == None:
        plt.tight_layout()
        plt.show() 
    else:
        return ax

def ftest(s1,s2):
    '''F检验样本总体方差是否相等'''
    F = np.var(s1)/np.var(s2)
    v1 = len(s1) - 1
    v2 = len(s2) - 1
    p_val = 1 - 2*abs(0.5-f.cdf(F,v1,v2))
    if p_val < 0.05:
        equal_var=False
    else:
        equal_var=True
    return equal_var
         
def ttest_ind_fun(s1,s2):
    '''t检验独立样本所代表的两个总体均值是否存在差异'''
    equal_var = ftest(s1,s2)
    ttest,pval = ttest_ind(s1,s2,equal_var=equal_var)
    if pval < 0.05:
        flag = 'Reject'
    else:
       flag = 'Accept'
    return pval, flag

def get_attr_hap(g, attr, key):
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    key_list = np.array(get_nominal(g, attr, key))
    hap_list = []
    for hap_key in hap_keys:
        h = pd.Series(g.vs[hap_key])[key_list==True]
        hap_list.append(h.dropna())
    return hap_list

def get_comm_hap(g, comm_key, comm_id):
    hap_keys = ['hap_outline', 'hap_plot', 'hap_synopsis', 'hap_keyword', 'hap_review', 'hap_describe']
    hap_list = []
    for hap_key in hap_keys:
        member_list = np.where(np.array(g.vs[comm_key]) == comm_id)
        h = pd.Series(g.vs[hap_key]).iloc[member_list]
        hap_list.append(h.dropna())
    return hap_list