"""
Code developed for the Bachelor's thesis:

"Causal Discovery with DirectLiNGAM: Experimental Evaluation of Ordering Heuristics"

Author: Lorenzo Venneri
University of Milano-Bicocca
Year: 2026
"""
import numpy as np
import pandas as pd
import time
import lingam
from graphviz import Digraph
import random
from lingam.utils import predict_adaptive_lasso
from contextlib import redirect_stdout
from io import StringIO
from joblib import Parallel, delayed
import multiprocessing
import os

# Codice per generazione grafo causale casuale

# Genero casualmente matrice di adiacenza/causalità
def generate_random_causal_matrix(d, prob):
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            if random.random() < prob:
                W[i, j] = 1
    return W

# Genero graficamente il grafo corrispondente
def plot_causal_graph(W, node_labels=None, title="Grafo Causale Originale"):
    dot = Digraph(comment=title)
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled',
             fontname='Arial', fontsize='12',
             fillcolor='lightblue', color='steelblue', penwidth='1.5')
    dot.attr('edge', fontname='Arial', fontsize='10', arrowsize='0.8')

    d = W.shape[0]

    # Aggiungo nodi
    if node_labels is None:
        node_labels = {i: f"X{i}" for i in range(d)}

    for i in range(d):
        dot.node(str(i), label=node_labels[i], fillcolor='lightyellow' if i==0 else 'lightblue')

    # Aggiungo archi
    for i in range(d):
        for j in range(d):
            if W[i, j] != 0:
                color = 'red' if W[i, j] < 0 else 'black'
                penwidth = '2' if abs(W[i, j]) > 1 else '1.5'
                dot.edge(str(i), str(j),
                         color=color,
                         penwidth=penwidth,
                         label='' if abs(W[i,j])==1 else str(W[i,j]))

    dot.attr(label=title, fontsize='16', labelloc='t', fontname='Arial Bold')
    dot.attr(margin='0.5')

    return dot

# Codice per generazione parametri ed equazioni
def generate_parameters(W, distributionp, mean, std):
    d = W.shape[0]  # calcola numero nodi
    parameters = np.zeros_like(W)

    for i in range(d):
        for j in range(d):  # controllo per ogni arco nella matrice se c'è una relazione causale tra i e j
            if W[i, j] != 0:
                if distributionp == 'normal':
                    # Genera il parametro usando una distribuzione normale
                    param = np.random.normal(mean, std)
                    while not (-1.5 <= param <= -0.5 or 0.5 <= param <= 1.5):
                        param = np.random.normal(mean, std)

                    param = np.round(param, 2)
                    parameters[i, j] = param

                elif distributionp == 'uniform':
                    # Genera con distribuzione uniforme
                    if np.random.rand() > 0.5:
                        param = np.random.uniform(-1.5, -0.5)
                    else:
                        param = np.random.uniform(0.5, 1.5)

                    param = np.round(param, 2)
                    parameters[i, j] = param

                elif distributionp == 'exponential':
                    # Genera con distribuzione esponenziale
                    param = np.random.exponential(1)
                    if param < 0:
                        param = -param
                    param = (param % 2) - 1.5  # Trasforma in [-1.5, 1.5]
                    if param < -1.5:
                        param = -1.5
                    if param > 1.5:
                        param = 1.5

                    param = np.round(param, 2)
                    parameters[i, j] = param

    return parameters
    
# Funzione per generare le equazioni in base al grafo causale
def generate_equations2(W, parameters, noise_std=0.1):
    d = W.shape[0]
    equations = []

    for i in range(d):
        terms = []

        # Genitori: W[j, i] != 0 significa j -> i
        for j in range(d):
            if W[j, i] != 0:
                terms.append(f"{np.round(parameters[j, i], 2)}*x{j}")

        # Rumore esogeno sempre presente
        terms.append(f"e{i}")

        equation = f"x{i} = " + " + ".join(terms)
        equations.append(equation)

    return equations


def parse_equations2(equations):
    terms = {}
    for eq in equations:
        lhs, rhs = eq.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()

        coeffs = []
        variables = []

        for part in rhs.split("+"):
            part = part.strip()
            if not part:
                continue
            if "*" in part:
                coeff, var = part.split("*")
                coeffs.append(float(coeff.strip()))
                variables.append(var.strip())
            else:
                coeffs.append(1.0)
                variables.append(part.strip())

        terms[lhs] = {"coeffs": coeffs, "vars": variables}

    return terms

# Funzione per generare i dati
def generate_data2(terms, sized, distribution):
    data = {}

    # 1) genera tutti i rumori e_i che compaiono in qualunque equazione
    noise_vars = set()
    for lhs, details in terms.items():
        for v in details["vars"]:
            if v.startswith("e"):
                noise_vars.add(v)

    for ev in noise_vars:
        data[ev] = generate_noise(distribution, sized)

    x_vars = list(terms.keys())
    parents = {}
    for x, details in terms.items():
        p = set(v for v in details["vars"] if v.startswith("x"))
        parents[x] = p
    remaining = set(x_vars)
    progress = True

    while remaining and progress:
        progress = False
        ready = [x for x in remaining if parents[x].issubset(data.keys())]

        for x in ready:
            details = terms[x]
            var_data = np.zeros(sized, dtype=float)

            for c, v in zip(details["coeffs"], details["vars"]):
                if v not in data:
                    raise ValueError(
                        f"Variabile '{v}' non generata prima di '{x}'. "
                        "Controlla l'ordine/struttura (possibile ciclo o parsing)."
                    )
                var_data += c * data[v]

            data[x] = var_data
            remaining.remove(x)
            progress = True

    if remaining:
        raise ValueError(
            f"Impossibile generare queste variabili (ciclo o dipendenze in avanti): {sorted(remaining)}"
        )
        
    return {k: v for k, v in data.items() if k.startswith("x")}

# Funzione per scegliere e generare il rumore con la distribuzione scelta nell'intervallo [1,3]
def generate_noise(distribution, size):
    if distribution == 'uniform':
        return np.random.uniform(1, 3, size=size)  
            
    elif distribution == 'exponential':
        # Esponenziale
        exp_noise = np.random.exponential(1, size=size)  
        exp_noise = np.interp(exp_noise, (exp_noise.min(), exp_noise.max()), (1, 3))
        return exp_noise
            
    elif distribution == 'laplace':
        # Laplace o Doppia esponenziale 
        laplace_noise = np.random.laplace(0, 1, size=size) 
        laplace_noise = np.interp(laplace_noise, (laplace_noise.min(), laplace_noise.max()), (1, 3))
        return laplace_noise

# Codice per trasformare matrice
def transform_matrix(A):
    # Crea una copia della matrice per non modificare l'originale
    transformed_matrix = np.zeros_like(A)

    # Itera sulla matrice
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:  # Se c'è un valore diverso da 0
                transformed_matrix[j, i] = 1  # Inverti la direzione e metti 1
            else:
                transformed_matrix[i, j] = 0  # Mantieni 0 dove c'era 0

    return transformed_matrix

# Codice per calcolare structural hamming distance
def structural_hamming_distance(true_graph, inferred_graph):
    # Calcola gli archi aggiunti (dove c'è un arco nel grafo inferito ma non nel grafo vero)
    added_edges = np.sum(np.logical_and(inferred_graph == 1, true_graph == 0))

    # Calcola gli archi rimossi (dove c'è un arco nel grafo vero ma non nel grafo inferito)
    removed_edges = np.sum(np.logical_and(true_graph == 1, inferred_graph == 0))

    # La Structural Hamming Distance è la somma di tutte le differenze
    shd = added_edges + removed_edges
    # Stampa i risultati per ciascuna categoria
    print(f"Archi aggiunti: {added_edges}")
    print(f"Archi rimossi: {removed_edges}")

    # Stampa la SHD finale
    print(f"Structural Hamming Distance: {shd}")
    return shd

# Codice per generare matrice prior knowledge dall'ordine causale dato
def generate_prior_knowledge(ordine_varianze, d):
    prior_knowledge = np.ones((d, d))

    # Aggiungi le relazioni causali dirette in base all'ordine causale
    for i in range(len(ordine_varianze) - 1):
        for j in range(i + 1, len(ordine_varianze)):
            # x_i causa x_j solo se i < j nell'ordine causale
            prior_knowledge[ordine_varianze[i], ordine_varianze[j]] = 0
    np.fill_diagonal(prior_knowledge, 0)
    print(prior_knowledge)
    return prior_knowledge

# Codice per creare grafo dalla matrice prior knowledge
def make_prior_knowledge_graph(prior_knowledge_matrix):
    di = Digraph(engine='dot')

    labels = [f'x{i}' for i in range(prior_knowledge_matrix.shape[0])]
    for label in labels:
        di.node(label, label)

    dirs = np.where(prior_knowledge_matrix > 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        di.edge(labels[from_], labels[to])

    dirs = np.where(prior_knowledge_matrix < 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        if to != from_:
            di.edge(labels[from_], labels[to], style='dashed')
    return di

# funzioni per calcolare metriche strutturali

def precision(true_graph, inferred_graph):
    # Veri Positivi (TP): Predizioni corrette
    tp = np.sum(np.logical_and(inferred_graph == 1, true_graph == 1))
    
    # Falsi Positivi (FP): Relazioni causali predette ma non esistenti
    fp = np.sum(np.logical_and(inferred_graph == 1, true_graph == 0))
    
    # Precision = TP / (TP + FP)
    precision_score = tp / (tp + fp) if (tp + fp) != 0 else 0
    return precision_score

def recall(true_graph, inferred_graph):
    # Vero Positivi (TP): Predizioni corrette
    tp = np.sum(np.logical_and(inferred_graph == 1, true_graph == 1))
    
    # Falsi Negativi (FN): Relazioni causali esistenti ma non predette
    fn = np.sum(np.logical_and(inferred_graph == 0, true_graph == 1))
    
    # Recall = TP / (TP + FN)
    recall_score = tp / (tp + fn) if (tp + fn) != 0 else 0
    return recall_score

def mean_squared_error(true_graph, inferred_graph):
    # Calcola la differenza tra i grafi, elevata al quadrato
    mse = np.mean((true_graph - inferred_graph) ** 2)
    return mse
    
 # Codice che permette di sfruttare direttamente l'ordine causale calcolato con le varianze per la ricostruzione del modello con Direct Lingam
def _estimate_adjacency_matrix(self, X, order):
        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(order)):
            target = order[i]
            predictors = order[:i]
            # target is exogenous variables if predictors are empty
            if len(predictors) == 0:
                continue

            B[target, predictors] = predict_adaptive_lasso(X, predictors, target)

        self._adjacency_matrix = B
        return B
        
# Codice per scegliere possibili ordini causali alternativi in base alle differenza tra le varianze dei nodi
def order_options(X,soglia):
  # Inizializza una lista per memorizzare gli ordini
  all_orders = []
  vaar=X.var()
  ordr = list(vaar.sort_values().index)

  orig=ordr

  # Itera e verifica gli scambi basati sulla soglia
  for _ in range(len(ordr) - 1):
      current_order = ordr.copy()  # Ripristina sempre l'ordine originale
      for i in range(len(ordr) - 1):
          var1, var2 = vaar[current_order[i]], vaar[current_order[i + 1]]
          diff = abs(var1 - var2)

          # Se la differenza di varianza è sotto la soglia, scambiamo
          if diff < soglia:
                  current_order[i], current_order[i + 1] = current_order[i + 1], current_order[i]

                  # Aggiunge il nuovo ordine alla lista se non è già presente
                  if current_order not in all_orders  and current_order != orig:
                      all_orders.append(current_order.copy())  # Aggiungi il nuovo ordine alla lista

      # Dopo ogni ciclo, ripristina l'ordine originale per il prossimo ciclo
      ordr = current_order.copy()
  return all_orders


  # codice riepilogo con limite valutazione ordine
def riepilogo2(d, prob, distribution, distributionp, sized, soglia, limite):
    W = generate_random_causal_matrix(d, prob)
    params = generate_parameters(W, distributionp, mean=0, std=1.5)
    equations = generate_equations2(W, params)
    terms = parse_equations2(equations)
    data = generate_data2(terms, sized, distribution)
    Y = pd.DataFrame({f"x{i}": data[f"x{i}"] for i in range(d)})
    start_time = time.time()
    model = lingam.DirectLiNGAM()
    model.fit(Y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    a=model.adjacency_matrix_
    A = transform_matrix(a)
    with redirect_stdout(StringIO()):
        DRshd = structural_hamming_distance(W,A)
    PrecDL= precision(W,A)
    RecDL=recall(W,A)
    MerDL=mean_squared_error(W,A)
    start_time1 = time.time()
    variances = Y.var()
    sorted_columns = variances.sort_values().index
    variable_names = ['x' + str(i) for i in range(d)]
    variable_to_index = {name: index for index, name in enumerate(variable_names)}
    ordine_varianze = [variable_to_index[var] for var in sorted_columns]
    with redirect_stdout(StringIO()):
        prior_knowledge = generate_prior_knowledge(ordine_varianze, d)
    model1 = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model1.fit(Y)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1
    b=model1.adjacency_matrix_
    B = transform_matrix(b)
    with redirect_stdout(StringIO()):
        PRshd=structural_hamming_distance(W,B)
    PrecPR= precision(W,B)
    RecPR=recall(W,B)
    MerPR=mean_squared_error(W,B)
    best_order=ordine_varianze
    Z = Y.values
    from CausalDisco.baselines import (
        var_sort_regress,
        r2_sort_regress
    )
    start_timevs = time.time()
    with redirect_stdout(StringIO()):
        WvarS=1.0*(var_sort_regress(Z)!=0)
    end_timevs = time.time()
    elapsed_timevs = end_timevs - start_timevs
    with redirect_stdout(StringIO()):
        varShd=structural_hamming_distance(W,WvarS)
    RecVR=recall(W,WvarS)
    PrecVR=precision(W,WvarS)
    MerVR=mean_squared_error(W,WvarS)
    start_timer2 = time.time()
    with redirect_stdout(StringIO()):
        Wr2=1.0*(r2_sort_regress(Z)!=0)
    end_timer2 = time.time()
    elapsed_timer2 = end_timer2 - start_timer2
    with redirect_stdout(StringIO()):
        r2Shd=structural_hamming_distance(W,Wr2)
    RecR2=recall(W,Wr2)
    PrecR2=precision(W,Wr2)
    MerR2=mean_squared_error(W,Wr2)
    start_time2 = time.time()
    variances = Y.var()
    sorted_columns = variances.sort_values().index
    variable_names = ['x' + str(i) for i in range(d)]

    variable_to_index = {name: index for index, name in enumerate(variable_names)}

    ordine_varianze = [variable_to_index[var] for var in sorted_columns]
    ordini_possibili=order_options(Y,soglia)
    convord=converted_orders(ordini_possibili,d)
    limite = max(limite, len(convord) // 3)
    Morder= M_ordine2(
        Y, ordine_varianze,
        center_first=True
    )
    convord_limited = random.sample(convord, k=min(limite, len(convord))) #limito valutazione di ordini alternativi per ridurre tempo di esecuzione
    results = Parallel(
        n_jobs=-1, 
        prefer="processes" 
    )(
        delayed(eval_order2)(order,Y)
        for order in convord_limited
    )
    morder, best_orderv = max(results, key=lambda x: x[0])
    if morder>Morder:
        best_order=best_orderv
    c=_estimate_adjacency_matrix(lingam.DirectLiNGAM(), Z, order=best_order)
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    C=transform_matrix(c)
    with redirect_stdout(StringIO()):
        Dshd=structural_hamming_distance(W,C)
    PrecAO= precision(W,C)
    RecAO=recall(W,C)
    MerAO=mean_squared_error(W,C)
    start_time3 = time.time()
    variances = Y.var()
    sorted_columns = variances.sort_values().index
    variable_names = ['x' + str(i) for i in range(d)]

    variable_to_index = {name: index for index, name in enumerate(variable_names)}

    ordine_varianze = [variable_to_index[var] for var in sorted_columns]
    l=_estimate_adjacency_matrix(lingam.DirectLiNGAM(), Z, order=ordine_varianze)

    end_time3 = time.time()

    elapsed_time3 = end_time3 - start_time3
    L=transform_matrix(l)
    with redirect_stdout(StringIO()):
        Lshd=structural_hamming_distance(W,L)
    PrecL= precision(W,L)
    RecL=recall(W,L)
    MerL=mean_squared_error(W,L)
    data = {
        "Metodo di apprendimento": [
            "DirectLiNGAM",
            "DirectLiNGAM con prior knowledge",
            "DirectLiNGAM con inserimento dell'ordine",
            "DirectLiNGAM con euristica su ordine",
            "R2 sortability",
            "Var sortability"
        ],
        "Tempo di calcolo (secondi)": [
            elapsed_time,
            elapsed_time1,
            elapsed_time3,
            elapsed_time2,
            elapsed_timer2,
            elapsed_timevs
        ],
        "Structural Hamming Distance": [
            DRshd,
            PRshd,
            Lshd,
            Dshd,
            r2Shd,
            varShd
        ],
        "Precision": [
            PrecDL,
            PrecPR,
            PrecL,
            PrecAO,
            PrecR2,
            PrecVR
        ],
        "Recall": [
            RecDL,
            RecPR,
            RecL,
            RecAO,
            RecR2,
            RecVR
        ],
        "MSE": [
            MerDL,
            MerPR,
            MerL,
            MerAO,
            MerR2,
            MerVR
        ],
        "Nodi":[
            d,
            d,
            d,
            d,
            d,
            d
        ],
        "Campioni":[
            sized,
            sized,
            sized,
            sized,
            sized,
            sized
        ],
        "Soglia scambio":[
            soglia,
            soglia,
            soglia,
            soglia,
            soglia,
            soglia
        ],
        "Sparsità grafo":[
            prob,
            prob,
            prob,
            prob,
            prob,
            prob
        ]
    }

    df = pd.DataFrame(data)
    return df    

# funzione per creare effettivamente un ordine causale da un ordine di variabili attraverso una conversione
def converted_orders(ordini_possibili,d):
    converted = []
    variable_names = ['x' + str(i) for i in range(d)]
    variable_to_index = {name: index for index, name in enumerate(variable_names)}
    for order in ordini_possibili:
        converted_order = [variable_to_index[var_name] for var_name in order]
        converted.append(converted_order)
    return converted

# funzione per calcolare la funzione M di un ordine, utile per parallelizzazione    
def eval_order2(order,Y):
    Morderi = M_ordine2(
        Y, order,
        center_first=True
    )
    return Morderi, order
    
# funzioni necessarie al calcolo di M    
def residual_std(xi_std, xj_std):
    beta = np.mean(xi_std * xj_std)   # var(xj_std)=1
    return xi_std - beta * xj_std

def entropy(u):
    """Calculate entropy using the maximum entropy approximations."""
    k1 = 79.047
    k2 = 7.4129
    gamma = 0.37457
    return (1 + np.log(2 * np.pi)) / 2 - k1 * (
        np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

def M_for_fixed_i_fast_entropy(X, U, i, entropy, normalize=True):
    M = 0.0

    xi = X[:, i]
    sxi = xi.std()
    if sxi == 0.0:
        return 0.0

    xi_std = (xi - xi.mean()) / sxi
    H_xi = entropy(xi_std)

    H_xj = {}

    for j in U:
        if j == i:
            continue

        xj = X[:, j]
        sxj = xj.std()
        if sxj == 0.0:
            continue

        xj_std = (xj - xj.mean()) / sxj

        if j not in H_xj:
            H_xj[j] = entropy(xj_std)

        ri_j = residual_std(xi_std, xj_std)
        rj_i = residual_std(xj_std, xi_std)

        sri = ri_j.std()
        srj = rj_i.std()
        if sri == 0.0 or srj == 0.0:
            continue

        dmi = (H_xj[j] + entropy(ri_j / sri)) - (H_xi + entropy(rj_i / srj))
        v = min(0.0, float(dmi))
        M += v * v

    denom = len(U) - 1
    if normalize and denom > 0:
        M /= denom

    return -M

def compute_residual_fast(xi, xj):
    xj_c = xj - xj.mean()
    denom = np.mean(xj_c * xj_c)
    if denom == 0.0:
        return xi
    xi_c = xi - xi.mean()
    beta = np.mean(xi_c * xj_c) / denom
    return xi - beta * xj


def M_ordine2(X_df, order, center_first=True):
    X_work = X_df.to_numpy(copy=True).astype(float)
    p = X_work.shape[1]

    order = list(order)
    if len(order) != p:
        raise ValueError("order must have length == n_features")

    if center_first:
        X_work = X_work - X_work.mean(axis=0, keepdims=True)

    remaining = np.array(order, dtype=int)
    sommaM = 0.0

    for m in order:
        U = remaining

        # 1) usa la versione nuova per il punteggio M_m
        sommaM += M_for_fixed_i_fast_entropy(X_work, U, m, entropy, normalize=True)

        # 2) update dataset: residualizza tutto su xm con fast general
        xm = X_work[:, m]  # niente copy necessario
        for i in U:
            if i != m:
                X_work[:, i] = compute_residual_fast(X_work[:, i], xm)

        remaining = remaining[remaining != m]

    return sommaM

 # codice riepilogo senza limitazione ad ordini alternativi valutati
def riepilogo(d, prob, distribution, distributionp, sized, soglia, Mparallelo):
    W = generate_random_causal_matrix(d, prob)
    params = generate_parameters(W, distributionp, mean=0, std=1.5)
    equations = generate_equations2(W, params)
    terms = parse_equations2(equations)
    data = generate_data2(terms, sized, distribution)
    Y = pd.DataFrame({f"x{i}": data[f"x{i}"] for i in range(d)})
    start_time = time.time()
    model = lingam.DirectLiNGAM()
    model.fit(Y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    a=model.adjacency_matrix_
    A = transform_matrix(a)
    with redirect_stdout(StringIO()):
        DRshd = structural_hamming_distance(W,A)
    PrecDL= precision(W,A)
    RecDL=recall(W,A)
    MerDL=mean_squared_error(W,A)
    if (PrecDL + RecDL) != 0:
        F1DL = 2 * (PrecDL * RecDL) / (PrecDL + RecDL)
    else:
        F1DL = 0
    start_time1 = time.time()
    variances = Y.var()
    sorted_columns = variances.sort_values().index
    variable_names = ['x' + str(i) for i in range(d)]
    variable_to_index = {name: index for index, name in enumerate(variable_names)}
    ordine_varianze = [variable_to_index[var] for var in sorted_columns]
    with redirect_stdout(StringIO()):
        prior_knowledge = generate_prior_knowledge(ordine_varianze, d)
    model1 = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model1.fit(Y)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1
    b=model1.adjacency_matrix_
    B = transform_matrix(b)
    with redirect_stdout(StringIO()):
        PRshd=structural_hamming_distance(W,B)
    PrecPR= precision(W,B)
    RecPR=recall(W,B)
    MerPR=mean_squared_error(W,B)
    if (PrecPR + RecPR) != 0:
        F1PR = 2 * (PrecPR * RecPR) / (PrecPR + RecPR)
    else:
        F1PR = 0
    best_order=ordine_varianze
    Z = Y.values
    from CausalDisco.baselines import (
        var_sort_regress,
        r2_sort_regress
    )
    start_timevs = time.time()
    with redirect_stdout(StringIO()):
        WvarS=1.0*(var_sort_regress(Z)!=0)
    end_timevs = time.time()
    elapsed_timevs = end_timevs - start_timevs
    with redirect_stdout(StringIO()):
        varShd=structural_hamming_distance(W,WvarS)
    RecVR=recall(W,WvarS)
    PrecVR=precision(W,WvarS)
    MerVR=mean_squared_error(W,WvarS)
    if (PrecVR + RecVR) != 0:
        F1VR = 2 * (PrecVR * RecVR) / (PrecVR + RecVR)
    else:
        F1VR = 0
    start_timer2 = time.time()
    with redirect_stdout(StringIO()):
        Wr2=1.0*(r2_sort_regress(Z)!=0)
    end_timer2 = time.time()
    elapsed_timer2 = end_timer2 - start_timer2
    with redirect_stdout(StringIO()):
        r2Shd=structural_hamming_distance(W,Wr2)
    RecR2=recall(W,Wr2)
    PrecR2=precision(W,Wr2)
    MerR2=mean_squared_error(W,Wr2)
    if (PrecR2 + RecR2) != 0:
        F1R2 = 2 * (PrecR2 * RecR2) / (PrecR2 + RecR2)
    else:
        F1R2 = 0
    start_time2 = time.time()
    variances = Y.var()
    sorted_columns = variances.sort_values().index
    variable_names = ['x' + str(i) for i in range(d)]

    variable_to_index = {name: index for index, name in enumerate(variable_names)}

    ordine_varianze = [variable_to_index[var] for var in sorted_columns]
    ordini_possibili=order_options(Y,soglia)
    convord=converted_orders(ordini_possibili,d)
    Morder= M_ordine2(
        Y, ordine_varianze,
        center_first=True
    )
    if Mparallelo==True :
            n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) 
            results = Parallel(
                n_jobs=n_jobs,
                prefer="processes" 
            )(
                delayed(eval_order2)(order,Y)
                for order in convord
            )
            morder, best_orderv = max(results, key=lambda x: x[0])
            if morder>Morder:
                best_order=best_orderv
    else :
        for idx, order in enumerate(convord):
            Morderi=M_ordine2(
                    Y, order,
                    center_first=True
        )
            if Morderi>Morder:
                Morder=Morderi
                best_order=order
    c=_estimate_adjacency_matrix(lingam.DirectLiNGAM(), Z, order=best_order)
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    C=transform_matrix(c)
    with redirect_stdout(StringIO()):
        Dshd=structural_hamming_distance(W,C)
    PrecAO= precision(W,C)
    RecAO=recall(W,C)
    MerAO=mean_squared_error(W,C)
    if (PrecAO + RecAO) != 0:
        F1AO = 2 * (PrecAO * RecAO) / (PrecAO + RecAO)
    else:
        F1AO = 0
    start_time3 = time.time()
    variances = Y.var()
    sorted_columns = variances.sort_values().index
    variable_names = ['x' + str(i) for i in range(d)]

    variable_to_index = {name: index for index, name in enumerate(variable_names)}

    ordine_varianze = [variable_to_index[var] for var in sorted_columns]
    l=_estimate_adjacency_matrix(lingam.DirectLiNGAM(), Z, order=ordine_varianze)

    end_time3 = time.time()

    elapsed_time3 = end_time3 - start_time3
    L=transform_matrix(l)
    with redirect_stdout(StringIO()):
        Lshd=structural_hamming_distance(W,L)
    PrecL= precision(W,L)
    RecL=recall(W,L)
    MerL=mean_squared_error(W,L)
    if (PrecL + RecL) != 0:
        F1L = 2 * (PrecL * RecL) / (PrecL + RecL)
    else:    
        F1L = 0
    data = {
        "Metodo di apprendimento": [
            "DirectLiNGAM",
            "DirectLiNGAM con prior knowledge",
            "DirectLiNGAM con inserimento dell'ordine",
            "DirectLiNGAM con euristica su ordine",
            "R2 sortability",
            "Var sortability"
        ],
        "Tempo di calcolo (secondi)": [
            elapsed_time,
            elapsed_time1,
            elapsed_time3,
            elapsed_time2,
            elapsed_timer2,
            elapsed_timevs
        ],
        "Structural Hamming Distance": [
            DRshd,
            PRshd,
            Lshd,
            Dshd,
            r2Shd,
            varShd
        ],
        "Precision": [
            PrecDL,
            PrecPR,
            PrecL,
            PrecAO,
            PrecR2,
            PrecVR
        ],
        "Recall": [
            RecDL,
            RecPR,
            RecL,
            RecAO,
            RecR2,
            RecVR
        ],
        "MSE": [
            MerDL,
            MerPR,
            MerL,
            MerAO,
            MerR2,
            MerVR
        ],
        "F1 Score": [
            F1DL,
            F1PR,
            F1L,
            F1AO,
            F1R2,
            F1VR
        ],
        "Nodi":[
            d,
            d,
            d,
            d,
            d,
            d
        ],
        "Campioni":[
            sized,
            sized,
            sized,
            sized,
            sized,
            sized
        ],
        "Soglia scambio":[
            soglia,
            soglia,
            soglia,
            soglia,
            soglia,
            soglia
        ],
        "Sparsità grafo":[
            prob,
            prob,
            prob,
            prob,
            prob,
            prob
        ]
    }

    df = pd.DataFrame(data)
    return df 
