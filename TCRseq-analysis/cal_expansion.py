import os
import argparse
import gzip
import cPickle as pickle

import numpy as np
import pandas as pd


def read_vdj(filename, count=False, use_dna=False):
    if filename.endswith('.txt'):
        r = open(filename, 'r')
    elif filename.endswith('.txt.gz'):
        r = gzip.open(filename, 'rb')
    r.readline()
    for line in r:
        ele = line.split('\t')
        cc = float(ele[0])
        fq = float(ele[1])
        if cc <= 0:
            continue
        dna = ele[2]
        aa = ele[3]
        v = ele[4].split('*')[0]
        j = ele[6].split('*')[0]
        if '*' in aa or '_' in aa: ## in-frame
            continue
        if not (v.startswith('TRBV') and j.startswith('TRBJ')):
            continue
        if use_dna:
            seq = dna
        else:
            seq = aa
        if count:
            yield v, seq, j, cc
        else:
            yield v, seq, j
    r.close()

def clone_count(filename):
    f = {}
    t = 0.0
    for v, seq, j, c in read_vdj(filename, count=True, use_dna=False):
        t += c 
        vdj = (v,seq,j)
        if vdj in f:
            f[vdj] += c
        else:
            f[vdj] = c
    return f, t

def get_smp_ids(i):
    if pd.isnull(i):
        return []
    if i.find('_') < 0:
        return i
    a, B = i.split('_')
    return [a+'_'+b for b in B.split('+')] ## use all replicates

def entropy(s):
    p = np.array(s.values())
    p /= p.sum()
    return -(p*np.log2(p)).sum()

def clonality(s):
    if len(s) < 2:
        return 0
    return 1 - entropy(s)/np.log2(len(s))

def single(s):
    p = np.array(s.values())
    return p[p==1].sum() / float(p.sum())

def ham(s1 ,s2, dcut=1):
    c = 0
    for v1,aa1,j1 in s1:
        for v2,aa2,j2 in s2:
            if v1 == v2 and j1 == j2:
                d = sum([i != j for i,j in zip(aa1,aa2)])
                if d <= dcut:
                    c += 1
    return c

def jaccard(a,b):
    A = set(a.keys())
    B = set(b.keys())
    U = list(A | B)
    A0 = np.array([a.get(i,0) for i in U])
    B0 = np.array([b.get(i,0) for i in U])
    O = np.minimum(A0, B0)
    return O.sum() / float(A0.sum() + B0.sum() - O.sum())

def KL_divergence(a,b):
    O = list(set(a.keys()) & set(b.keys()))
    P = np.array([a[i] for i in O])
    Q = np.array([b[i] for i in O])
    P /= float(P.sum())
    Q /= float(Q.sum())
    return (P*np.log2(P/Q)).sum()

def JS_divergence(a,b):
    return (KL_divergence(a,b)+KL_divergence(b,a))*0.5

def clonal_expansion(meta_file,
                     data_path='../data/DanaFarberShipp2018May_clean/', 
                     out_clone='../work/_expanded_clones.csv.gz',
                     out_table='../work/_expanded_summary.csv'):
    ## read meta data
    meta = pd.read_csv(meta_file, index_col=0)
    meta.rename(columns={'Zumla patient no':'ZID'}, inplace=True)
    meta.index = meta.ZID

    ## read repertoire for each patient
    patients = {}
    for f in os.listdir(data_path):
        if '_SD_' not in f and f.endswith('.txt.gz') and 'gt1yr' in f: ## only choose patient with ASCT > 1 year
            ele = f.split('_')
            if len(ele) != 4:
                continue
            pid = ele[0]
            if pid in patients:
                patients[pid].append(f)
            else:
                patients[pid] = [f]

    ## save clones
    summary = []
    save_clones = gzip.open(out_clone, 'wb')
    save_clones.write('V,CDR3,J,PID,Type,Base,Fold\n')
    collect_fold = []
    for pid in patients:
        m = meta.loc[[pid]].values[0].tolist()
        files = patients[pid]
        if len(files) == 3:
            post1, post2, pre = sorted(files)
        else:
            continue
        print pre, post1, post2
        assert 'Pre' in pre
        pr, tr = clone_count(data_path+pre)
        p1, t1 = clone_count(data_path+post1)
        p2, t2 = clone_count(data_path+post2)
        pr_set = set(pr.keys())
        p1_set = set(p1.keys())
        p2_set = set(p2.keys())
        pool = pr_set | p1_set | p2_set

        ## check all clones
        all_fold = []
        expand_naive = set()
        expand_memory = set()
        for vdj in pool:
            ## define the baseline for clonal expansion
            if vdj not in pr:
                base = 0.5 / tr
            else:
                base = pr[vdj] / float(tr)
            ## maximum fold change
            fold = max(p1.get(vdj,0)/float(t1)/base, p2.get(vdj,0)/float(t2)/base)
            collect_fold.append(fold)

            if fold > 0:
                all_fold.append(fold)
            if fold >= 2: ## expansion definition
                if vdj not in pr or pr[vdj] <= 1: ## one or less
                    expand_naive.add(vdj)
                    save_clones.write('%s,%s,%s,'%vdj+'%s,Naive,%s,%s\n'%(pid, base, fold))
                else:
                    expand_memory.add(vdj)
                    save_clones.write('%s,%s,%s,'%vdj+'%s,Memory,%s,%s\n'%(pid, base, fold))

        ## I: Time, F: file, P: clone counts, PS: clones, T: total counts
        for I, F, P, PS, T in [('Pre',   pre,   pr, pr_set, tr), 
                               ('Post1', post1, p1, p1_set, t1), 
                               ('Post2', post2, p2, p2_set, t2)]:
            if I == 'Pre': ## change from Post1 to Post2
                jac = jaccard(p1, p2)
                div = JS_divergence(p1, p2)
            else: ## Pre to Post1 or Post2
                jac = jaccard(pr, P)
                div = JS_divergence(pr, P)
            summary.append(m+[
                    I, F, T,
                    entropy(P),
                    clonality(P),
                    np.mean(all_fold),
                    jac,
                    div,
                    len(expand_naive),
                    len(expand_memory),
                    len(expand_naive & PS) / float(len(PS)),
                    sum([P[i] for i in P if i in expand_naive]) / T,
                    len(expand_memory & PS) / float(len(PS)),
                    sum([P[i] for i in P if i in expand_memory]) / T])

    save_clones.close()
    np.savetxt(out_clone+'.all_fold.gz', np.array(collect_fold))

    ## save summary
    summary = pd.DataFrame(summary, columns=list(meta) + [
                    'Time', 'File', 'Total.Clones',
                    'Entropy',
                    'Clonality',
                    'Mean.Fold.Change',
                    'Jaccard.Index',
                    'JS.Divergence',
                    'Expand.Naive', 
                    'Expand.Memory',
                    'Expanded.Naive.Clonotype.Ratio',
                    'Expanded.Naive.Clone.Ratio',
                    'Expanded.Memory.Clonotype.Ratio',
                    'Expanded.Memory.Clone.Ratio'])
    summary.to_csv(out_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='../../tcrseq-analysis/', help='Data path')
    parser.add_argument('--workpath', type=str, default='./work/', help='Work path')

    args = parser.parse_args()

    clonal_expansion(meta_file=args.datapath+'PBMC_PFS_2018Sep.csv',
                     data_path=args.datapath+'DanaFarberShipp2018May_clean/', 
                     out_clone=args.workpath+'expanded_clones.csv.gz',
                     out_table=args.workpath+'expanded_summary.csv')

