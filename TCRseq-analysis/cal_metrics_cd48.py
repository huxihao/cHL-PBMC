import os
import argparse
import gzip
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='./DanaFarberShipp2020Feb/', help='Data path')
parser.add_argument('--workpath', type=str, default='./work/', help='Work path')
parser.add_argument('--vdjtools', type=str, default='vdjtools')

args = parser.parse_args()

def get_rep(infile):
    if infile.endswith('.txt'):
        inp = open(infile, 'r')
    elif infile.endswith('.txt.gz'):
        inp = gzip.open(infile, 'rb')
    else:
        print 'Unknown format', infile
        return infile
    inp.readline() ## header
    total = 0.0
    save = [{}, {}, {}] ## CDR3 DNA, CDR3 AA, V-J genes
    for line in inp:
        count, freq, dna, aa, v, d, j = line.split('\t')[:7]
        count = float(count)
        if count < 0:
            continue
        total += count
        case = [(v, dna, j), (v, aa, j), (v, j)]
        for d, k in zip(save, case):
            if k in d:
                d[k] += count
            else:
                d[k] = count
    return total, save

def stat_rep(rep):
    p = np.array(rep.values())
    out = [p.sum(), (p==1).sum(), len(p), p.mean()]
    p /= p.sum()
    en = - (p * np.log2(p)).sum()
    co = 1 - en / np.log2(len(p))
    return out + [en, co]

def save_metric(datapath, outfile):
    metric = open(outfile, 'w')
    metric.write('File\tTotal')
    for case in ['DNA', 'AA', 'Gene']:
        for stat in ['Total', 'Single', 'Uniq', 'Average', 'Entropy', 'Clonality']:
            metric.write('\t'+case+'.'+stat)
    metric.write('\n')
    if type(datapath) == type(''):
        data = []
        for f in os.listdir(datapath):
            if not f.endswith('.txt.gz'):
                continue
            data.append((f, datapath+f))
    else:
        data = datapath
    for ele in data:
        f, p = ele[:2]
        print 'Process', f
        total, cc = get_rep(p)
        metric.write(f+'\t'+str(total))
        for d in cc:
            metric.write('\t'+'\t'.join([str(i) for i in stat_rep(d)]))
        metric.write('\n')

        # including an in-silico mixed case
        if '_CD4' in f:
            p8 = p.replace('_CD4','_CD8')
            total8, cc8 = get_rep(p8)
            metric.write(f.replace('_CD4','_Mix')+'\t'+str(total+total8))
            for d, d8 in zip(cc, cc8): # CD4 and CD8
                for i in d8:
                    if i in d:
                        d[i] = d[i] + d8[i]
                    else:
                        d[i] = d8[i]
                metric.write('\t'+'\t'.join([str(i) for i in stat_rep(d)]))
            metric.write('\n')

    metric.close()
    return

def run_vdjtools(datapath, workpath):
    meta = []
    for f in sorted(os.listdir(datapath)):
        if not f.endswith('.txt.gz'):
            continue
        meta.append((os.path.abspath(os.path.join(datapath,f)), f.replace('.txt.gz','')))
    
    os.chdir(workpath)
    meta = pd.DataFrame(meta, columns=['File', 'Name'])
    meta.to_csv('metadata.txt', sep='\t', index=False)

    if os.path.exists('file_out_clust'):
        print 'No need to run VDJtools'
        return
    os.system(args.vdjtools+' CalcDiversityStats -m metadata.txt VDJtools')
    os.system(args.vdjtools+' CalcPairwiseDistances -i aa -p -m metadata.txt VDJtools')
    os.system(args.vdjtools+' ClusterSamples -i aa -p VDJtools VDJtools')

## new codes for CD4 and CD8 TCR-seq
subject_id_map = None
def map_subject(id, ref='../cd4-cd8-batch/CD4_CD8_sorted_samples.csv'):
    global subject_id_map
    if subject_id_map is None:
        meta = pd.read_csv(ref, converters={i: str for i in range(0, 50)})
        meta = meta[['C1D1','C4D1','ZID']].drop_duplicates()
        print meta.head()
        subject_id_map = {}
        for row in meta.itertuples():
            subject_id_map[row.C1D1] = row.ZID
            subject_id_map[row.C4D1] = row.ZID
    return subject_id_map.get(id, 'Unknown')

def count_clones(datapath, clonefile, outfile):
    clones = pd.read_csv(clonefile)
    clones['PID'] = clones['PID'].astype(str)
    pos = set()
    neg = set()
    for row in clones.itertuples():
        if row.Type == 'Naive':
            pos.add((row.V, row.CDR3, row.J, row.PID))
        if row.Type == 'Memory':
            neg.add((row.V, row.CDR3, row.J, row.PID))
    print len(pos), len(neg)

    metric = open(outfile, 'w')
    metric.write('File\tZID\tTime\tCell.Type\tClones.Total\tClones.Expand.Naive\tClones.Expand.Memory\n')
    if type(datapath) == type(''):
        data = []
        for f in os.listdir(datapath):
            if not f.endswith('.txt.gz'):
                continue
            data.append((f, datapath+f))
    else:
        data = datapath
    for ele in data:
        f, p = ele[:2]
        keys = f.split('/')[-1].replace('.txt.gz','').split('_')
        if len(keys) == 2:
            PID, TYPE = keys
            TIME = 'C1D1'
        elif len(keys) == 3:
            PID, TIME, TYPE = keys
            if TIME == 'new':
                TIME = 'C1D1' # pre-ICB
        else:
            raise ValueError("Unknown keys "+'_'.join(keys))
        PID = map_subject(PID)
        print 'Process', PID, TIME, TYPE, f
        total, counts = get_rep(p)
        v_dna_j, v_aa_j, v_j = counts
        sp = [v_aa_j[i] for i in v_aa_j if tuple(list(i)+[PID]) in pos]
        sn = [v_aa_j[i] for i in v_aa_j if tuple(list(i)+[PID]) in neg]
        P = sum(sp)
        N = sum(sn)
        metric.write('\t'.join([f,PID,TIME,TYPE])+'\t'+str(total)+'\t'+str(P)+'\t'+str(N)+'\n')

        # including an in-silico mixed case
        if '_CD4' in f:
            p8 = p.replace('_CD4','_CD8')
            total8, counts8 = get_rep(p8)
            f = f.replace('_CD4','_Mix')
            v_dna_j8, v_aa_j8, v_j8 = counts
            sp8 = [v_aa_j8[i] for i in v_aa_j8 if tuple(list(i)+[PID]) in pos]
            sn8 = [v_aa_j8[i] for i in v_aa_j8 if tuple(list(i)+[PID]) in neg]
            P8 = sum(sp8)
            N8 = sum(sn8)
            metric.write('\t'.join([f,PID,TIME,'Mix'])+'\t'+
                    str(total+total8)+'\t'+str(P+P8)+'\t'+str(N+N8)+'\n')


    metric.close()

if __name__ == '__main__':
    save_metric(datapath=args.datapath, 
                outfile=os.path.join(args.datapath, '_rep_metric.txt'))
#    run_vdjtools(datapath=args.datapath, workpath=args.workpath)
    count_clones(datapath=args.datapath, 
                 clonefile=os.path.join(args.workpath, 'expanded_clones.csv.gz'),
                 outfile=os.path.join(args.datapath, '_rep_clones.txt'))

