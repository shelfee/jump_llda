import random
import time
# import sys
from collections import Counter
import numpy as np
def gen_m(trans):
    n_k = {}
    trans_array = []
    i = 0
    for k in trans:
        n_k[i] = k
        i += 1
        trans_array.append([v for (key, v) in trans[k].items()])
    trans_m = np.mat(trans_array)
    topic_num = len(trans.keys())
    alpha = [1.0 / topic_num for i in xrange(topic_num)]
    alpha_m = np.mat(alpha)
    for i in range(100):
        alpha_m = np.dot(alpha_m, trans_m)
    m = {}
    i = 0
    l = list(np.asarray(alpha_m)[0])
    for t in l:
        m[n_k[i]] = t
        i += 1
    return m



def transfer_prop(file, files):
    dis_dic = {}
    for f in files:
        l1 = f.split('.')
        l2 = file.split('.')
        i = 0
        while i < len(l1) and i < len(l2):
            if l1[i] == l2[i]:
                i += 1
            else:
                break
        diff = 1.0 / 5 ** ((len(l1) - i + len(l2) - i) / 2.0 )
        if dis_dic.has_key(diff):
            dis_dic[diff].append(f)
        else:
            dis_dic[diff] = [f]
    props = {}
    sump = 0
    for k in dis_dic.keys():
        for t in dis_dic[k]:
            props[t] = k
            sump += k
    for k in props.keys():
        props[k] /= sump

    return props


def gen_trans(files):
    trans = {}
    for f in files:
        trans[f] = transfer_prop(f, files)

    return trans


def reverse_trans(trans):
    i = 0
    k_n = {}
    n_k = {}
    for k1 in trans.keys():
        k_n[k1] = i
        n_k[i] = k1
        i += 1
    arr = [[0 for kk in trans.keys()] for k in trans.keys()]
    for k1 in trans.keys():
        for k2 in trans[k1].keys():
            arr[k_n[k1]][k_n[k2]] = trans[k1][k2]
    a = np.array(arr)

    b = np.linalg.inv(a)


    r_trans = {k: {kk: 0 for kk in trans.keys()} for k in trans.keys()}
    for k1 in trans.keys():
        for k2 in trans[k1].keys():
            r_trans[k1][k2] = b[k_n[k1]][k_n[k2]]
    return r_trans

def multiple(trans, v):
    u = {k: 0.0 for k in trans.keys()}
    sumv = sum([v[k] for k in v.keys()])
    for k1 in u.keys():
        for k2 in v.keys():
            u[k1] += v[k2] * trans[k2][k1]
    sumu = sum([u[k] for k in u.keys()])
    for k in u.keys():
        u[k] = u[k] * sumv / sumu
    return v


def load_data():
    f_data = open('AspectJ.txt', 'r')
    r_data = eval(f_data.read())
    data = []
    for l in r_data:
        rec = {'tags': l[0], 'words': l[1]}
        data.append(rec)


    f_data.close()
    return data


def split_data(data, fold):
    tr_datas = []
    te_datas = []

    for f in xrange(fold):
        index = 0
        tr_data = []
        te_data = []

        for d in data:
            if (index % fold) != f:
                tr_data.append(d)
            else:
                te_data.append(d)
            index += 1

        print (len(tr_data), len(te_data))

        tr_datas.append(tr_data)
        te_datas.append(te_data)

    return tr_datas, te_datas


def llda_cvb0_init(data):
    t_vocab = []
    w_vocab = []
    for d in data:
        for t in d["tags"]:
            t_vocab.append(t)
        for w in d["words"]:
            w_vocab.append(w)

    print "word count: %d" % len(w_vocab)
    t_vocab = list(set(t_vocab))
    w_vocab = list(set(w_vocab))

    print "TV: %d, WV: %d" % (len(t_vocab), len(w_vocab))

    # random gamma for words
    gamma = {}
    d_index = 1
    for d in data:
        g = []
        tags = d["tags"]
        words = d["words"]

        for w in words:
            g_sum = 0
            ts = {}
            for t in tags:
                r = random.random()
                g_sum += r
                ts[t] = r

            for t in tags:
                ts[t] /= g_sum

            g.append((w, ts))

        gamma[d_index] = g

        d_index += 1

    return gamma, t_vocab, w_vocab


def calc_n0_n0all(gamma_d):
    sum_n0 = {}
    sum_n0_all = 0

    for w, ts in gamma_d:
        for t in ts:
            sum_n0[t] = sum_n0.get(t, 0) + ts[t]
            sum_n0_all += ts[t]

    return sum_n0, sum_n0_all


def calc_n1_n1all(gamma, t_vocab, w_vocab):
    sum_n1 = {}
    sum_n1_all = {}

    for t in t_vocab:
        sum_n1[t] = {}
        sum_n1_all[t] = 0
        for w in w_vocab:
            sum_n1[t][w] = 0

    for d in gamma:
        for w, ts in gamma[d]:
            for t in ts:
                sum_n1[t][w] += ts[t]
                sum_n1_all[t] += ts[t]

    return sum_n1, sum_n1_all


def llda_cvb0(gamma, t_vocab, w_vocab, alpha, eta, count):
    t_num = len(t_vocab)
    v_num = len(w_vocab)

    alpha_l = alpha/t_num
    veta = v_num*eta

    theta = {}
    phi = {}
    pl = {}
    for t in t_vocab:
        phi[t] = {}
        for w in w_vocab:
            phi[t][w] = 0

    # init theta, pl, and plm
    for d in gamma:
        w, ts = gamma[d][0]
        theta[d] = {}
        for t in ts:
            theta[d][t] = 0

    for t in t_vocab:
        pl[t] = 0

    # calc n1, n1_all, n2, n2_all
    n1, n1_all = calc_n1_n1all(gamma, t_vocab, w_vocab)

    for c in xrange(1, count+1):
        start_time = time.clock()

        for d in gamma:
            n0, n0_all = calc_n0_n0all(gamma[d])
            #r_trans = reverse_trans(gen_trans(gamma[d][0][1].keys()))

            #m0 = multiple(r_trans, n0)
            for w, ts in gamma[d]:
                g_sum = 0

                # remove current word, so need re-calc n0, n1, n1_all
                for t in ts:
                    n0[t] -= ts[t]
                    n1[t][w] -= ts[t]
                    n1_all[t] -= ts[t]
                    if n0[t] < 0 or n1[t][w] < 0 or n1_all[t] < 0:
                        print 1

                for t in ts:
                    g = (n0[t] + alpha_l)*(n1[t][w] + eta)/(n1_all[t] + veta)
                    ts[t] = g
                    g_sum += g

                for t in ts:
                    ts[t] /= g_sum

                # add current word, so need re-calc n1, n1_all, n2, n2_all again
                for t in ts:
                    n0[t] += ts[t]
                    n1[t][w] += ts[t]
                    n1_all[t] += ts[t]

        print "%03d, elapse: %d" % (c, time.clock() - start_time)

    for d in gamma:
        n0, n0_all = calc_n0_n0all(gamma[d])
        #r_trans = reverse_trans(gen_trans(gamma[d][0][1].keys()))
        #m0 = multiple(r_trans, n0)
        for t in theta[d]:
            theta[d][t] = (n0[t] + alpha_l) / (n0_all + len(theta[d]) * alpha_l)

    for d in theta:
        for t in theta[d]:
            pl[t] += theta[d][t]

    for t in pl:
        pl[t] /= len(theta)

    for t in phi:
        for w in w_vocab:
            phi[t][w] = (n1[t][w] + eta)/(n1_all[t] + veta)
    f = open('pz.txt', 'a')
    print >>f, pl
    f.close()
    return pl, phi


def llda_cvb0_train(data, alpha, eta, count):
    gamma, t_vocab, w_vocab = llda_cvb0_init(data)
    pl, phi = llda_cvb0(gamma, t_vocab, w_vocab, alpha, eta, count)
    return pl, phi

'''
def load_dict():
    f_dict = open('Eclipsemo1_dic.txt', 'r')
    dict = eval(f_dict.readline())
    f_dict.close()

    r_dict = {}
    for k in dict:
        r_dict[dict[k]] = k

    return r_dict
'''


def calc_pws(ws, t_vocab, pz, phi):
    pws = {}

    ws = list(set(ws))
    for w in ws:
        pws[w] = 0
        for t in t_vocab:
            pws[w] += phi[t][w]*pz[t]

    return pws


def calc_pwds(ws):
    pwds = {}

    ws_sum = len(ws)
    cws = Counter(ws)
    for w in cws:
        pwds[w] = cws[w]*1.0/ws_sum

    return pwds


def llda_test(data, pz, phi):
    l0 = phi.keys()[0]
    w_vocab = set(phi[l0].keys())
    t_vocab = pz.keys()
    trans = gen_trans(t_vocab)
    pz = multiple(trans, pz)
    r5 = 0
    r10 = 0
    p5 = 0
    p10 = 0
    h5 = 0
    h10 = 0
    r20, p20, h20 = 0, 0, 0


    #f_result = open(fn, 'w')

    for d in data:
        wss = d["words"]
        tags = d["tags"]

        #print len(ws)
        ws_temp = wss[:]
        ws = wss[:]
        for w in ws_temp:
            if w not in w_vocab:
                ws.remove(w)
        del ws_temp
        #print len(ws)

        pws = calc_pws(ws, t_vocab, pz, phi)
        pwds = calc_pwds(ws)
        # print pwds
        # print pws

        pzd = {}
        for t in t_vocab:
            pzd[t] = 0
            for w in ws:
                pzd[t] += phi[t][w]*pz[t]*pwds[w]/pws[w]

        del pwds
        del pws

        # test one record
        s_ptd = sorted(pzd.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        s_ptd = s_ptd[0:20]
        #print >> f_result, tags
        #print >> f_result, s_ptd

        count = 0
        c5 = 0
        c10 = 0
        c20 = 0
        for t, rd in s_ptd:
            if t in tags:
                if count < 5:
                    c5 += 1
                if count < 10:
                    c10 += 1
                c20 += 1
            count += 1

        
        if c5 > 0:
            h5 += 1
        if c10 > 0:
            h10 += 1
        if c20 > 0:
            h20 += 1

        r5 += float(c5)/len(tags)
        p5 += float(c5)/5
        r10 += float(c10)/len(tags)
        p10 += float(c10)/10
        r20 += float(c20)/len(tags)
        p20 += float(c20)/20

    data_num = len(data)
    r5 /= data_num
    r10 /= data_num
    r20 /= data_num
    p5 /= float(data_num)
    p10 /= float(data_num)
    p20 /= float(data_num)
    h5 /= float(data_num)
    h10 /= float(data_num)
    h20 /= float(data_num)

    #print >> f_result, "r5: %.5f, h5: %.5f, r10: %.5f, h10: %.5f" % (r5, h5, r10, h10)
    #print "r5: %.5f, h5: %.5f, r10: %.5f, h10: %.5f" % (r5, h5, r10, h10)

    #f_result.close()

    return r5, p5, r10, p10, h5, h10, r20, p20, h20


def main():
    data = load_data()
    fold = 10
    open('pz.txt', 'w').close()
    tr_datas, te_datas = split_data(data, fold)

    # fold_id = int(sys.argv[1])

    alpha = 50.0
    eta = 0.01

    #r_dict = load_dict()

    ar5, ap5, ar10, ap10, ah5, ah10 = 0, 0, 0, 0, 0, 0
    ar20, ap20, ah20 = 0,0,0
    avg_train = 0.0
    avg_predict = 0.0

    f_report = open('AspectJResult.txt', 'w')

    for i in xrange(10):
        train_start = time.time()
        pl, phi = llda_cvb0_train(tr_datas[i], alpha, eta, 100)
        train_end = time.time()
        train_cost = train_end - train_start
        print "fold", i, "train time is", train_cost
        print >> f_report, "fold", i, "train time is", train_cost

        predict_start = time.time()

        r5, p5, r10, p10, h5, h10, r20, p20, h20 = llda_test(te_datas[i], pl, phi)
        predict_end = time.time()
        predict_cost = predict_end - predict_start
        print "fold", i, "predict time is", predict_cost
        print >> f_report, "fold", i, "predict time is", predict_cost

        ar5 += r5
        ap5 += p5
        ar10 += r10
        ap10 += p10
        ah5 += h5
        ah10 += h10
        ar20 += r20
        ap20 += p20
        ah20 += h20
        
        avg_train += train_cost
        avg_predict += predict_cost

        print "fold-%02d:\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f" % (i, r5, p5,h5, r10, p10,h10,r20, p20, h20)
        print >> f_report, "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f" % (r5, p5,h5, r10, p10,h10,r20, p20, h20)

    ar5 /= 10
    ap5 /= 10
    ar10 /= 10
    ap10 /= 10
    ah5 /= 10
    ah10 /= 10
    ar20 /= 10
    ap20 /= 10
    ah20 /= 10

    avg_train /= 10
    avg_predict /= 10

    print "r5: %.5f, p5: %.5f, r10: %.5f, p10: %.5f, h5: %.5f, h10: %.5f \t%.5f\t%.5f\t%.5f" % (ar5, ap5, ar10, ap10, ah5, ah10, ar20, ap20, ah20)
    print "-------avg-------"
    print "avg time", avg_train, avg_predict

    print >> f_report, "---avg---"
    print >> f_report, "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f" % (ar5, ap5,ah5, ar10, ap10,  ah10, ar20,ap20,ah20)
    print >> f_report, "avg time", avg_train, avg_predict
    f_report.close()

if __name__ == '__main__':
    main()
