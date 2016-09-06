import random


topiclist = set([])

word_dic = {}
alpha = 50
beta = 0.01

def load_data(filename):
    data = eval(open(filename).read())
    return data


def split_data(data, fold):
    tests = []
    trains = []
    i = 0
    for t in data:
        if i % fold == 0:
            tests.append(t)
        else:
            trains.append(t)
        i += 1
    return trains, tests


def init_train(trains):
    global topiclist, word_dic
    for t in trains:
        topics = t[0]
        words= t[1]
        for k in topics:
            topiclist.add(k)
        for w in words:
            if not word_dic.has_key(w):
                word_dic[w] = 1
            else:
                word_dic[w] += 1

    Ndk_ = []
    N_k_ = {k: 0 for k in topiclist}
    N_kt = {k: {w: 0 for w in word_dic.keys()} for k in topiclist}
    #random gamma for every word
    gammas = []
    for t in trains:
        Nk_ = {k: 0 for k in t[0]}
        d_gammas = []
        for w in t[1]:
            gamma = {}
            sump = 0
            for k in t[0]:
                gamma[k] = random.random()
                sump += gamma[k]
            for k in t[0]:
                gamma[k] /= sump
                Nk_[k] += gamma[k]
                N_k_[k] += gamma[k]
                N_kt[k][w] += gamma[k]
            d_gammas.append((w, gamma))
        Ndk_.append(Nk_)
        gammas.append(d_gammas)
    return gammas, Ndk_, N_kt, N_k_


def cvb0_train(gammas, Ndk_, N_kt, N_k_):
    global alpha, beta
    alpha_l = alpha / len(topiclist)
    v_beta = beta * len(word_dic.keys())
    for count in range(100):
        for d in range(len(gammas)):
            for w, ts in gammas[d]:
                g_sum = 0
                for k in ts:
                    Ndk_[d][k] -= ts[k]
                    N_k_[k] -= ts[k]
                    N_kt[k][w] -= ts[k]

                for k in ts:
                    g = (alpha_l + Ndk_[d][k]) * (beta + N_kt[k][w]) / (v_beta + N_k_[k])
                    ts[k] = g
                    g_sum += g

                for k in ts:
                    ts[k] /= g_sum
                    Ndk_[d][k] += ts[k]
                    N_k_[k] += ts[k]
                    N_kt[k][w] += ts[k]

    thita = []
    phi = {k: {w: 0 for w in word_dic.keys()} for k in topiclist}
    for d in range(len(gammas)):
        t = {}
        for w, ts in gammas[d]:

            for k in ts:
                t[k] = (Ndk_[d][k] + alpha_l) / (len(gammas[d][1]) + alpha_l * len(Ndk_[d].keys()))

        thita.append(t)
    for k in topiclist:
        for w in word_dic.keys():
            phi[k][w] = (beta + N_kt[k][w]) / (v_beta + N_k_[k])

    pl = {k: 0 for k in topiclist}
    for d in thita:
        for k in d:
            pl[k] += d[k]

    for k in pl:
        pl[k] /= len(thita)

    return pl, phi


def cvb0_test(pl, phi, i_tests):
    global word_dic, topiclist
    hit5, hit10, hit20 = 0.0, 0.0, 0.0
    tests = []
    for t in i_tests:
        ws = []
        for w in t[1]:
            if w in word_dic.keys():
                ws.append(w)
        tests.append((t[0], ws))
    for t in tests:
        h5, h10, h20 = False, False, False
        pt = {k: 0 for k in topiclist}
        for w in t[1]:
            g = {}
            sumg = 0
            for k in topiclist:
                g[k] = phi[k][w] * pl[k] * word_dic[w] / len(word_dic.keys())
                sumg += g[k]
            for k in topiclist:
                g[k] /= sumg
                pt[k] += g[k]
        SortL = sorted(pt.iteritems(), key=lambda d: d[1], reverse=True)
        i = 0
        for sl in SortL[0:20]:
            if sl[0] in t[0]:
                if i < 5:
                    h5 = True
                    h10 = True
                    h20 = True
                else:
                    if i < 10:
                        h10 = True
                        h20 = True
                    else:
                        if i < 20:
                            h20 = True
            i += 1
        if h5:
            hit5 += 1
        if h10:
            hit10 += 1
        if h20:
            hit20 += 1

    print hit5 / len(tests), '\t', hit10 / len(tests), '\t', hit20 / len(tests)


def main():
    data = load_data('AspectJ.txt')
    fold = 10
    for i in range(fold):
        trains, tests = split_data(data, fold)
        gammas, Ndk_, N_kt, N_k_ = init_train(trains)
        pl, phi = cvb0_train(gammas, Ndk_, N_kt, N_k_)
        cvb0_test(pl, phi, tests)

if __name__ == '__main__':
    main()