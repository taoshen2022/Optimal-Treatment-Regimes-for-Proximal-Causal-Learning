from Experiments import ITR_experiment
sc_num = 6
rep_num = 200

for s_num in range(sc_num):
    for r_num in range(rep_num):
        s, r1, r2, r3, r4 = ITR_experiment(s_num, r_num)