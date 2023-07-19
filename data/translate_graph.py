gf = '/home/hadoop/wzy/ADSampling/results/glove1.2m/glove1.2m_ef500_M32.graph'
gf2 = '/home/hadoop/wzy/ADSampling/results/glove1.2m/glove1.2m_ef500_M32_graph.csv'
with open(gf, 'r') as f:
    lines = f.readlines()
    i = 0
    with open(gf2, 'w') as f2:
        for line in lines:
            line = line.strip()
            pos = line.find(":")
            line = line[pos+1:]
            line = line.split(',')
            for ele in line:
                if len(ele) > 0:
                    f2.write(f"{i},{int(ele)}\n")
            i += 1
    