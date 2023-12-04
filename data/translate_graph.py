gf = '/home/hadoop/wzy/ADSampling/results/deep/SIMD_deep_kGraph_K100_perform_variance.log_deepquery4023'
import re
gf2 = '/home/hadoop/wzy/ADSampling/results/deep/SIMD_deep_kGraph_K100_perform_variance.log_deepquery4023.csv'
with open(gf, 'r') as f:
    lines = f.readlines()
    i = 0
    with open(gf2, 'w') as f2:
        start = False
        for line in lines:
            if line[0] == '#':
                start = True
                continue
            if start and len(line) > 50:
                line = line.strip()
                eles = re.split(r'\s+', line)
                f2.write(f'{int(eles[5])},{int(eles[7])}, {float(eles[6])}\n')
print(f'save to {gf2}')
