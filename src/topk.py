import numpy as np
import torch as th

m = [10, 20, 30, 40]

mat_DTI_count = np.zeros([708, 1512])

for mi in m:
    for r in range(3):
        for f in range(10):
            filetxt = 'top40_' + 'r' + str(r + 1) + '_f' + str(f + 1) + '.csv'
            temp = np.loadtxt(filetxt, dtype=int, delimiter=',')
            for i in range(708):
                for j in range(40):
                    if temp[i][j] > 0:
                        mat_DTI_count[i, temp[i][j]] += 1

    # print(np.max(mat_DTI_count))

    mat_DTI = th.from_numpy(mat_DTI_count)

    top_values, top_indices = th.topk(mat_DTI, mi)
    topk_indices = top_indices.numpy()
    np.savetxt('top' + str(mi) + '.csv', topk_indices, fmt='%d', delimiter=',')

    dti_newly = np.loadtxt('../data/DTI_newly.csv', dtype=int, delimiter=',')

    list_hit = []
    count_hit = 0

    for i in range(len(dti_newly)):
        for j in range(mi):
            if topk_indices[dti_newly[i][0], j] == dti_newly[i][1]:
                list_hit.append([dti_newly[i][0], dti_newly[i][1]])
                count_hit = count_hit + 1

    print("m=", mi, ":", count_hit)
