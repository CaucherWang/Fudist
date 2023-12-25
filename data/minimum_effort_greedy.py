from turtle import circle
from requests import get
from sklearn.utils import deprecated
from utils import *
from queue import PriorityQueue
import os
from unionfind import UnionFind
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'


fig = plt.figure(figsize=(6,4.8))

    
def get_density_scatter(k_occur, lid):
    x = k_occur
    y = lid
    # x = np.log2(x)
    # y = np.array([np.log2(i) if i > 0 else i for i in y])
    # y = np.log(n_rknn)
    print(np.max(x), np.max(y))

    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    def using_mpl_scatter_density(figure, x, y):
        ax = figure.add_subplot(1, 1, 1, projection='scatter_density')
        density = ax.scatter_density(x, y, cmap=white_viridis)
        figure.colorbar(density, label='#points per pixel')

    fig = plt.figure(figsize=(8,6))
    using_mpl_scatter_density(fig, x, y)
    # plt.xlabel('k-occurrence (450)')
    # plt.xlabel('Query difficulty')
    plt.xlabel('ME_GREEDY')
    # plt.ylabel('Query performance (recall@50=0.96)')
    plt.ylabel('NDC')
    # plt.ylabel('local intrinsic dimensionality')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    plt.xlim(0, 1600)
    # plt.ylim(0, 25000)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    plt.savefig(f'./figures/{dataset}/{dataset}-query-difficulty.png')
    print(f'save to figure ./figures/{dataset}/{dataset}-query-difficulty.png')


def plot_mahalanobis_distance_distribution(X, Q, dataset):
    ma_dist = get_mahalanobis_distance(X, Q)
    # randomly sample 10,000 base vectors
    S = np.random.choice(X.shape[0], 10000, replace=False)
    base_ma_dist = get_mahalanobis_distance(X, X[S])
    
    plt.hist(base_ma_dist, bins=50, edgecolor='black', label='base', color='orange')
    plt.hist(ma_dist, bins=50, edgecolor='black', label='query', color='steelblue')
    plt.xlabel('Mahalanobis distance')
    plt.ylabel('number of points')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title(f'{dataset} Mahalanobis distance distribution')
    plt.savefig(f'./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')
    print(f'save to file ./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')

    
def resolve_performance_variance_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ndcs = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            ndcs = line[:-1].split(',')
            ndcs = [int(x) for x in ndcs]
        return ndcs
    
    
def update_reachable(G, X, V, unsettled):
    '''
    Only points in Y are considered.
    Check whether any two points in X are reachable on G.
    '''
    set_X = set(X)
    set_V = set(V)
    successed = set()
    for node in unsettled:
        visited = set()
        visited_X = set()
        stack = [node]
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            if vertex in set_X:
                visited_X.add(vertex)
                
            for v in G[vertex]:
                if v in set_X:
                    visited_X.add(v)
                if v not in visited and v in set_V:
                    stack.append(v)
            
            if len(visited_X) == len(set_X):
                successed.add(node)
                break

        if len(visited_X) < len(set_X):
            unsettled -= successed
            return
            
    unsettled.clear()

def get_reachable(G, q, V):
    '''
    Only points in Y are considered.
    Check whether any two points in X are reachable on G.
    '''
    set_V = set(V)
    visited = set()
    visited_V = set()
    visited_V.add(q)
    stack = [q]
    while stack:
        vertex = stack.pop()
        
            
        for v in G[vertex]:
            if v in set_V:
                visited_V.add(v)
                if v not in visited:
                    stack.append(v)
                    visited.add(vertex)
        
        if len(visited_V) == len(set_V):
            break

    return visited_V

# deprecated
def get_me_greedy(G, KNN, GT_list, revG):
    K = KNN.shape[0]
    reachable_tbl = np.zeros((GT_list.shape[0], GT_list.shape[0]), dtype=np.int32)
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_reachable_tbl():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                reachable_tbl[i][id_to_index[point]] = 1
            
    def check_settled():
        success = set()
        for index in unsettled_knn:
            flag = True
            for i in range(K):
                if reachable_tbl[index][i] == 0:
                    flag = False
                    break
            if flag:
                success.add(index)
        for index in success:
            unsettled_knn.remove(index)
            
    init_reachable_tbl()
    unsettled_knn = set(np.arange(K))
    check_settled()
    
    # neighbors_set = {}
    # for point in KNN:
    #     neighbors_set[point] = set(G[point])
    
    cur_index = len(KNN)
    Y = set(KNN)
    while len(unsettled_knn) > 0:
        new_point = GT_list[cur_index]
        Y.add(new_point)
        # set new_point's reachable set
        out_neighbors = G[new_point]
        # neighbors_set[new_point] = set(out_neighbors)
        reachable_set = set()
        reachable_set.add(cur_index)
        for out_neighbor in out_neighbors:
            if out_neighbor not in Y:
                continue
            reachable_tbl[cur_index][id_to_index[out_neighbor]] = 1
            reachable_set.add(id_to_index[out_neighbor])
            for i in range(K):
                if reachable_tbl[cur_index][i] == 0 and reachable_tbl[id_to_index[out_neighbor]][i] == 1:
                    reachable_tbl[cur_index][i] = 1
                    reachable_set.add(i)
                                
        # find all the points that can reach new_point
        in_neigbors_index = set()
        for point in revG[new_point]:
            in_neigbors_index.add(id_to_index[point])
            for i in range(cur_index):
                if reachable_tbl[i][id_to_index[point]] == 1:
                    in_neigbors_index.add(i)
        for index in in_neigbors_index:
            for reach in reachable_set:
                reachable_tbl[index][reach] = 1
        
        check_settled()
        cur_index += 1
    
    if len(unsettled_knn) > 0:
        return GT_list.shape[0]
    return cur_index
            
def get_me_greedy_usg(G, KNN, GT_list, revG):
    K = KNN.shape[0]
    assert np.allclose(KNN, GT_list[:K])

    uf = UnionFind(np.arange(K))
    usg = {}    # key is index of point, value is the out-neighering RSCCs
    for i in range(K):
        usg[i] = set()
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_usg():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                index = id_to_index[point]
                if i == index:
                    continue
                usg[i].add(index)
    
    def find_circle_in_usg():
        # sort the points by indegree
        visited = set()
        circle = []
        
        def dfs(node, path, path_indexes):
            nonlocal circle
            visited.add(node)
            path.append(node)
            path_indexes[node] = len(path) - 1

            neighbor_set = usg[node].copy()
            for neighbor in neighbor_set:
                neighbor_root  = uf.find(neighbor)  # neighbors of usg may contain outdated points
                if neighbor_root not in usg[node]:
                    usg[node].remove(neighbor)
                    usg[node].add(neighbor_root)
                if neighbor_root in path_indexes:
                    cycle_start = path_indexes[neighbor_root]
                    cur_len = len(path) - cycle_start
                    if cur_len > len(circle):
                        circle = path[cycle_start:]
                elif neighbor_root not in visited:
                    dfs(neighbor_root, path, path_indexes)

            
            path_indexes.pop(node)
            path.pop()
                    
        for node in usg:
            if node not in visited:
                dfs(node, [], {})
        return circle
                
    def union_usg(circle):
        # only update the rscc involved, other rsccs are not affected
        # merge the circle
        first = circle[0]
        out_neighbors = usg[first]
        for i in range(1, len(circle)):
            uf.union(circle[i], first)  # who is the root depends on the size
            out_neighbors = out_neighbors.union(usg[circle[i]])
        
        out_neighbors = set([uf.find(x) for x in out_neighbors])
        root = uf.find(first)
        if root in out_neighbors:
            out_neighbors.remove(root)
        for i in range(len(circle)):
            if root != circle[i]:
                del usg[circle[i]]
        usg[root] = out_neighbors
        
                
    init_usg()
    while True:
        circle = find_circle_in_usg()
        assert len(circle) != 1
        if len(circle) <= 0:
            break
        union_usg(circle)
    
    cur_knn_components = set()
    for i in range(K):
        cur_knn_components.add(uf.find(i))
    if len(cur_knn_components) == 1:
        return K
    
    Y = set(KNN)
        
    for i in range(K, GT_list.shape[0]):
        new_point = GT_list[i]
        out_neighbors = set(G[new_point]).intersection(Y)
        in_neighbors = set(revG[new_point]).intersection(Y)
        Y.add(new_point)
        out_neighbor_roots = set([uf.find(id_to_index[x]) for x in out_neighbors])
        in_neighbor_roots = set([uf.find(id_to_index[x]) for x in in_neighbors])
        uf.add(i)
        usg[i] = out_neighbor_roots.copy()
        intersect = out_neighbor_roots.intersection(in_neighbor_roots)
        if len(intersect) > 0:
            union_usg(list(intersect))
        i_root = uf.find(i)
        for in_neighbor_root in in_neighbor_roots:
            if in_neighbor_root in usg:
                usg[in_neighbor_root].add(i_root)
        
        while True:
            cur_knn_components = set([uf.find(r) for r in cur_knn_components])
            if len(cur_knn_components) == 1:
                return i + 1

            circle = find_circle_in_usg()
            if len(circle) <= 0:
                break
            union_usg(circle)

    return GT_list.shape[0]
  
# as long as one point belonging to kNN can reach all other KNN on G_Y, 
def get_delta0_rigorous_point(G, KNN, GT_list, revG):
    K = KNN.shape[0]
    assert np.allclose(KNN, GT_list[:K])

    uf = UnionFind(np.arange(K))
    usg = {}    # key is index of point, value is the out-neighering RSCCs
    for i in range(K):
        usg[i] = set()
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_usg():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                index = id_to_index[point]
                if i == index:
                    continue
                usg[i].add(index)
    
    def find_circle_in_usg():
        # sort the points by indegree
        visited = set()
        circle = []
        
        def dfs(node, path, path_indexes):
            nonlocal circle
            visited.add(node)
            path.append(node)
            path_indexes[node] = len(path) - 1

            neighbor_set = usg[node].copy()
            for neighbor in neighbor_set:
                neighbor_root  = uf.find(neighbor)  # neighbors of usg may contain outdated points
                if neighbor_root != neighbor:
                    usg[node].remove(neighbor)
                    usg[node].add(neighbor_root)
                if neighbor_root in path_indexes:
                    cycle_start = path_indexes[neighbor_root]
                    cur_len = len(path) - cycle_start
                    if cur_len > len(circle):
                        circle = path[cycle_start:]
                elif neighbor_root not in visited:
                    dfs(neighbor_root, path, path_indexes)

            
            path_indexes.pop(node)
            path.pop()
                    
        for node in usg:
            if node not in visited:
                dfs(node, [], {})
        return circle
                
    def union_usg(circle):
        # only update the rscc involved, other rsccs are not affected
        # merge the circle
        first = circle[0]
        out_neighbors = usg[first]
        for i in range(1, len(circle)):
            uf.union(circle[i], first)  # who is the root depends on the size
            out_neighbors = out_neighbors.union(usg[circle[i]])
        
        out_neighbors = set([uf.find(x) for x in out_neighbors])
        root = uf.find(first)
        if root in out_neighbors:
            out_neighbors.remove(root)
        for i in range(len(circle)):
            if root != circle[i]:
                del usg[circle[i]]
        usg[root] = out_neighbors
        
                
    init_usg()
    while True:
        circle = find_circle_in_usg()
        assert len(circle) != 1
        if len(circle) <= 0:
            break
        union_usg(circle)
    
    
    # key is the root, value is the number of points of KNN in the rscc
    cur_knn_components = dict()
    for i in range(K):
        root = uf.find(i)
        if root not in cur_knn_components:
            cur_knn_components[root] = {i}
        else:
            cur_knn_components[root].add(i)
    
    def check_finished():
        nonlocal cur_knn_components
        # update cur_knn_components
        new_knn_components = dict()
        for root in cur_knn_components:
            new_root = uf.find(root)
            if new_root in new_knn_components:
                new_knn_components[new_root] = new_knn_components[new_root].union(cur_knn_components[root])
            else:
                new_knn_components[new_root] = cur_knn_components[root]
        cur_knn_components = new_knn_components
        # check whether we can finish
        def get_reachable_in_usg(root):
            reachable = set()
            visited = set()
            reachable = reachable.union(cur_knn_components[root])
            visited.add(root)
            stack = [root]
            while stack:
                node = stack.pop()
                neighbors = usg[node].copy()
                for neighbor in neighbors:
                    neighbor_root = uf.find(neighbor)
                    if neighbor_root != neighbor:
                        usg[node].remove(neighbor)
                        usg[node].add(neighbor_root)
                    if neighbor_root in visited:
                        continue
                    visited.add(neighbor_root)
                    stack.append(neighbor_root)
                    if neighbor_root in cur_knn_components:
                        reachable = reachable.union(cur_knn_components[neighbor_root])
            return reachable
        
        for root in cur_knn_components:
            cur_size = len(cur_knn_components[root])
            if cur_size == K:
                return True
            reachable = get_reachable_in_usg(root)
            if len(reachable) == K:
                return True
        return False
    
    if check_finished():
        return K
    

    Y = set(KNN)
        
    for i in range(K, GT_list.shape[0]):
        new_point = GT_list[i]
        out_neighbors = set(G[new_point]).intersection(Y)
        in_neighbors = set(revG[new_point]).intersection(Y)
        Y.add(new_point)
        out_neighbor_roots = set([uf.find(id_to_index[x]) for x in out_neighbors])
        in_neighbor_roots = set([uf.find(id_to_index[x]) for x in in_neighbors])
        uf.add(i)
        usg[i] = out_neighbor_roots.copy()
        intersect = out_neighbor_roots.intersection(in_neighbor_roots)
        if len(intersect) > 0:
            union_usg(list(intersect))
        i_root = uf.find(i)
        for in_neighbor_root in in_neighbor_roots:
            if in_neighbor_root in usg:
                usg[in_neighbor_root].add(i_root)
        
        while True:
            circle = find_circle_in_usg()
            if len(circle) <= 0:
                break
            union_usg(circle)
            
        if check_finished():
            return i + 1

    return GT_list.shape[0]

# as long as one point belonging to the max rcc can reach all other KNN on G_Y,
def get_delta0_max_knn_rscc_point(G, KNN, GT_list, revG):
    K = KNN.shape[0]
    assert np.allclose(KNN, GT_list[:K])

    uf = UnionFind(np.arange(K))
    usg = {}    # key is index of point, value is the out-neighering RSCCs
    for i in range(K):
        usg[i] = set()
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_usg():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                index = id_to_index[point]
                if i == index:
                    continue
                usg[i].add(index)
    
    def find_circle_in_usg():
        # sort the points by indegree
        visited = set()
        circle = []
        
        def dfs(node, path, path_indexes):
            nonlocal circle
            visited.add(node)
            path.append(node)
            path_indexes[node] = len(path) - 1

            neighbor_set = usg[node].copy()
            for neighbor in neighbor_set:
                neighbor_root  = uf.find(neighbor)  # neighbors of usg may contain outdated points
                if neighbor_root != neighbor:
                    usg[node].remove(neighbor)
                    usg[node].add(neighbor_root)
                if neighbor_root in path_indexes:
                    cycle_start = path_indexes[neighbor_root]
                    cur_len = len(path) - cycle_start
                    if cur_len > len(circle):
                        circle = path[cycle_start:]
                elif neighbor_root not in visited:
                    dfs(neighbor_root, path, path_indexes)

            
            path_indexes.pop(node)
            path.pop()
                    
        for node in usg:
            if node not in visited:
                dfs(node, [], {})
        return circle
                
    def union_usg(circle):
        # only update the rscc involved, other rsccs are not affected
        # merge the circle
        first = circle[0]
        out_neighbors = usg[first]
        for i in range(1, len(circle)):
            uf.union(circle[i], first)  # who is the root depends on the size
            out_neighbors = out_neighbors.union(usg[circle[i]])
        
        out_neighbors = set([uf.find(x) for x in out_neighbors])
        root = uf.find(first)
        if root in out_neighbors:
            out_neighbors.remove(root)
        for i in range(len(circle)):
            if root != circle[i]:
                del usg[circle[i]]
        usg[root] = out_neighbors
        
                
    init_usg()
    while True:
        circle = find_circle_in_usg()
        assert len(circle) != 1
        if len(circle) <= 0:
            break
        union_usg(circle)
    
    
    # key is the root, value is the number of points of KNN in the rscc
    cur_knn_components = dict()
    for i in range(K):
        root = uf.find(i)
        if root not in cur_knn_components:
            cur_knn_components[root] = {i}
        else:
            cur_knn_components[root].add(i)
    
    def check_finished():
        nonlocal cur_knn_components
        # update cur_knn_components
        new_knn_components = dict()
        for root in cur_knn_components:
            new_root = uf.find(root)
            if new_root in new_knn_components:
                new_knn_components[new_root] = new_knn_components[new_root].union(cur_knn_components[root])
            else:
                new_knn_components[new_root] = cur_knn_components[root]
        cur_knn_components = new_knn_components
        # check whether we can finish
        def get_reachable_in_usg(root):
            reachable = set()
            visited = set()
            reachable = reachable.union(cur_knn_components[root])
            visited.add(root)
            stack = [root]
            while stack:
                node = stack.pop()
                neighbors = usg[node].copy()
                for neighbor in neighbors:
                    neighbor_root = uf.find(neighbor)
                    if neighbor_root != neighbor:
                        usg[node].remove(neighbor)
                        usg[node].add(neighbor_root)
                    if neighbor_root in visited:
                        continue
                    visited.add(neighbor_root)
                    stack.append(neighbor_root)
                    if neighbor_root in cur_knn_components:
                        reachable = reachable.union(cur_knn_components[neighbor_root])
            return reachable
        
        max_size = 0
        max_knn_rscc_root = -1
        for root in cur_knn_components:
            cur_size = len(cur_knn_components[root])
            if cur_size == K:
                return True
            if cur_size > max_size:
                max_size = cur_size
                max_knn_rscc_root = root
                
        reachable = get_reachable_in_usg(max_knn_rscc_root)
        if len(reachable) == K:
            return True
        return False
    
    if check_finished():
        return K
    

    Y = set(KNN)
        
    for i in range(K, GT_list.shape[0]):
        new_point = GT_list[i]
        out_neighbors = set(G[new_point]).intersection(Y)
        in_neighbors = set(revG[new_point]).intersection(Y)
        Y.add(new_point)
        out_neighbor_roots = set([uf.find(id_to_index[x]) for x in out_neighbors])
        in_neighbor_roots = set([uf.find(id_to_index[x]) for x in in_neighbors])
        uf.add(i)
        usg[i] = out_neighbor_roots.copy()
        intersect = out_neighbor_roots.intersection(in_neighbor_roots)
        if len(intersect) > 0:
            union_usg(list(intersect))
        i_root = uf.find(i)
        for in_neighbor_root in in_neighbor_roots:
            if in_neighbor_root in usg:
                usg[in_neighbor_root].add(i_root)
        
        while True:
            circle = find_circle_in_usg()
            if len(circle) <= 0:
                break
            union_usg(circle)
            
        if check_finished():
            return i + 1

    return GT_list.shape[0]

# as long as one point belonging to the max rcc can reach all other KNN on G_Y,
def get_delta0_max_knn_rscc_point_recall(G, KNN, GT_list, revG, recall):
    K = KNN.shape[0]
    assert np.allclose(KNN, GT_list[:K])
    target = math.ceil(recall * K)

    uf = UnionFind(np.arange(K))
    usg = {}    # key is index of point, value is the out-neighering RSCCs
    for i in range(K):
        usg[i] = set()
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_usg():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                index = id_to_index[point]
                if i == index:
                    continue
                usg[i].add(index)
    
    def find_circle_in_usg():
        # sort the points by indegree
        visited = set()
        circle = []
        
        def dfs(node, path, path_indexes):
            nonlocal circle
            visited.add(node)
            path.append(node)
            path_indexes[node] = len(path) - 1

            neighbor_set = usg[node].copy()
            for neighbor in neighbor_set:
                neighbor_root  = uf.find(neighbor)  # neighbors of usg may contain outdated points
                if neighbor_root != neighbor:
                    usg[node].remove(neighbor)
                    usg[node].add(neighbor_root)
                if neighbor_root in path_indexes:
                    cycle_start = path_indexes[neighbor_root]
                    cur_len = len(path) - cycle_start
                    if cur_len > len(circle):
                        circle = path[cycle_start:]
                elif neighbor_root not in visited:
                    dfs(neighbor_root, path, path_indexes)

            
            path_indexes.pop(node)
            path.pop()
                    
        for node in usg:
            if node not in visited:
                dfs(node, [], {})
        return circle
                
    def union_usg(circle):
        # only update the rscc involved, other rsccs are not affected
        # merge the circle
        first = circle[0]
        out_neighbors = usg[first]
        for i in range(1, len(circle)):
            uf.union(circle[i], first)  # who is the root depends on the size
            out_neighbors = out_neighbors.union(usg[circle[i]])
        
        out_neighbors = set([uf.find(x) for x in out_neighbors])
        root = uf.find(first)
        if root in out_neighbors:
            out_neighbors.remove(root)
        for i in range(len(circle)):
            if root != circle[i]:
                del usg[circle[i]]
        usg[root] = out_neighbors
        
                
    init_usg()
    while True:
        circle = find_circle_in_usg()
        assert len(circle) != 1
        if len(circle) <= 0:
            break
        union_usg(circle)
    
    
    # key is the root, value is the number of points of KNN in the rscc
    cur_knn_components = dict()
    for i in range(K):
        root = uf.find(i)
        if root not in cur_knn_components:
            cur_knn_components[root] = {i}
        else:
            cur_knn_components[root].add(i)
    
    def check_finished():
        nonlocal cur_knn_components
        # update cur_knn_components
        new_knn_components = dict()
        for root in cur_knn_components:
            new_root = uf.find(root)
            if new_root in new_knn_components:
                new_knn_components[new_root] = new_knn_components[new_root].union(cur_knn_components[root])
            else:
                new_knn_components[new_root] = cur_knn_components[root]
        cur_knn_components = new_knn_components
        # check whether we can finish
        def get_reachable_in_usg(root):
            reachable = set()
            visited = set()
            reachable = reachable.union(cur_knn_components[root])
            visited.add(root)
            stack = [root]
            while stack:
                node = stack.pop()
                neighbors = usg[node].copy()
                for neighbor in neighbors:
                    neighbor_root = uf.find(neighbor)
                    if neighbor_root != neighbor:
                        usg[node].remove(neighbor)
                        usg[node].add(neighbor_root)
                    if neighbor_root in visited:
                        continue
                    visited.add(neighbor_root)
                    stack.append(neighbor_root)
                    if neighbor_root in cur_knn_components:
                        reachable = reachable.union(cur_knn_components[neighbor_root])
            return reachable
        
        max_size = 0
        max_knn_rscc_root = -1
        for root in cur_knn_components:
            cur_size = len(cur_knn_components[root])
            if cur_size >= target:
                return True
            if cur_size > max_size:
                max_size = cur_size
                max_knn_rscc_root = root
                
        reachable = get_reachable_in_usg(max_knn_rscc_root)
        if len(reachable) >= target:
            return True
        return False
    
    if check_finished():
        return K
    

    Y = set(KNN)
        
    for i in range(K, GT_list.shape[0]):
        new_point = GT_list[i]
        out_neighbors = set(G[new_point]).intersection(Y)
        in_neighbors = set(revG[new_point]).intersection(Y)
        Y.add(new_point)
        out_neighbor_roots = set([uf.find(id_to_index[x]) for x in out_neighbors])
        in_neighbor_roots = set([uf.find(id_to_index[x]) for x in in_neighbors])
        uf.add(i)
        usg[i] = out_neighbor_roots.copy()
        intersect = out_neighbor_roots.intersection(in_neighbor_roots)
        if len(intersect) > 0:
            union_usg(list(intersect))
        i_root = uf.find(i)
        for in_neighbor_root in in_neighbor_roots:
            if in_neighbor_root in usg:
                usg[in_neighbor_root].add(i_root)
        
        while True:
            circle = find_circle_in_usg()
            if len(circle) <= 0:
                break
            union_usg(circle)
            
        if check_finished():
            return i + 1

    return GT_list.shape[0]

def get_delta0_max_knn_rscc_point_recall_prob(G, KNN, GT_list, revG, recall, prob):
    K = KNN.shape[0]
    assert np.allclose(KNN, GT_list[:K])
    target = math.ceil(recall * K)
    prob = math.ceil(prob * K)

    uf = UnionFind(np.arange(K))
    usg = {}    # key is index of point, value is the out-neighering RSCCs
    for i in range(K):
        usg[i] = set()
    id_to_index = {}
    for i in range(GT_list.shape[0]):
        id_to_index[GT_list[i]] = i
    
    def init_usg():
        V = set(KNN)
        for i in range(len(KNN)):
            # check whether other points in KNN is reachable
            visited_by_iNN = get_reachable(G, KNN[i], V)
            for point in visited_by_iNN:
                index = id_to_index[point]
                if i == index:
                    continue
                usg[i].add(index)
    
    def find_circle_in_usg():
        # sort the points by indegree
        visited = set()
        circle = []
        
        def dfs(node, path, path_indexes):
            nonlocal circle
            visited.add(node)
            path.append(node)
            path_indexes[node] = len(path) - 1

            neighbor_set = usg[node].copy()
            for neighbor in neighbor_set:
                neighbor_root  = uf.find(neighbor)  # neighbors of usg may contain outdated points
                if neighbor_root != neighbor:
                    usg[node].remove(neighbor)
                    usg[node].add(neighbor_root)
                if neighbor_root in path_indexes:
                    cycle_start = path_indexes[neighbor_root]
                    cur_len = len(path) - cycle_start
                    if cur_len > len(circle):
                        circle = path[cycle_start:]
                elif neighbor_root not in visited:
                    dfs(neighbor_root, path, path_indexes)

            
            path_indexes.pop(node)
            path.pop()
                    
        for node in usg:
            if node not in visited:
                dfs(node, [], {})
        return circle
                
    def union_usg(circle):
        # only update the rscc involved, other rsccs are not affected
        # merge the circle
        first = circle[0]
        out_neighbors = usg[first]
        for i in range(1, len(circle)):
            uf.union(circle[i], first)  # who is the root depends on the size
            out_neighbors = out_neighbors.union(usg[circle[i]])
        
        out_neighbors = set([uf.find(x) for x in out_neighbors])
        root = uf.find(first)
        if root in out_neighbors:
            out_neighbors.remove(root)
        for i in range(len(circle)):
            if root != circle[i]:
                del usg[circle[i]]
        usg[root] = out_neighbors
        
                
    init_usg()
    while True:
        circle = find_circle_in_usg()
        assert len(circle) != 1
        if len(circle) <= 0:
            break
        union_usg(circle)
    
    
    # key is the root, value is the points of KNN in the rscs
    cur_knn_components = dict()
    for i in range(K):
        root = uf.find(i)
        if root not in cur_knn_components:
            cur_knn_components[root] = {i}
        else:
            cur_knn_components[root].add(i)
    
    def check_finished():
        nonlocal cur_knn_components
        # update cur_knn_components
        new_knn_components = dict()
        for root in cur_knn_components:
            new_root = uf.find(root)
            if new_root in new_knn_components:
                new_knn_components[new_root] = new_knn_components[new_root].union(cur_knn_components[root])
            else:
                new_knn_components[new_root] = cur_knn_components[root]
        cur_knn_components = new_knn_components
        # check whether we can finish
        def get_reachable_in_usg(root):
            reachable = set()
            visited = set()
            reachable = reachable.union(cur_knn_components[root])
            visited.add(root)
            stack = [root]
            while stack:
                node = stack.pop()
                neighbors = usg[node].copy()
                for neighbor in neighbors:
                    neighbor_root = uf.find(neighbor)
                    if neighbor_root != neighbor:
                        usg[node].remove(neighbor)
                        usg[node].add(neighbor_root)
                    if neighbor_root in visited:
                        continue
                    visited.add(neighbor_root)
                    stack.append(neighbor_root)
                    if neighbor_root in cur_knn_components:
                        reachable = reachable.union(cur_knn_components[neighbor_root])
            return reachable
                        
        eligible_size = 0
        for root in cur_knn_components:
            reachable = get_reachable_in_usg(root)
            if len(reachable) >= target:
                eligible_size += len(cur_knn_components[root])
                
        if eligible_size >= prob:
            return True
        return False
    
    if check_finished():
        return K
    

    Y = set(KNN)
        
    for i in range(K, GT_list.shape[0]):
        new_point = GT_list[i]
        out_neighbors = set(G[new_point]).intersection(Y)
        in_neighbors = set(revG[new_point]).intersection(Y)
        Y.add(new_point)
        out_neighbor_roots = set([uf.find(id_to_index[x]) for x in out_neighbors])
        in_neighbor_roots = set([uf.find(id_to_index[x]) for x in in_neighbors])
        uf.add(i)
        usg[i] = out_neighbor_roots.copy()
        intersect = out_neighbor_roots.intersection(in_neighbor_roots)
        if len(intersect) > 0:
            union_usg(list(intersect))
        i_root = uf.find(i)
        for in_neighbor_root in in_neighbor_roots:
            if in_neighbor_root in usg: # maybe self-loop?
                usg[in_neighbor_root].add(i_root)
        
        while True:
            circle = find_circle_in_usg()
            if len(circle) <= 0:
                break
            union_usg(circle)
            
        if check_finished():
            return i + 1

    return GT_list.shape[0]

def get_me(G, GT_list, delta0_point, K, recall):
    ava_set = set(GT_list[:delta0_point])
    knn_set = set(GT_list[:K])
    union_ret_set = set()
    target = math.ceil(recall * K)
    for i in range(K):
        # BFS from i to find all kNN, record #points in paths
        visited = set()
        queue = deque()
        queue.append((GT_list[i], []))
        found_knn = set()
        ret_set = set()
        while queue:
            if len(found_knn) >= target:
                break 
            node, path = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in knn_set and node not in found_knn:
                found_knn.add(node)
                path.append(node)
                ret_set = ret_set.union(set(path))
            for neighbor in G[node]:
                if neighbor >=0 and neighbor in ava_set and neighbor not in visited:
                    queue.append((neighbor, path + [node]))
        union_ret_set = union_ret_set.union(ret_set)
        
    return union_ret_set

source = './data/'
result_source = './results/'
exp = 'kgraph'
dataset = 'deep'
GT_NUM_str = ""
idx_postfix = '_plain'
Kbuild =275
kgraph_od = '_500'
M=100
efConstruction = 2000
target_recall = 0.98
target_prob = 0.96
R = 32
L = 40
C = 500
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth{GT_NUM_str}.ivecs')
    GT_dist_path = os.path.join(source, dataset, f'{dataset}_groundtruth_dist.fvecs')
    GT_self_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth_2048.ivecs')
    KG = 499
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')
    ind_path2 = os.path.join(source, dataset, f'{dataset}_ind_32.ibin')
    inter_knn_dist_avg_path = os.path.join(source, dataset, f'{dataset}_inter_knn_dist_avg50.fbin')
    hnsw_ind_path = os.path.join(source, dataset, f'{dataset}_hnsw_ef{efConstruction}_M{M}_ind_{KG}.ibin')
    standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')
    standard_reversed_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}_reversed')
    nsg_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg')
    reversed_nsg_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg_reversed')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs')
    reversed_kgraph_path = os.path.join(source, dataset, f'{dataset}_K{Kbuild}_self_groundtruth.ivecs_reversed')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    ma_dist_path = os.path.join(source, dataset, f'{dataset}_ma_distance.fbin')
    ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_ma_base_distance.fbin')
    me_greedy_path = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin')
    me_greedy_path_opt = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin_usg')
    # me_max_out_path = os.path.join(source, dataset, f'{dataset}_me_max_out.ibin')
    delta0_point_path = os.path.join(source, dataset, f'{dataset}_delta0_point_K{Kbuild}.ibin')
    delta0_rigorous_point_path = os.path.join(source, dataset, f'{dataset}_delta0_rigorous_point_K{Kbuild}.ibin')
    delta0_max_knn_rscc_point_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_K{Kbuild}.ibin')
    self_delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_self_delta0_max_knn_rscc_point_recall{target_recall}_K{Kbuild}.ibin')
    kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    in_ds_kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log_in-dataset')
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain')
    query_performance_log_paths = []
    for i in range(3, 12):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance.log_plain_shuf{i}'))

    # in_ds_kgraph_query_performance = np.array(resolve_performance_variance_log(in_ds_kgraph_query_performance_log_path))
    # kgraph_query_performance = np.array(resolve_performance_variance_log(kgraph_query_performance_log_path))
    # query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
    # query_performances = [query_performance]
    # for i in range(9):
    #     query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
    # query_performances = np.array(query_performances)
    
    # query_performance_avg = np.sum(query_performances, axis=0) / len(query_performances)

    # recalls = []
    # with open(result_path, 'r') as f:
    #     lines = f.readlines()
    #     for i in range(3, len(lines), 7):
    #         line = lines[i].strip().split(',')[:-1]
    #         recalls.append(np.array([int(x) for x in line]))

    # low_recall_positions = []
    # for recall in recalls:
    #     low_recall_positions.append(np.where(recall < 40)[0])
    X = read_fvecs(base_path)
    # G = read_hnsw_index_aligned(index_path, X.shape[1])
    # G = read_hnsw_index_unaligned(index_path, X.shape[1])
    Q = read_fvecs(query_path)
    # Q = read_fvecs(query_path)
    GT = read_ivecs(GT_path)
    # GT_dist = read_fvecs(GT_dist_path)
    # GT_self = read_ivecs(GT_self_path)
    G = None
    revG = None
    if exp == 'kgraph':  
        # KGraph = read_ivecs(kgraph_path)
        # KGraph_clean = clean_kgraph(KGraph)
        # write_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs_clean'), KGraph_clean)

        KGraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth{kgraph_od}.ivecs_clean'))[:, :Kbuild]
        another_kgraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'))
        # diff = np.where(KGraph != another_kgraph)
        # print(diff[0])
        # real_Diff_index = []
        # i = 0
        # while i < diff[0].shape[0]:
        #     if diff[0][i] != diff[0][i+1]:
        #         real_Diff_index.append((diff[0][i], diff[1][i]))
        #         i += 1
        #     else:
        #         i += 2
        # print(real_Diff_index)
        revG = get_reversed_graph_list(KGraph)
        write_obj(reversed_kgraph_path, revG)
        # revG = read_obj(reversed_kgraph_path)
        exit(0)
        G = KGraph
        delta0_max_knn_rscc_point_recall_prob_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall}_prob{target_prob}_K{Kbuild}.ibin')
    elif exp == 'hnsw':
        hnsw = read_ibin(standard_hnsw_path)
        revG = get_reverse_graph(hnsw)
        write_obj(standard_reversed_hnsw_path, revG)
        revG = read_obj(standard_reversed_hnsw_path)
        exit(0)
        G = hnsw
        delta0_max_knn_rscc_point_recall_prob_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall}_prob{target_prob}_ef{efConstruction}_M{M}.ibin_hnsw{idx_postfix}')
    elif exp == 'nsg':
        ep, nsg = read_nsg(nsg_path)
        revG = get_reversed_graph_list(nsg)
        write_obj(reversed_nsg_path, revG)
        revG = read_obj(reversed_nsg_path)
        # exit(0)
        G = nsg
        delta0_max_knn_rscc_point_recall_prob_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall}_prob{target_prob}_L{L}_R{R}_C{C}.ibin_nsg')
    elif exp == 'mrng':
        
    else:
        print(exp)
        raise NotImplementedError
    # lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # q_lengths = get_query_length(X, Q)
    # n_rknn = read_ibin_simple(ind_path)
    # n_rknn2 = read_ibin_simple(ind_path2)
    # indegree = get_indegree_list(G)
    # write_ibin_simple(hnsw_ind_path, indegree)
    # indegree = read_ibin_simple(hnsw_ind_path)
    
    
    # hardness = []
    # for i in range(Q.shape[0]):
    #     if i % 100 == 0:
    #         print(f'{i} / {Q.shape[0]} ')
    #     query = Q[i]
    #     gt = GT[i][:50]
    #     knn_dist = GT_dist[i][49]
    #     flag = False
    #     unsettled = set(gt)
    #     for j in range(50, 10000):
    #         subset = GT[i][:j]
    #         update_reachable(KGraph, gt, subset, unsettled)
    #         if len(unsettled) == 0:
    #             flag = True
    #             hardness.append(j)
    #             break
    #     if flag == False:
    #         hardness.append(10000)
    #         print(i)
    
    # hardness = read_ibin_simple(self_delta0_max_knn_rscc_point_recall_path)
    hardness = np.zeros(Q.shape[0])
    # import concurrent.futures

    # def process(i):
    #     me_greedy = get_delta0_max_knn_rscc_point_recall(G, GT[i][:50], GT[i], revG, target_recall)
    #     hardness[i] = me_greedy

    # with concurrent.futures.ThreadPoolExecutor(24) as executor:
    #     list(tqdm(executor.map(process, range(Q.shape[0])), total=Q.shape[0]))
    # print(f'save to file {me_greedy_path_opt}')
    # with open(me_greedy_path_opt, 'w') as f:
    for i in range(Q.shape[0]):
        if i % 100 == 0:
            print(f'{i} / {Q.shape[0]}: {datetime.now()} ')
        # if i != 8464:
        #     continue
        # me_greedy = get_me_greedy_usg(KGraph, GT[i][:50], GT[i], revG)
        # me_greedy = get_delta0_rigorous_point(KGraph, GT[i][:50], GT[i], revG)
        me_greedy = get_delta0_max_knn_rscc_point_recall_prob(G, GT[i][:50], GT[i], revG, target_recall, target_prob)
        hardness[i] = me_greedy
        # f.write(f'{me_greedy},')
    # write_ibin_simple(me_greedy_path_opt, np.array(hardness))
    # write_ibin_simple(delta0_rigorous_point_path, np.array(hardness))
    write_ibin_simple(delta0_max_knn_rscc_point_recall_prob_path, np.array(hardness))
    # write_ibin_simple(self_delta0_max_knn_rscc_point_recall_path, np.array(hardness))
    
    # me_greedy = read_ibin_simple(me_greedy_path)
    # print(np.where(me_greedy != hardness))
    
    
    # get_density_scatter(pagerank, in_ds_kgraph_query_performance)
    
