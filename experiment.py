import numpy as np
from tqdm import tqdm
import heapq as hq
from collections import defaultdict
import time

obstacles = set()
A_heuistic_mem = {}

def possibleMoves(fieldX, fieldY, state):
    moves = []
    x, y = state
    if x + 1 < fieldX and (x+1,y) not in obstacles:
        moves.append(True)
    else:
        moves.append(False)
    if y + 1 < fieldY and (x,y+1) not in obstacles:
        moves.append(True)
    else:
        moves.append(False)
    if x - 1 >= 0 and (x-1,y) not in obstacles:
        moves.append(True)
    else:
        moves.append(False)
    if y - 1 >= 0 and (x,y-1) not in obstacles:
        moves.append(True)
    else:
        moves.append(False)
    return np.array(moves)

def heuristic_init_1(finish):
    heuristic = [[[float("inf")]*Y for i in range(X)] for j in range(len(finish))]
    for n,f in enumerate(finish):
        t = 1
        cur = [f]
        heuristic[n][f[0]][f[1]] = 0
        while cur:
            new = []
            for state in cur:
                for nxt_x,nxt_y in basic_moves(state)[1:]:
                    if heuristic[n][nxt_x][nxt_y]==float("inf"):
                        heuristic[n][nxt_x][nxt_y] = t
                        new.append((nxt_x,nxt_y))
            t+=1
            cur = new
    return heuristic

def Mheuristic_init_1(finish):
    heuristic = [[[(float("inf"), None)]*Y for i in range(X)] for j in range(len(finish))]
    for n,f in enumerate(finish):
        t = 1
        cur = [f]
        heuristic[n][f[0]][f[1]] = (0, finish[n])
        while cur:
            new = []
            for state in cur:
                for nxt_x,nxt_y in basic_moves(state)[1:]:
                    if heuristic[n][nxt_x][nxt_y][0]==float("inf"):
                        heuristic[n][nxt_x][nxt_y] = (t, state)
                        new.append((nxt_x,nxt_y))
            t+=1
            cur = new
    return heuristic

def Mheuristic_1(state, heuristic):
    return max([heuristic[n][x][y][0] for n,(x,y) in enumerate(state)])

def heuristic_1(state, heuristic_table):
    return max([heuristic_table[n][x][y] for n,(x,y) in enumerate(state)])


def basic_moves(state):
    moves = [state]
    x, y = state
    if x + 1 < X and (x+1,y) not in obstacles:
        moves.append((x+1,y))
    if y + 1 < Y and (x,y+1) not in obstacles:
        moves.append((x,y+1))
    if x - 1 >= 0 and (x-1,y) not in obstacles:
        moves.append((x-1,y))
    if y - 1 >= 0 and (x,y-1) not in obstacles:
        moves.append((x,y-1))
    return moves

def _moves_1(pos,i = 0):
    if i == len(pos):
        return [[]]
    ans = []
    for move in basic_moves(pos[i]):
        for nxt in _moves_1(pos,i+1):
            if move not in nxt:
                ans.append([move]+nxt)
    return ans

def moves_1(pos):
    ans = []
    for i in _moves_1(pos)[1:]:

        for a in range(len(i)):
            if any([(i[a][0],i[a][1]) == (pos[b][0],pos[b][1]) for b in range(a+1, len(i))]):
                break
        else:
            ans.append((1, tuple(i)))
    return ans

class Node:
    def __init__(self, val, parent = None):
        self.val = val
        self.parent = parent

def beckprop(v_k, C_l, f, heap, C, node, heuristic,heuristic_table):
    # if v_k:

    if any([i not in C[v_k] for i in C_l]):
        C[v_k].update(C_l)
        if v_k not in heap:
            h = heuristic(v_k,heuristic_table)
            hq.heappush(heap, (h+f[v_k],-f[v_k],v_k, node))
        # for v_m in back_set[v_k]:
        if node.parent:
            beckprop(node.parent.val, C[v_k], f, heap, C, node.parent,heuristic,heuristic_table)

def _Mmoves_1(pos,paired, i = 0):
    if i == len(paired):
        return [[]]
    ans = []
    for move in basic_moves(pos[paired[i]]):
        for nxt in _Mmoves_1(pos,paired,i+1):
            if move not in nxt:
                ans.append([move]+nxt)

    return ans

def Mmoves_1(pos, paired):
    if paired:
        ans = []
        for i in _Mmoves_1(pos, paired)[1:]:
            for a in range(len(i)):
                if any([(i[a][0], i[a][1]) == (pos[paired[b]][0], pos[paired[b]][1]) for b in range(a + 1, len(i))]):
                    break
            else:
                ans.append((1, i))
        return ans
    else:
        return [(1, [])]

def Mstar_1(start, finish, moves, heuristic):
    n = len(start)
    # back_set = defaultdict(set)
    C = defaultdict(set)
    heuristic_table = Mheuristic_init_1(finish)
    f = defaultdict(int)
    f[start] = 0
    node = Node(start)
    heap = [(0, 0, start, node)]
    queries = 0
    while heap:
        queries +=1
        _, fk, pos, parent = hq.heappop(heap)
        fk *=-1
        if pos == finish:
            return Node(finish, parent), queries
        paired = sorted(C[pos])

        for cost, p_move in moves(pos, paired):
            t = 0
            move = []
            occupied = defaultdict(int)
            new_collisions = set()
            for i in range(n):
                if i in paired:
                    move.append(p_move[t])
                    t+=1
                else:
                    move.append(heuristic_table[i][pos[i][0]][pos[i][1]][1])
                if move[-1] in occupied.values():
                    new_collisions.update({i, occupied[move[-1]]})
                if not move[-1]:
                    return None, queries
                occupied[i] = move[-1]
            move = tuple(move)
            for a in range(n):
                for b in range(a + 1, n):
                    if (move[a][0], move[a][1]) == (pos[b][0], pos[b][1]):
                        new_collisions.update({a, b})
            C[move].update(new_collisions)
            if new_collisions:
                beckprop(pos, C[move], f, heap, C, parent,heuristic, heuristic_table)
            else:
                if move not in [i[2] for i in heap]:

                    h = heuristic(move,heuristic_table)
                    nxt_f = fk + cost
                    f[move] = nxt_f
                    hq.heappush(heap, (nxt_f + h, -nxt_f, move, Node(move, parent)))
    return None, queries


def heuristic_2(state, heuristic_table):
    if state in A_heuistic_mem:
        if A_heuistic_mem[state] <= max([heuristic_table[n][x][y] for n,(x,y) in enumerate(state)]):
            del A_heuistic_mem[state]
        else:
            return A_heuistic_mem[state]
    return max([heuristic_table[n][x][y] for n,(x,y) in enumerate(state)])

def Astar_coupled2(start, finish, moves, heuristic ):
    heuristic_table = heuristic_init_1(finish)
    node = Node(start)
    heap = [(0,0,start, node)]
    visited = {start}
    queries = 0
    while heap:
        queries+=1
        _, f, pos, parent = hq.heappop(heap)
        oldh = _+f
        f*=-1
        newh = float("inf")
        for cost, nxt in moves(pos):
            if nxt in visited:
                continue
            if nxt == finish:
                return Node(finish, parent), queries
            visited.add(nxt)
            h = heuristic(nxt, heuristic_table)
            nxt_f = f+cost
            tmp = h+cost
            if tmp<newh:
                newh = tmp
            hq.heappush(heap, (nxt_f + h, -nxt_f, nxt, Node(nxt, parent)))
        if newh>oldh:
            A_heuistic_mem[pos] = newh
            while parent.parent:
                parent = parent.parent
                oldh = heuristic(parent.val, heuristic_table)
                newh = float("inf")
                for cost, nxt in moves(parent.val):
                    h = heuristic(nxt, heuristic_table)
                    tmp = h + cost
                    if tmp < newh:
                        newh = tmp
                if newh>oldh:
                    A_heuistic_mem[pos] = newh
                else:
                    break
    return None, queries

def Astar_coupled(start, finish, moves, heuristic):
    heuristic_table = heuristic_init_1(finish)
    node = Node(start)
    heap = [(0,0,start, node)]
    visited = {start}
    queries = 0
    while heap:
        queries+=1
        _, f, pos, parent = hq.heappop(heap)
        f*=-1
        for cost, nxt in moves(pos):
            if nxt in visited:
                continue
            if nxt == finish:
                return Node(finish, parent), queries
            visited.add(nxt)
            h = heuristic(nxt, heuristic_table)
            nxt_f = f+cost
            hq.heappush(heap, (nxt_f+h, -nxt_f, nxt, Node(nxt, parent)))
    return None, queries




if __name__ == "__main__":
    np.random.seed(123456)
    for size in [16,23,48,64]:
        runs = 100
        X = size
        Y = size
        no_tiles = X * Y
        start = ((1, 1), (X - 2, Y - 2))
        finish = ((X - 1, Y - 1), (0, 0))
        data = []
        for no_run in tqdm(range(runs)):
            _heuistic_mem = {}
            no_new = no_tiles//50
            not_obtacles = {(i,j) for i in range(X) for j in range(Y)}.difference(list(start)+list(finish))
            not_obtacles = np.array(list(not_obtacles))
            obtacles = set()
            for j in range(13):
                np.random.shuffle(not_obtacles)
                new_obstacles = not_obtacles[:no_new]
                obtacles.update([tuple(obst) for obst in new_obstacles])
                not_obtacles = not_obtacles[no_new:]
                pers = len(obtacles)/no_tiles*100
                t = time.time()
                node, q = Astar_coupled(start, finish, moves_1, heuristic_1)
                Astar_time = time.time() - t
                data.append(["Coupled A*", pers, Astar_time])
                t = time.time()
                node, q = Astar_coupled2(start, finish, moves_1, heuristic_2)
                Astar2_time = time.time() - t
                data.append(["Modified Coupled A*", pers, Astar2_time])
                t = time.time()
                node, q = Mstar_1(start, finish, Mmoves_1, Mheuristic_1)
                Mstar_time = time.time() - t
                data.append(["M*", pers, Mstar_time])
        data = np.array(data)
        # print(data)
        np.save(f"./experiments/{size}x{size}.npy", data)