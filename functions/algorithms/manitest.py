import functools
import operator
from math import pi

import functions.helpers.general as g
import numpy as np
import torch
from torch.autograd import Variable


def manitest(input_image, net, mode, maxIter=50000, lim = None, hs = None, cuda_on = True,
             stop_when_found=None, verbose=True):

    def list_index(a_list, inds):

        return [a_list[i] for i in inds]

    def group_chars(mode):
        if mode == 'rotation':
            hs = torch.Tensor([pi/20])
        elif mode == 'translation':
            hs = torch.Tensor([0.25,0.25])
        elif mode == 'rotation+scaling':
            hs = torch.Tensor([pi/20, 0.1])
        elif mode == 'rotation+translation':
            hs = torch.Tensor([pi/20,0.25,0.25])
        elif mode == 'scaling+translation':
            hs = torch.Tensor([0.1,0.25,0.25])
        elif mode == 'similarity':
            hs = torch.Tensor([pi/20,0.5,0.5,0.1])
        else:
            raise NameError('Wrong mode name entered')

        if cuda_on:
            hs.cuda()

        return hs

    def gen_simplices(cur_vec, cur_dim):
        nonlocal num_simpl
        nonlocal simpls
        if cur_dim == dimension_group + 1:
            if not cur_vec:
                return


            simpls.append(cur_vec)
            num_simpl = num_simpl + 1
            return


        if (n_vec[2*cur_dim - 2] == i or n_vec[2*cur_dim-1] == i):
            cur_vec = cur_vec + [i]
            gen_simplices(cur_vec, cur_dim + 1)
        else:
            gen_simplices(cur_vec, cur_dim + 1)

            if (n_vec[2*cur_dim - 2] != -1):
                cur_vec_l  = cur_vec + [n_vec[2*cur_dim - 2]]
                gen_simplices(cur_vec_l,cur_dim+1)

            if (n_vec[2*cur_dim - 1] != -1):
                cur_vec_r  = cur_vec + [n_vec[2*cur_dim - 1]]
                gen_simplices(cur_vec_r,cur_dim+1)

    def check_oob(coord):
        inside = 1
        for u in range(len(coord)):
            if (coord[u] > lim[u,1]+1e-8 or coord[u] < lim[u,0] - 1e-8):
                inside = 0
                break

        return inside

    def get_and_create_neighbours(cur_node):
        nonlocal id_max, coords, dist, visited, neighbours, W, ims
        #1)Generate coordinates of neighbouring nodes
        for l in range(dimension_group):

            #Generate coordinates
            coordsNeighbour1 = coords[cur_node].clone()
            coordsNeighbour1[l] += hs[l]

            coordsNeighbour2 = coords[cur_node].clone()
            coordsNeighbour2[l] -= hs[l]


            if check_oob(coordsNeighbour1):
                #Can we find a similar coordinate?
                dists = (torch.stack(coords,0)
                    - coordsNeighbour1.repeat(len(coords), 1)).abs().sum(dim=1)
                II1 = (dists < 1e-6).nonzero()
                if not II1.size():
                    id_max += 1
                    #create node: i) coords, ii)visited, iii)distance
                    coords.append(coordsNeighbour1)
                    dist.append(np.inf)
                    visited.append(0)
                    #Assing the NodeID to IDNeighbours
                    neighbours.append([-1]*2*dimension_group)
                    neighbours[cur_node][2*l] = id_max
                    #Do the reverse
                    neighbours[id_max][2*l+1] = cur_node
                    W.append(None)
                    ims.append([])
                else:
                    #Node already exists
                    neighbours[cur_node][2*l] = II1[0,0]
                    #Do the reverse
                    neighbours[II1[0,0]][2*l+1] = cur_node

            if check_oob(coordsNeighbour2):
                #Can we find a similar coordinate?
                dists = (torch.stack(coords,0)
                    - coordsNeighbour2.repeat(len(coords), 1)).abs().sum(dim=1)
                II2 = (dists < 1e-6).nonzero()
                if not II2.size():
                    id_max += 1
                    #create node: i) coords, ii)visited, iii)distance
                    coords.append(coordsNeighbour2)
                    dist.append(np.inf)
                    visited.append(0)
                    #Assing the NodeID to IDNeighbours
                    neighbours.append([-1]*2*dimension_group)
                    neighbours[cur_node][2*l+1] = id_max
                    #Do the reverse
                    neighbours[id_max][2*l] = cur_node
                    W.append(None)
                    ims.append([])
                else:
                    #Node already exists
                    neighbours[cur_node][2*l+1] = II2[0,0]
                    #Do the reverse
                    neighbours[II2[0,0]][2*l] = cur_node

    def generate_metric(cur_node):
        nonlocal ims, W

        tau = coords[cur_node]
        tfm = g.para2tfm(tau,mode,1)
        I = tfm(input_image)
        ims[cur_node] = I
        J = g.jacobian(input_image, I, tfm, mode, 1)
        J_n = J.resize_(J.size()[0],n)

        curW = J_n.mm(J_n.transpose(0,1))

        W[cur_node] = curW

    def evaluate_classifier(cur_node):
        nonlocal manitest_score, manitest_image, fooling_tfm, out_label

        x = Variable(ims[cur_node].unsqueeze(0))
        output = net(x)
        _, k_I = torch.max(output.data,1)
        pred_label = k_I[0]

        if pred_label != input_label:
            manitest_score = dist[cur_node]/input_image.norm()
            manitest_image = ims[cur_node]
            fooling_tfm = g.para2tfm(coords[cur_node],mode,1)
            out_label = pred_label
            return True

        return False


###
    e = g.init_param(mode)

    if cuda_on:
        net.cuda()
        input_image = input_image.cuda()
        e = e.cuda()

    dimension_group = e.size()[0]
    n = functools.reduce(operator.mul, input_image.size(), 1)

    stop_flag = False
    point_dists = None
    if stop_when_found is not None:
        stop_flag = True
        num_stopping_points = stop_when_found.size()[0]
        point_dists = torch.Tensor(num_stopping_points)
        remaining_points = num_stopping_points

    if hs is None:
        hs = group_chars(mode)

    if lim is None:
        lim = np.zeros((dimension_group,2))
        lim[:,0] = -np.inf
        lim[:,1] = np.inf

    dist = [0.0];
    visited =[0];
    coords = [e];
    ims = [input_image]
    W = [None]

    id_max = 0;
    neighbours = [[-1]*2*dimension_group];

    #Generate input label
    x = Variable(input_image.unsqueeze(0))
    output = net(x)
    _, k_I = torch.max(output.data,1)
    input_label = k_I[0]

    #Output Variables
    manitest_score = np.inf
    manitest_image = input_image.clone()
    fooling_tfm = e
    out_label = input_label

    for k in range(maxIter):

        if k%100== 0 and verbose:
            print('>> k = {}'.format(k))

        tmp_vec = np.array(dist[0:id_max+1])#copy the list
        tmp_vec[np.asarray(visited) == 1] = np.inf
        i = np.argmin(tmp_vec)
        visited[i] = 1

        #evaluate the classifier and check if it is fooled
        if stop_flag:
            dists = torch.norm(coords[i].repeat(num_stopping_points,1)-stop_when_found,2,1)
            if dists.min()<1e-6:
                _, ind = torch.min(dists,0)
                point_dists[ind[0,0]] = dist[i]
                remaining_points -= 1


                if remaining_points == 0:
                    break

        elif evaluate_classifier(i):
            break


        get_and_create_neighbours(i);

        for j in neighbours[i]:

            if j == -1:
                continue

            #Consider unknown neighbours only
            if visited[j]:
                continue

            #Look at the neighbours of j (vector of size 2*dimension_group)
            n_vec = neighbours[j]

            num_simpl = 1
            simpls = []
            gen_simplices([],1)

            if W[j] is None:
                generate_metric(j)

            for j_ in range(num_simpl-1):
                X = torch.stack(list_index(coords,simpls[j_])) - coords[j].repeat(len(simpls[j_]),1)
                if cuda_on:
                    v = torch.cuda.FloatTensor(list_index(dist,simpls[j_])).unsqueeze(1)
                    one_vector = torch.ones(v.size()).cuda()
                else:
                    v = torch.FloatTensor(list_index(dist,simpls[j_])).unsqueeze(1)
                    one_vector = torch.ones(v.size())


                M_prime = (X.mm(W[j]).mm(X.transpose(0,1)))
                try:
                    invM_prime_v, _ = torch.gesv(v,M_prime)
                except:
                    invM_prime_v = v*np.inf
                try:
                    invM_prime_1, _ = torch.gesv(one_vector,M_prime)
                except:
                    invM_prime_1 = one_vector*np.inf
                invM_prime_v.transpose_(0,1)
                # one_vector.squeeze_()
                # v.squeeze_()

                #Solve second order equation
                # dz^2 * one_vector' * invM_prime * one_vector
                # - 2 * dz * one_vector' * invM_prime * v + v' * invM_prime * v - 1
                Delta = (invM_prime_v.sum())**2 - invM_prime_1.sum()*(invM_prime_v.mm(v) - 1 )
                Delta = Delta[0,0]

                if Delta >= 0:
                    #Compute solution
                    x_c = (invM_prime_v.sum()+np.sqrt(Delta))/invM_prime_1.sum()


                    #Test that it is not on the border of the simplex
                    te, _ = torch.gesv(x_c - v,M_prime)

                    if te.min() > 0:
                        dist[j] = min(dist[j], x_c)

    return manitest_score, manitest_image, fooling_tfm, dist, coords, input_label, out_label, point_dists, k
