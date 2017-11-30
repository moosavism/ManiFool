import functools
import operator

import functions.helpers.general as g
import numpy as np
import torch
from functions.helpers.geodesic_distance import geodesic_distance
from torch.autograd import Variable

def get_output_label(output, input_ind=None):
    """
    Returns the output(maximum) label from the output variable
    :param output: Variable type output of a network
    :param input_ind: If the input had multiple images, the index of the requested output label
    :return: the output label of the network
    """
    if input_ind:
        _, k_org = torch.max(output.data[input_ind,:],0)
    else:
        _, k_org = torch.max(output.data,1)


    k_org = k_org[0]

    return k_org

def manifool_single_target(I_org, net, mode, target,
                           maxIter = 501,
                           step_sizes = [0.2],
                           gamma = 0.9,
                           batch_size=None,
                           cuda_on=True,
                           crop = None,
                           numerical = False,
                           verbose = 2):
    """
    Implements the binary ManiFool algorithm found in [1] for finding adversarial transformations for a given network,
    a given image and a target label on this network.

    Inputs:
    :Tensor I_org: the input image for the network(net)
    :Module net: network (input: image, output: values of activation BEFORE softmax)
    :string mode: transformation set (see general.py for available transformations)
    :int target: target label l from the network to get binary classifier g_l(x) = f_{k_org}(x)-f_l(x).
    :int maxIter: maximum number of iterations
    :[double] step_sizes: list of step sizes for line search
    :double gamma: momentum coefficient
    :int batch_size: batch size for line search
    :bool cuda_on: use GPU if True
    :(int,int) crop: tuple for the crop size. If it is a single integer, then (crop,crop) used as the size
    :bool numerical: use numerically or arithmetically calculated gradient
    :int verbose: the value of verbosity(0:no print, 1:only general, 2:everything)

    Output:
    :int k_I: the output label of the fooled image
    :int k_org: the output label of the input image
    :int it: number of iterations for finding an example
    :Transform tfm2: fooling transformation example
    :Tensor I_c: image transformed with tfm2

    If it == maxIter at the output, it indicates that the algorithm has failed to find a fooling transformation example.

    [1] C. Kanbak, S.-M. Moosavi-Dezfooli, P. Frossard,"Geometric robustness of deep networks:analysis and improvement",
    arxiv:1711.09115[cs],Nov. 2017. url: https://arxiv.org/abs/1711.09115
    """

    def get_gradient():
        """
        Calculates the gradient of the classifier(network) and projects it onto the tangential space of the image
        appearance manifold at current image.

        Output:
        :Tensor u_tar: projected gradient
        :double dist: relative distance to the boundary(for linearized classifier on the tangential space)
        """
        net.zero_grad()
        if x.grad is not None:
            x.grad.data.zero_()

        if cuda_on:
            gg = torch.cuda.FloatTensor([1,-1])
        else:
            gg = torch.FloatTensor([1,-1])

        output[:,[k_org,target]].backward(gg,retain_graph=True)
        w_tar = x.grad.data + 0
        w_tar = w_tar.view(n,1)

        try:
            u_tar = torch.gels(w_tar,J_n)[0]
            u_tar = u_tar[:J_n.size()[1]]
            u_tar.squeeze_()
            try:
                dist = abs(output.data[0, target] - output.data[0, k_org]) / (J_n.mm(u_tar)).norm()
            except ZeroDivisionError:
                dist = np.inf
        except:
            u_tar = None
            dist = np.inf

        return u_tar, dist


    def line_search(step_sizes, diff_pre):
        """
        Does a line search for given step sizes using the previous movement vector(v_pre) and the current projected
        gradient(u)

        Inputs:
        :[double] step_sizes: list of step sizes for line search
        :double diff_pre: value of g_l(x) at previous iteration

        Outputs:

        :double s: the chosen step size
        :double chosen_diff: the g_l(x) for step size s
        :Tensor I_chosen: the transformed image for step size s
        :int k_I: the label of I_chosen
        """
        if I_org.is_cuda:
            I_batch = torch.cuda.FloatTensor(torch.Size((batch_size,))+I_org.size())
        else:
            I_batch = torch.FloatTensor(torch.Size((batch_size,))+I_org.size())

        diff_max = -np.inf
        for ind in range(0,len(step_sizes),batch_size):

            # Generate transformed images
            for i,s in enumerate(step_sizes[ind:ind+batch_size]):
                v_t = s*u + gamma*v_pre

                tfm_ = tfm.compose(g.para2tfm(-v_t,mode,1))
                I_batch[i,:,:,:] = tfm_(I_org)

            if crop:
                I_c = g.center_crop_tensor(I_batch,crop)
            else:
                I_c = I_batch

            # Calculate g_l in batch and find the minimizing step size
            x = Variable(I_c, requires_grad = True)

            output = net(x)
            f_org = output.data[:,k_org]
            diff_curr = f_org - output.data[:,target]
            diff_x, s = torch.max((diff_pre - diff_curr),0)

            if diff_max < diff_x[0]:
                diff_max = diff_x[0]
                s_final = s + ind
                k_I = get_output_label(output,s[0])
                chosen_diff = diff_curr[s[0]]
                I_chosen = I_batch[s[0],:].clone()

        s = s_final[0]

        return step_sizes[s], chosen_diff, I_chosen, k_I

    #START OF MANIFOOL

    if not batch_size:
        batch_size = len(step_sizes)

    if cuda_on:
        net.cuda()
        I_org = I_org.cuda()

    I = I_org.clone()


    if crop:
        I_org_c = g.center_crop_tensor(I_org,crop+2)
        I_c = g.center_crop_tensor(I,crop)
    else:
        I_c = I

    n = functools.reduce(operator.mul, I_c.size(), 1)

    x = Variable(I_c.unsqueeze(0),requires_grad = True)

    output = net(x)

    k_org = get_output_label(output)

    k_I = k_org
    tau = g.init_param(mode)
    tfm = g.para2tfm(tau,mode,1)
    v_pre = tau
    if cuda_on:
        v_pre = v_pre.cuda()
    it = 0
    diff = 2*torch.max(output.data.abs())
    while k_org == k_I and it < maxIter:
        #Find Jacobian
        if crop:
            if numerical:
                J = g.jacobian(I_org_c,g.center_crop_tensor(I,crop+2),tfm,mode,interp_order=1)
            else:
                J = g.jacobian_from_gradient(g.center_crop_tensor(I,crop+2),mode)

            J = J[:,:,1:crop+1,1:crop+1]
        else:
            if numerical:
                J = g.jacobian(I_org,I,tfm,mode,interp_order=1)
            else:
                J = g.jacobian_from_gradient(I,mode)

        J_n = J.resize_(J.size()[0],n).transpose(0,1)

        # Choose min gradient on the manifold
        u, dist = get_gradient()

        # If dist is infinity (i.e. if u is perpendicular to the manifold) stop iterating as the alg. wont converge.
        if dist == np.inf:
            it = maxIter -1

        # Normalize the gradient
        if dist == 0.:
            dist += 1e-5
        u = u/u.norm()

        #Transform the image accordingly
        s, diff, I, k_I = line_search(step_sizes, diff)
        v_t = s*u + gamma*v_pre
        tfm.compose_(g.para2tfm(-v_t,mode,1))

        if crop:
            I_c = g.center_crop_tensor(I,crop)
        else:
            I_c = I

        x = Variable(I_c.unsqueeze(0), requires_grad = True)

        output = net(x)
        k_I = get_output_label(output)

        if I_c.norm() == 0:
            it = maxIter-1


        if verbose == 2:
            print(diff, dist, torch.max(s*u.abs()))

        it += 1
        v_pre = v_t.clone()

    if it == maxIter:
        return k_I, k_org, it, tfm, I_c

    # Normally, the update step of the movement direction(u) tries to maximize the decrease of the classifier, however,
    # in the last iteration, we want to get a point close to the boundary and not as low as we can get. Thus, the we do
    # a backtrack to get a smaller vector at the last step.

    tfm2 = tfm.compose(g.para2tfm(v_t,mode,1))#reverse the last step
    I = tfm2(I_org)
    if crop:
        I_c = g.center_crop_tensor(I,crop)
    else:
        I_c = I

    x = Variable(I_c.unsqueeze(0), requires_grad = True)

    output = net(x)
    _, k_I = torch.max(output.data,1)
    k_I = k_I[0]

    #Backtrack loop
    while k_org == k_I:
        v_t = 0.8*v_t
        tfm2 = tfm.compose(g.para2tfm(v_t,mode,1))
        I = tfm2(I_org)

        if crop:
            I_c = g.center_crop_tensor(I,crop)
        else:
            I_c = I

        x = Variable(I_c.unsqueeze(0), requires_grad = True)

        output = net(x)
        _, k_I = torch.max(output.data,1)
        k_I = k_I[0]

    if crop:
        I_c = g.center_crop_tensor(I,crop)
    else:
        I_c = I

    return k_I, k_org, it, tfm2, I_c

def manifool(I_org, net, mode,
             maxIter = 501,
             step_sizes = [0.2],
             gamma = 0.9,
             batch_size=None,
             cuda_on=True,
             crop = None,
             numerical = False,
             verbose = 1,
             geo_step = 0.05,
             N_c = 10):
    """
    Implements the mutliclass ManiFool algorithm found in [1] for finding adversarial transformations for a given
    network, a given image and a target label on this network.

    Inputs:
    :Tensor I_org: the input image for the network(net)
    :Module net: network (input: image, output: values of activation BEFORE softmax)
    :string mode: transformation set (see general.py for available transformations)
    :int maxIter: maximum number of iterations
    :[double] step_sizes: list of step sizes for line search
    :double gamma: momentum coefficient
    :int batch_size: batch size for line search
    :bool cuda_on: use GPU if True
    :(int,int) crop: tuple for the crop size. If it is a single integer, then (crop,crop) used as the size
    :bool numerical: use numerically or arithmetically calculated gradient
    :int verbose: the value of verbosity(0:no print, 1:only general, 2:everything)
    :double geo_step: step size for calculating geodesic distance
    :int N_c: number of classes for which the binary ManiFool is run

    Output:
    The output of the function is given as a dictionary. The outputs are labeled as:
    :int org_label: the original label of the image
    :int fooling_label: the label of the fooling image
    :int target: the target used for finding the fooling image
    :double geo_dist: the geodesic score of the output transformation
    :Transform tfm: the adversarial transformation
    :Tensor output_image: image transformed by the adversarial transformation

    If target == org_label at the output, it indicates that the algorithm has failed to find a fooling transformation
    example.

    [1] C. Kanbak, S.-M. Moosavi-Dezfooli, P. Frossard,"Geometric robustness of deep networks:analysis and improvement",
    arxiv:1711.09115[cs],Nov. 2017. url: https://arxiv.org/abs/1711.09115
    """

    out = {}

    if cuda_on:
        net.cuda()
        I_org = I_org.cuda()

    I = I_org.clone()

    if crop:
        I_c = g.center_crop_tensor(I,crop)
    else:
        I_c = I

    x = Variable(I_c.unsqueeze(0),requires_grad = True)

    output = net(x)
    _, inds = output.data.squeeze().sort(dim=0, descending=True)

    top_inds = inds[1:N_c+1] #choose top N_c labels for running binary ManiFool
    out['org_label'] = inds[0]

    min_geo_dist = np.inf

    # Main Loop
    for target in top_inds:
        if verbose:
            print('---------- Target = {} ----------'.format(target))

        # Run binary ManiFool
        k_f, k_org, it_f, tfm, II = manifool_single_target(
                                                    I_org,net,mode,target,
                                                    maxIter=maxIter,
                                                    step_sizes = step_sizes,
                                                    gamma = gamma,
                                                    batch_size = batch_size,
                                                    cuda_on=cuda_on,
                                                    crop = crop,
                                                    numerical= numerical,
                                                    verbose = verbose
                                                    )

        # If binary case is successful calculate the geodesic score and update the minimum accordingly
        if it_f < maxIter:
            geo_dist = geodesic_distance(I_org, geo_step, tfm.tform_matrix, mode)[0]/I_org.norm()

            if verbose:
                print("Net fooled.\nOriginal Output: {}\n"
                "New Output: {}\nIterations: {}".format(k_org, k_f, it_f))
                print('Geodesic Distance: {}'.format(geo_dist))

            if min_geo_dist > geo_dist:
                min_geo_dist = geo_dist
                min_tar = target
                min_out_label = k_f
                min_tfm = tfm
                min_fooled_im = II

        elif verbose:
            print("Max iteration reached before fooling")

    # Failure conditions
    if min_geo_dist == np.inf:
        min_geo_dist = np.inf
        min_tar = inds[0]
        min_out_label = None
        min_tfm = None
        min_fooled_im = None
        if verbose:
            print("ManiFool failed to find a fooling trasform")


    out.update({'fooling_label':min_out_label, 'target':min_tar,
                'geo_dist':min_geo_dist, 'tfm':min_tfm, 'output_image':min_fooled_im})

    return out
