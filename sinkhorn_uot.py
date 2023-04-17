import warnings

import numpy as np
import torch
# from ot.unbalanced import  sinkhorn_knopp_unbalanced
from torch_backend import get_torch_backend


def sinkhorn_knopp_unbalanced_torch(a, b, M, reg, reg_m, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False):
    '''
    torch version edited by QWC on 2022.10.27

    method : https://pythonot.github.io/gen_modules/ot.unbalanced.html
    implementation version with numpy : https://pythonot.github.io/_modules/ot/unbalanced.html#sinkhorn_knopp_unbalanced
    '''

    M, a, b = list_to_array(M, a, b)
    # a = a.double()
    # b = b.double()
    # M = M.double()
    device = a.device

    nx = get_torch_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M).to(device) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M).to(device) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, 1), type_as=M).to(device) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M).to(device) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M).to(device) / dim_a
        v = nx.ones(dim_b, type_as=M).to(device) / dim_b

    K = nx.exp(M / (-reg))

    fi = reg_m / (reg_m + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fi
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fi

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.
        )
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]


def PUOT1(a, b, M, reg, reg_m, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False, mass=None):
    '''
    torch version edited by QWC on 2022.10.27

    method : https://pythonot.github.io/gen_modules/ot.unbalanced.html
    implementation version with numpy : https://pythonot.github.io/_modules/ot/unbalanced.html#sinkhorn_knopp_unbalanced
    '''

    M, a, b = list_to_array(M, a, b)
    # a = a.double()
    # b = b.double()
    # M = M.double()
    device = a.device

    nx = get_torch_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M).to(device) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M).to(device) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, 1), type_as=M).to(device) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M).to(device) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M).to(device) / dim_a
        v = nx.ones(dim_b, type_as=M).to(device) / dim_b


    if mass is None:
        #m = np.min((np.sum(a), np.sum(b))) * 1.0
        mass = torch.min(a.sum(), b.sum()) * 1.0

    if mass < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if mass > torch.min(a.sum(), b.sum()):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    K = nx.exp(M / (-reg))
    K = K * (mass / K.sum())

    fi = reg_m / (reg_m + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fi
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fi

        scale = mass/ (u[:, None] * K * v[None, :]).sum()
        scale = scale**0.5
        u *= scale 
        v *= scale 

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.
        )
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]



def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]


def sinkhorn_stabilized_unbalanced_torch(a, b, M, reg, reg_m,  numItermax=1000, tau=1e5,
                                   stopThr=1e-6, verbose=False, log=False):
    r"""
    Solve the entropic regularization unbalanced optimal transport
    problem and return the loss

    The function solves the following optimization problem using log-domain
    stabilization as proposed in :ref:`[10] <references-sinkhorn-stabilized-unbalanced>`:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F + \mathrm{reg}\cdot\Omega(\gamma) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-stabilized-unbalanced>`


    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,) or array-like (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If many, compute all the OT distances :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    tau : float
        thershold for max value in u or v for log scaling
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        - gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_distance : (n_hists,) array-like
            the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
            log dictionary returned only if `log` is `True`
    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, 1., 1.)
    array([[0.51122823, 0.18807035],
           [0.18807035, 0.51122823]])


    .. _references-sinkhorn-stabilized-unbalanced:
    References
    ----------
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    a, b, M = list_to_array(a, b, M)
    nx = get_torch_backend(M, a, b)
    device = a.device


    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M).to(device) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M).to(device) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, n_hists), type_as=M).to(device) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M).to(device) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M).to(device) / dim_a
        v = nx.ones(dim_b, type_as=M).to(device) / dim_b

    # print(reg)
    K = nx.exp(-M / reg)

    fi = reg_m / (reg_m + reg)

    cpt = 0
    err = 1.
    alpha = nx.zeros(dim_a, type_as=M)
    beta = nx.zeros(dim_b, type_as=M)
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        f_alpha = nx.exp(- alpha / (reg + reg_m))
        f_beta = nx.exp(- beta / (reg + reg_m))

        if n_hists:
            f_alpha = f_alpha[:, None]
            f_beta = f_beta[:, None]
        u = ((a / (Kv + 1e-16)) ** fi) * f_alpha
        Ktu = nx.dot(K.T, u)
        v = ((b / (Ktu + 1e-16)) ** fi) * f_beta
        absorbing = False
        if nx.any(u > tau) or nx.any(v > tau):
            absorbing = True
            if n_hists:
                alpha = alpha + reg * nx.log(nx.max(u, 1))
                beta = beta + reg * nx.log(nx.max(v, 1))
            else:
                alpha = alpha + reg * nx.log(nx.max(u))
                beta = beta + reg * nx.log(nx.max(v))
            K = nx.exp((alpha[:, None] + beta[None, :] - M) / reg)
            v = nx.ones(v.shape, type_as=v)
        Kv = nx.dot(K, v)

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            u = uprev
            v = vprev
            break
        if (cpt % 10 == 0 and not absorbing) or cpt == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.max(nx.abs(u - uprev)) / max(
                nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
            )
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1

    if err > stopThr:
        warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." +
                      "Try a larger entropy `reg` or a lower mass `reg_m`." +
                      "Or a larger absorption threshold `tau`.")
    if n_hists:
        logu = alpha[:, None] / reg + nx.log(u)
        logv = beta[:, None] / reg + nx.log(v)
    else:
        logu = alpha / reg + nx.log(u)
        logv = beta / reg + nx.log(v)
    if log:
        log['logu'] = logu
        log['logv'] = logv
    if n_hists:  # return only loss
        res = nx.logsumexp(
            nx.log(M + 1e-100)[:, :, None]
            + logu[:, None, :]
            + logv[None, :, :]
            - M[:, :, None] / reg,
            axis=(0, 1)
        )
        res = nx.exp(res)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        ot_matrix = nx.exp(logu[:, None] + logv[None, :] - M / reg)
        if log:
            return ot_matrix, log
        else:
            return ot_matrix

def PUOT2(a, b, M, reg, reg_m, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False, mass=None):
    '''
    torch version edited by QWC 

    method : https://pythonot.github.io/gen_modules/ot.unbalanced.html
    implementation version with numpy : https://pythonot.github.io/_modules/ot/unbalanced.html#sinkhorn_knopp_unbalanced
    '''

    M, a, b = list_to_array(M, a, b)
    # a = a.double()
    # b = b.double()
    # M = M.double()
    device = a.device

    nx = get_torch_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M).to(device) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M).to(device) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, 1), type_as=M).to(device) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M).to(device) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M).to(device) / dim_a
        v = nx.ones(dim_b, type_as=M).to(device) / dim_b


    if mass is None:
        #m = np.min((np.sum(a), np.sum(b))) * 1.0
        mass = torch.min(a.sum(), b.sum()) * 1.0

    if mass < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if mass > torch.min(a.sum(), b.sum()):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    K = nx.exp(M / (-reg))
    K = K * (mass / K.sum())

    fi = reg_m / (reg_m + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fi
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fi

        scale = mass/ (u[:, None] * K * v[None, :]).sum()
        scale = scale**0.5
        u *= scale 
        v *= scale 

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.
        )
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]


def PUOT3(a, b, M, reg, m=None, numItermax=100,
                                 stopThr=1e-100, verbose=False, log=False, reg_m=0.5, OPT=False):
    
    a = a.double()
    b = b.double()
    M = M.double()
    device = a.device
    # m = m.to(device)

    dim_a, dim_b = M.shape
    #dx = np.ones(dim_a, dtype=np.float64)
    dx = torch.ones(dim_a, dtype=a.dtype).to(device)
    #dy = np.ones(dim_b, dtype=np.float64)
    dy = torch.ones(dim_b, dtype=b.dtype).to(device)

    if len(a) == 0:
        #a = np.ones(dim_a, dtype=np.float64) / dim_a
        a = torch.ones(dim_a, dtype=a.dtype) / dim_a

    if len(b) == 0:
        #b = np.ones(dim_b, dtype=np.float64) / dim_b
        b = torch.ones(dim_b, dtype=b.dtype) / dim_b

    if m is None:
        #m = np.min((np.sum(a), np.sum(b))) * 1.0
        m = torch.min(a.sum(), b.sum()) * 1.0

    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if m > torch.min(a.sum(), b.sum()):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    log_e = {'err': []}

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    #K = np.empty(M.shape, dtype=M.dtype)
    #K = torch.empty(M.shape, dtype=M.dtype).to(device)
    #np.divide(M, -reg, out=K)
    K = torch.div(M, -reg).to(device)
    #np.exp(K, out=K)
    K = torch.exp(K)
    K = K.to(device)
    #np.multiply(K, m / np.sum(K), out=K)
    K = K * (m / K.sum())

    err, cpt = 1, 0

    fi = reg_m / (reg_m + reg)
    if OPT:
        fi=1.0


    while (cpt < numItermax):  # err > stopThr and

        K1 = torch.matmul(torch.diag(torch.min((a / torch.sum(K, dim=1))**fi, dx)), K)

        #K2 = np.dot(K1, np.diag(np.minimum(b / np.sum(K1, axis=0), dy)))
        K2 = torch.matmul(K1, torch.diag(torch.min((b / torch.sum(K1, dim=0))**fi, dy)))

        K = K2 * (m / torch.sum(K2))

        cpt = cpt + 1
        #print('K:{}'.format(K))
        if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break
 
    return K
