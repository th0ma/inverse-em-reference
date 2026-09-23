import itertools
import math
import torch

from inverse_em.localization import CanonicalTargets, ScientificPrediction, diagnostic_assignment


def pred(rho,phi):
    rho=torch.tensor(rho,dtype=torch.float64);phi=torch.tensor(phi,dtype=torch.float64)
    return ScientificPrediction(rho,torch.cos(phi),torch.sin(phi),phi)


def target(rho,phi):return CanonicalTargets(torch.tensor(rho,dtype=torch.float64),torch.tensor(phi,dtype=torch.float64))


def test_s2_identity_then_swap_and_assignment_change():
    p=pred([[.8,.2]],[[math.pi,0.]])
    result=diagnostic_assignment(p,target([[.2,.8]],[[0.,math.pi]]))
    assert result.permutations==((0,1),(1,0)) and result.chosen_indices.tolist()==[1]
    assert result.assignment_changed.tolist()==[True] and result.candidate_costs[0,1]<result.candidate_costs[0,0]


def test_s3_lexicographic_candidate_order():
    order=(2,0,1);rho=[[.3,.5,.7]];phi=[[0.,1.,2.]]
    p=pred([[rho[0][i] for i in order]],[[phi[0][i] for i in order]])
    result=diagnostic_assignment(p,target(rho,phi))
    assert result.permutations==tuple(itertools.permutations(range(3)))
    assert result.permutations[int(result.chosen_indices[0])]==order


def test_exact_tie_retains_first_enumerated_permutation():
    p=pred([[.4,.4]],[[0.,0.]])
    result=diagnostic_assignment(p,target([[.4,.4]],[[0.,0.]]))
    assert torch.equal(result.candidate_costs[:,0],result.candidate_costs[:,1])
    assert result.chosen_indices.tolist()==[0] and result.assignment_changed.tolist()==[False]
