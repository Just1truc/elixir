import torch

R_left  = torch.tensor([[0., -1.],[1., 0.]])
R_right = torch.tensor([[0.,  1.],[-1., 0.]])

def next_action_90(a, g, h, eps=1e-9):

    a = a.float(); g = g.float(); h = h.float()
    d = g - a
    if torch.allclose(d, torch.zeros_like(d)):
        return 'forward'

    curr_d2 = (d*d).sum()
    new_d2  = ((g - (a + h))**2).sum()
    if new_d2 + eps < curr_d2:
        return 'forward'

    t = d / (torch.norm(d) + eps)
    hL = (R_left  @ h)
    hR = (R_right @ h)

    dotL = (hL * t).sum()
    dotR = (hR * t).sum()
    if dotL >= dotR:
        return 'left'
    else:
        return 'right'

agent_pos = torch.tensor([5, 5])
heading_pos = torch.tensor([0, 1])
target_pos = torch.tensor([4, 6])

print(next_action_90(agent_pos, target_pos, heading_pos))