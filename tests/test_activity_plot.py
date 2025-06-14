from artibot import globals as G

def test_ring_buffer_roll():
    i0 = G.timeline_index
    for _ in range(G.timeline_depth + 5):
        G.timeline_ind_on[G.timeline_index % G.timeline_depth] = [1,0,1,0,1,0]
        G.timeline_trades[G.timeline_index % G.timeline_depth] = 1
        G.timeline_index += 1
    assert G.timeline_index == i0 + G.timeline_depth + 5
    assert G.timeline_trades[(i0 + 5) % G.timeline_depth] == 1
