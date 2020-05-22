from dreamcoder.program import *


class ArcState:

    def __init__(self, color=0, draw_type='edit', history=None):
        self.history = history
        self.color = color
        self.draw_type = 'edit'

    def __str__(self): return f"ArcS(c={self.color},d={self.draw_type})"

    def __repr__(self): return str(self)

    def color_pixel(self, x, y):
        if self.history is None:
            return self
        else:
            return ArcState(color=self.color, draw_type=self.draw_type, history =
                            self.history + [(color, x, y)])

    def set_color(self, color):
        return ArcState(color=color, draw_type=self.draw_type, history =
                self.history if self.history is None else self.history + [self])

    def set_draw_type(self, draw_type):
        return ArcState(color=self.color, draw_type=draw_type, history =
                self.history if self.history is None else self.history + [self])
        

def _empty_grid(): return ArcState()

def _set_color(color):
    return lambda k: lambda s: k(s.set_color(color))

def _set_draw_type(draw_type):
    return lambda k: lambda s: k(s.set_draw_type(draw_type))

def _simpleLoop(n):
    def f(start, body, k):
        if start >= n: return k
        return body(start)(f(start + 1, body, k))
    return lambda b: lambda k: f(0,b,k)

def _embed(body):
    def f(k):
        def g(state):
            bodyState, bodyActions = body(_empty_grid)(hand)
            # Record history if we are doing that
            if state.history is not None:
                state = ArcState(color=state.color,
                                  draw_type =state.draw_type,
                                  history=bodyState.history)
            state, laterActions = k(state)
            return state, bodyActions + laterActions
        return g
    return f

# primitive functions can only have one argument; I think this is a requirement
# of lamdba-calculus style funcitons. see towerPrimitives.py, listPrimitives.py
# def _fill_pixel(x):
    # return lambda y: lambda k: lambda s: k(s.fill_pixel(x, y))


# for coloring. this is an "action" so it's special?
def _fill_pixel(x):
    def f(state, x, y, k):
        thisAction = [state.color]
        state = state.fill_pixel(x, y)
        state, rest = k(state)
        return hand, thisAction + rest
    return lambda y: lambda k: f(state, x, y, k)

tgrid = baseType("grid")

primitives = [
        Primitive("set_color", arrow(tint, tint, tint, tgrid, tgrid)), _draw),
        Primitive("empty", tgrid, _empty_grid)
] + [Primitive(str(j), tint, j) for j in range(9)]


common_primitives = [
    Primitive("tower_loopM", arrow(tint, arrow(tint, ttower, ttower), ttower, ttower), _simpleLoop),
    Primitive("tower_embed", arrow(arrow(ttower,ttower), ttower, ttower), _embed),
] + [Primitive(name, arrow(ttower,ttower), TowerContinuation(0, w, h))
     for name, (w, h) in blocks.items()] + \
         [Primitive(str(j), tint, j) for j in range(1,9) ]
primitives = common_primitives + [
    Primitive("left", arrow(tint, ttower, ttower), _left),
    Primitive("right", arrow(tint, ttower, ttower), _right)
    ]

new_primitives = common_primitives + [
    Primitive("moveHand", arrow(tint, ttower, ttower), _moveHand),
    Primitive("reverseHand", arrow(ttower, ttower), _reverseHand)
    ]

def executeTower(p, timeout=None):
    try:
        return runWithTimeout(lambda : p.evaluate([])(_empty_tower)(TowerState())[1],
                              timeout=timeout)
    except RunWithTimeout: return None
    except: return None

def animateTower(exportPrefix, p):
    print(exportPrefix, p)
    from dreamcoder.domains.tower.tower_common import renderPlan
    state,actions = p.evaluate([])(_empty_tower)(TowerState(history=[]))
    print(actions)
    trajectory = state.history + [state]
    print(trajectory)
    print()

    assert tuple(z for z in trajectory if not isinstance(z, TowerState) ) == tuple(actions)        

    def hd(n):
        h = 0
        for state in trajectory[:n]:
            if isinstance(state, TowerState):
                h = state.hand
        return h
    animation = [renderPlan([b for b in trajectory[:n] if not isinstance(b, TowerState)],
                            pretty=True, Lego=True,
                            drawHand=hd(n),
                            masterPlan=actions,
                            randomSeed=hash(exportPrefix))
                 for n in range(0,len(trajectory) + 1)]
    import scipy.misc
    import random
    r = random.random()
    paths = []
    for n in range(len(animation)):
        paths.append(f"{exportPrefix}_{n}.png")
        scipy.misc.imsave(paths[-1], animation[n])
    os.system(f"convert -delay 10 -loop 0 {' '.join(paths)} {exportPrefix}.gif")
#    os.system(f"rm {' '.join(paths)}")
