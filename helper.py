def estimate_grads(cs_loader, model, criterion):
    model.train()
    all_grads = []
    all_targets = []
    all_preds = []
    #cs_loader = DataLoader(dataset,batch_size = 128, shuffle = False)
    for i, batch in enumerate(cs_loader):
        input, target, index = batch
        input = input.to('cuda:0')
        all_targets.append(target)
        target = target.to('cuda:0')
        
        output, feat = model(input)
        _, pred = torch.max(output, 1)
        loss = criterion(output, target).mean()
        est_grad = grad(loss, feat)
        all_grads.append(est_grad[0].detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())
        
    all_grads = np.vstack(all_grads)
    #all_grads = torch.vstack(all_grads))
    all_targets = np.hstack(all_targets)
    all_preds = np.hstack(all_preds)
    return all_grads, all_targets

class FacilityLocation:
    def __init__(self, V, D=None, fnpy=None):
        if D is not None:
          self.D = D
        else:
          self.D = np.load(fnpy)

        self.D *= -1
        self.D -= self.D.min()
        self.V = V
        self.curVal = 0
        self.gains = []
        self.curr_max = np.zeros_like(self.D[0])

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            new_dists = np.stack([self.curr_max, self.D[ndx]], axis=0)
            return new_dists.max(axis=0).sum()
        else:
            return self.D[sset + [ndx]].sum()

    def add(self, sset, ndx, delta):
        self.curVal += delta
        self.gains += delta,
        self.curr_max = np.stack([self.curr_max, self.D[ndx]], axis=0).max(axis=0)
        return self.curVal

        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curVal = self.D[:, sset + [ndx]].max(axis=1).sum()
        else:
            self.curVal = self.D[:, sset + [ndx]].sum()
        self.gains.extend([self.curVal - cur_old])
        return self.curVal

def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)

    cnt = 0
    for index in V:
      _heappush_max(order, (F.inc(sset, index), index))
      cnt += 1

    n_iter = 0
    while order and len(sset) < B:
        n_iter += 1
        if F.curVal == len(F.D):
          # all points covered
          break

        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv > 0: 
            if not order:
                curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    return sset, vals