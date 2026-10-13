"""Pilot: 'existence vs discovery'. For real data, compare
(a) the l_inf / l_2 flip radius of an ordinarily trained network, with
(b) the certified radius of a plain 1-nearest-neighbour classifier on the same training set:
    1-NN keeps its answer under any change smaller than (d_other - d_same)/2, where d_same/d_other are the
    distances to the nearest training point of the same / another class.
If (b) >> (a), a more robust classifier is 'sitting in the data' and the question becomes why training does not find it."""
import pickle, sys
import numpy as np, torch
sys.path.insert(0, "/home/akayyam/adv_reason")
import root_cause_realdata as rd
import root_cause_cifar as rc

DEV = rd.DEV
N_TRAIN, N_TEST = 10000, 500


def nn_stats(xtr, ytr, xte, yte, p):
    """per test point: nearest same-class and other-class training distance, 1-NN prediction."""
    out_same, out_other, pred = [], [], []
    for i in range(0, len(xte), 50):
        d = torch.cdist(xte[i:i + 50], xtr, p=p)                       # (b, n)
        same = (ytr[None, :] == yte[i:i + 50, None])
        ds = torch.where(same, d, torch.full_like(d, float("inf"))).min(1).values
        do = torch.where(~same, d, torch.full_like(d, float("inf"))).min(1).values
        out_same.append(ds); out_other.append(do)
        pred.append(ytr[d.argmin(1)])
    return torch.cat(out_same).cpu().numpy(), torch.cat(out_other).cpu().numpy(), torch.cat(pred)


def run(name, xtr, ytr, xte, yte, model, hi_inf, hi_2):
    xtr, ytr, xte, yte = xtr.to(DEV), ytr.to(DEV), xte.to(DEV), yte.to(DEV)
    res = {}
    for norm, p, hi in [("inf", float("inf"), hi_inf), ("2", 2.0, hi_2)]:
        ds, do, pred = nn_stats(xtr, ytr, xte, yte, p)
        cert = np.maximum((do - ds) / 2, 0)
        ok = (pred == yte).cpu().numpy()
        fr = rd.min_eps(model, xte, yte, norm, hi=hi)          # net flip radius over correctly classified points
        res[norm] = dict(nn_acc=float(ok.mean()), cert_median=float(np.median(cert[ok])),
                         other_median=float(np.median(do)), same_median=float(np.median(ds)),
                         net_median=float(np.median(fr)), net_acc=float((model(xte).argmax(1) == yte).float().mean()))
        print(name, norm, res[norm], flush=True)
    return res


if __name__ == "__main__":
    out = {}
    # ---- MNIST
    tr, te = rd.load()
    torch.manual_seed(0)
    idx = torch.randperm(len(tr[0]))[:N_TRAIN]
    xtr, ytr = tr[0][idx], tr[1][idx]
    xte, yte = te[0][:N_TEST], te[1][:N_TEST]
    m = rd.train(xtr.to(DEV), ytr.to(DEV), 10)
    out["mnist"] = run("mnist", xtr, ytr, xte, yte, m, 1.0, 20.0)
    # ---- CIFAR-10
    ctr, cte = rc.load()
    idx = torch.randperm(len(ctr[0]))[:N_TRAIN]
    xtr, ytr = ctr[0][idx], ctr[1][idx]
    xte, yte = cte[0][:N_TEST], cte[1][:N_TEST]
    model = torch.nn.Sequential(rc.Net()).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    a, y = xtr.to(DEV), ytr.to(DEV)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=15 * ((len(a) + 127) // 128))
    model.train()
    for _ in range(15):
        pm = torch.randperm(len(a), device=DEV)
        for k in range(0, len(a), 128):
            b = pm[k:k + 128]; opt.zero_grad()
            torch.nn.functional.cross_entropy(model(a[b]), y[b]).backward(); opt.step(); sch.step()
    model.eval()
    out["cifar10"] = run("cifar10", xtr, ytr, xte, yte, model, 0.5, 10.0)
    pickle.dump(out, open("study/results/pilot_gap.pkl", "wb"))
