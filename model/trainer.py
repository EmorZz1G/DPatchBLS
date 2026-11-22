from model.patch_model.patch_bls_single import TwoPatchBLS_Single
from utils import TimeRecorder


def train(model: TwoPatchBLS_Single, train_data, device, verbose=False):
    model.train()
    train_data = train_data.to(device)
    k_res, b_res = model(train_data, verbose)
    k_res = k_res.cpu().detach().numpy()
    b_res = b_res.cpu().detach().numpy()
    return k_res, b_res


def valid(model: TwoPatchBLS_Single, valid_data, device, verbose=False):
    model.eval()
    valid_data = valid_data.to(device)
    k_res, b_res = model(valid_data, verbose)
    k_res = k_res.cpu().detach().numpy()
    b_res = b_res.cpu().detach().numpy()
    return k_res, b_res


def test(model: TwoPatchBLS_Single, test_data, device, verbose=False):
    tm = TimeRecorder(device)
    tm.start('Test')
    model.eval()
    test_data = test_data.to(device)
    kb, b, pred_score = model.decision_function(test_data, verbose)
    kb = kb.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    pred_score = pred_score.cpu().detach().numpy()
    t_te = tm.stop()
    return kb, b, pred_score, t_te


def trainer(model: TwoPatchBLS_Single, train_x, valid_x, test_x, device, verbose=False):
    tm = TimeRecorder(device)
    tm.start('Train')
    tr_k, tr_b = train(model, train_x, device, verbose)
    va_k, va_b = valid(model, valid_x, device, verbose)
    t_tr = tm.stop()
    kb, b, pred_score, t_te = test(model, test_x, device, verbose)
    return kb, b, pred_score, t_tr, t_te, tr_k, tr_b, va_k, va_b
