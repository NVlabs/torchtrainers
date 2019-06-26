#
# Copyright (c) 2013-2019 Thomas Breuel. All rights reserved.
# This file is part of torchtrainers (see unknown).
# See the LICENSE file for licensing terms (BSD-style).
#
"""Training-related part of the Keras engine.
"""

import os
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim
import time
import re
import copy

def get_info():
    import platform
    import getpass
    node = str(platform.node())
    user = getpass.getuser()
    now = time.ctime()
    return "{} {} {}".format(now, node, user)

def loader_test(source, nbatches=10, skip=10):
    for i, sample in enumerate(source):
        if i >= skip-1: break

    start = time.time()
    count = 0
    for i, sample in enumerate(source):
        xs = sample[0]
        count += len(xs)
        if i >= nbatches-1: break
    finish = time.time()

    delta = finish-start
    print("{:.2f} samples/s {:.2f} batches/s".format(count/delta, nbatches/delta))
    for index, a in enumerate(sample):
        if isinstance(a, torch.Tensor):
            print(index, ":", "Tensor", a.shape, a.device, a.dtype, a.min().item(), a.max().item())
        elif isinstance(a, np.ndarray):
            print(index, ":", "ndarray", a.shape, a.dtype, np.amin(a), np.amax(a))
        else:
            print(index, ":", type(a))


def astime(s):
    s = int(s+0.999)
    seconds = s%60
    s = s//60
    minutes = s%60
    s = s//60
    hours = s%24
    days = s//24
    result = ""
    if days>0: result = "{:d}d".format(days)
    result += "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return result

def within_jupyter():
    try:
        cfg = get_ipython().config
        if cfg["IPKernelApp"]["parent_appname"] is not None:
            return True
        else:
            return False
    except NameError:
        return False

def jupyter_plot(axis, ys, xs=None, sigma=0, xscale=None, yscale=None):
    #display.clear_output(wait=True)
    axis.cla()
    if xscale is not None:
        axis.set_xscale(xscale)
    if yscale is not None:
        axis.set_yscale(yscale)
    from scipy.ndimage import filters
    if sigma>0:
        ys = filters.gaussian_filter(np.array(ys, "f"), sigma, mode="nearest")
    if xs is not None:
        axis.plot(xs, ys)
    else:
        axis.plot(ys)
    #print(np.amin(ys), np.amax(ys))

class Progress(object):
    def __init__(self, chunk=1000):
        self.chunk = chunk
        self.losses = []
        self.avg_losses = []

    def add(self, count, loss):
        self.losses.append((count, loss))
        if self.losses[-1][0]-self.losses[0][0] > self.chunk:
            avg_loss = np.mean([x[1] for x in self.losses])
            self.avg_losses.append((self.losses[-1][0], avg_loss))
            self.losses = []

    def value(self, index=1):
        if len(self.avg_losses) > 0:
            return self.avg_losses[-1][index]
        elif len(self.losses) > 0:
            return self.losses[-1][index]
        else:
            return -1

    def plot(self, axis=None, **kw):
        axis = axis or plt.gca()
        if len(self.avg_losses)==0: return
        xs = [p[0] for p in self.avg_losses]
        ys = [p[1] for p in self.avg_losses]
        axis.plot(xs, ys, **kw)

def getvalue(x):
    if hasattr(x, "item"):
        return x.item()
    else:
        return float(x)

class MovingAverage0(object):
    def __init__(self, initial=1.0, range=5000):
        self.range = range
        self.maverage = initial
        self.count = 0
        self.total = 0
    def add(self, x, weight=1):
        self.total += x
        self.count += 1
        self.maverage = self.maverage * float(weight) / self.range + \
                     x * float(self.range - weight) / self.range
        return self.maverage
    def recent(self):
        return self.maverage
    def value(self):
        return float(self.total) / self.count

class MovingAverage(object):
    def __init__(self, range=50):
        self.range = range
        self.values = []
    def add(self, x, weight=1):
        self.values.append(x)
    def recent(self):
        if self.values==[]: return -1
        return np.mean(self.values[-self.range:])
    def value(self):
        if self.values==[]: return -1
        return np.mean(self.values)
    def __len__(self):
        return len(self.values)

class Misclassification(object):
    def __init__(self):
        self.counts = 0
        self.errs = 0
        self.moving = MovingAverage()
    def name(self):
        return "err"
    def add(self, pred, target):
        assert not torch.is_floating_point(target), target.dtype
        assert target.ndimension() == 1
        if target.ndimension() == pred.ndimension() - 1:
            _, pred = pred.max(-1)
        assert target.ndimension() == pred.ndimension(), (target.size(), pred.size())
        assert target.size() == pred.size(), (target.size(), pred.size())
        counts = len(target)
        errs = (target != pred).float().sum()
        self.counts += counts
        self.errs += errs
        self.moving.add(float(errs)/counts, counts)
    def value(self):
        return float(self.errs) / self.counts
    def recent(self):
        return self.moving.recent()


def apply1(f, x):
    if f is None:
        return x
    else:
        return f(x)

def jupyter_after_batch(self):
    #fig = plt.gcf()
    plt.close("all")

    fig = plt.figure(figsize=(12, 6))
    fig.clf()

    fig.add_subplot(1, 2, 1)
    fig.add_subplot(1, 2, 2)

    axis = fig.get_axes()[0]
    self.losses.plot(axis)
    axis = fig.get_axes()[1]
    self.errors.plot(axis)

    display.clear_output(wait=True)
    display.display(axis.figure)

def compare(x, y):
    if x<y: return -1
    if x>y: return 1
    return 0

def getmeta(meta, key, mode="training"):
    if key in meta:
        return meta[key]
    if key=="loss":
        return meta.get(mode+"_loss")
    errs = meta.get(mode+"_err")
    for k, v in errs:
        if k==key:
            return v
    return None

def extract_log(metalist, key="loss", mode="training"):
    xs, ys = [], []
    for meta in metalist:
        xs.append(getmeta(meta, "ntrain"))
        ys.append(getmeta(meta, key=key, mode=mode))
    return xs, ys

def compare_metas(meta1, meta2):
    if meta1 is None:
        return 1
    if meta2 is None:
        return -1
    if "testing_err" in meta1:
        return compare(meta1["testing_err"][0][1],
                       meta2["testing_err"][0][1])
    elif "testing_loss" in meta1:
        return compare(meta1["testing_loss"],
                       meta2["testing_loss"])
    else:
        return compare(meta1["training_loss"],
                       meta2["training_loss"])

class Trainer(object):
    """The `Model` class adds training & evaluation routines to a `Network`.
    """

    def __init__(self,
                 model,
                 criterion=None,
                 optimizer=None,
                 optimizer_factory=torch.optim.SGD,
                 preprocess=None,
                 encode_input=None,
                 encode_target=None,
                 decode_pred=None,
                 target_tensors=None,
                 stop_if=None,
                 device=None,
                 metrics=[],
                 verbose=True,
                 #loss_weights=None,
                 #sample_weight_mode=None,
                 #weighted_metrics=None,
                 ):

        self.device = device

        self.set_model(model)
        self.criterion = criterion
        self.criterion.to(device)
        self.optimizer_factory = optimizer_factory
        if optimizer is not None:
            self.set_optimizer(optimizer)
        self.verbose = verbose

        self.preprocess = preprocess
        self.encode_target = encode_target
        self.decode_pred = decode_pred
        self.encode_input = decode_pred

        self.metrics = metrics
        self.reset_metrics()

        self.count = 0
        self.after_batch = self.default_after_batch
        self.after_epoch = self.default_after_epoch
        self.report_every = 1
        self.record_last = True
        self.last_lr = None
        self.progress_prefix = "training"
        self.logs = []

    def set_model(self, model):
        self.model = model.to(self.device)
        if not hasattr(self.model, "META"):
            self.model.META = dict(ntrain=0)
        self.optimizer = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.zero_grad()

    def set_lr(self, lr, **kw):
        if lr != self.last_lr:
            self.optimizer = self.optimizer_factory(self.model.parameters(), lr, **kw)
            sopt = re.sub(r"\s+", " ", str(self.optimizer))
            if self.verbose:
                print("setting optimizer:", sopt)
            self.optimizer.zero_grad()
            self.last_lr = lr

    def reset_metrics(self, current_size=-1, current_count=0):
        self.current_size = current_size
        self.current_count = current_count
        self.losses = MovingAverage()
        self.mobjects = [m() for m in self.metrics]
        self.current_start = time.time()

    def report_recent_short(self):
        if self.current_size == 0:
            return "..."
        now = time.time()
        delta = now - self.current_start
        total_time = float(self.current_size) / self.current_count * delta
        remaining_time =  total_time - delta
        progress = self.progress_prefix
        progress += " {:3.0f}%".format(
            100.0 * float(self.current_count) / self.current_size)
        if self.current_count>= self.current_size:
            progress += " /{} ".format(
                astime(total_time))
        else:
            progress += "  {} ".format(
                astime(remaining_time))
        progress += " loss {:6.4f}".format(
            self.losses.recent())
        for m in self.mobjects[:1]:
            progress+= " {} {:6.4f}".format(m.name()[:3], m.recent())
        return progress

    def report_recent(self):
        if self.current_size == 0:
            return "..."
        now = time.time()
        delta = now - self.current_start
        total_time = float(self.current_size) / self.current_count * delta
        remaining_time =  total_time - delta
        progress = self.progress_prefix
        progress += "{:8d} / {:8d}".format(self.current_count, self.current_size)
        progress += " time {} / {}".format(astime(remaining_time), astime(total_time))
        progress += " {:3.0f}%".format(
            100.0 * float(self.current_count) / self.current_size)
        progress += " loss {:9.5f} [{:6d}]".format(self.losses.recent(), len(self.losses))
        for m in self.mobjects:
            progress+= " {} {:9.5f}".format(m.name()[:3], m.recent())
        return progress

    def report_metrics(self):
        report = "loss {:6.4f}".format(self.losses.value())
        for m in self.mobjects:
            report += " {} {:6.4f}".format(m.name()[:3], m.value())
        return report

    def default_after_batch(self):
        if self.current_count == 0: return
        progress = self.report_recent()
        print(progress + "\r", end="", flush=True)

    def default_after_epoch(self):
        print()


    def train_on_batch(self, x, target, sample_weight=None, class_weight=None):
        assert sample_weight is None
        assert class_weight is None
        self.model.train()
        x, target = apply1(self.preprocess, (x, target))
        if x is None: return
        batch_size = len(x)
        assert batch_size > 0
        assert len(x) == len(target)
        x = apply1(self.encode_input, x)
        target = apply1(self.encode_target, target)
        pred = self.model(x.to(self.device))
        assert len(pred) == batch_size
        target = target.to(self.device)
        loss = self.criterion(pred, target)
        if self.record_last:
            self.last_x = x.to("cpu")
            self.last_y = target.to("cpu")
            self.last_pred = pred.to("cpu")
            self.last_loss = getvalue(loss)
        self.losses.add(getvalue(loss), batch_size)
        for m in self.mobjects:
            m.add(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += len(x)
        self.model.META["ntrain"] += len(x)

    def predict_on_batch(self, x):
        self.model.eval()
        pred = self.model(x.to(device))
        if self.decode_pred:
            pred = self.decode_pred(pred)
        return pred

    def test_on_batch(self, x, target, sample_weight=None):
        assert sample_weight is None
        x, target = apply1(self.preprocess, (x, target))
        batch_size = len(x)
        x = apply1(self.encode_input, x)
        target = apply1(self.encode_target, target)
        target = target.to(self.device)
        self.model.eval()
        pred = self.model(x.to(self.device))
        loss = self.criterion(pred, target)
        self.losses.add(getvalue(loss), batch_size)
        for m in self.mobjects:
            m.add(pred, target)

    def fit_for(self,
                generator,
                samples_per_epoch,
                epochs=1,
                verbose=1,
                callbacks=None,
                class_weight=None,
                stop_if=None):
        assert callbacks is None
        assert class_weight is None
        self.model.eval()
        self.reset_metrics(samples_per_epoch * epochs)
        batch_count = 0
        try:
            while True:
                try:
                    for (x, target) in generator:
                        self.train_on_batch(x, target)
                        self.current_count += len(x)
                        if self.after_batch is not None and batch_count%self.report_every==0:
                            self.after_batch()
                        if stop_if is not None:
                            if stop_if(self):
                                return
                        if self.current_count >= self.current_size:
                            return
                        batch_count += 1
                finally:
                    if self.after_epoch is not None:
                        self.after_epoch()
        finally:
            self.model.META["training_info"] = get_info()
            self.model.META["training_loss"] = self.losses.value()
            self.model.META["training_err"] = self.get_metrics()
            self.logs.append(copy.copy(self.model.META))

    def test_for(self,
                     generator,
                     epoch_size=None,
                     callbacks=None,
                     class_weight=None,
                     verbose=0):
        assert callbacks is None
        assert class_weight is None
        self.model.eval()
        self.reset_metrics()
        count = 0
        try:
            for (x, target) in generator:
                self.test_on_batch(x, target)
                count += len(target)
                if epoch_size is not None and count >= epoch_size:
                    break
        finally:
            if len(self.logs) > 0 and self.logs[-1]["ntrain"] == self.model.META["ntrain"]:
                del self.logs[-1]
            self.model.META["testing_info"] = get_info()
            self.model.META["testing_loss"] = self.losses.value()
            self.model.META["testing_err"] = self.get_metrics()
            self.logs.append(copy.copy(self.model.META))
        return self.report_metrics()

    def get_metrics(self):
        return [(m.name(), m.value()) for m in self.mobjects]

    def metric_names(self):
        return [m.name() for m in self.mobjects]

    def get_metric(self, name):
        return next(m for m in self.mobjects if m.name()==name)

def find_save(savename, epochs):
    last_name = (-1, None)
    for i in range(epochs):
        name = savename.format(epoch=i)
        if os.path.exists(name):
            last_name = (i, name)
    return last_name

def find_lr(rates, n):
    assert rates[0][0] == 0
    for (i, r) in rates[::-1]:
        if n>=i: return r

def save_cpu(fname, model, device=None, LOGS=None):
    model = model.cpu()
    if LOGS is not None:
        model.LOGS = LOGS
    torch.save(model, fname)
    model.to(device)

def training(trainer,
             training,
             training_size,
             testing,
             testing_size,
             epochs=100,
             save_every=None,
             save_best=None,
             savename="model-{epoch:04d}.pyd",
             learning_rates=[(0, 0.9e-3)],
             oneline=False,
             restart=True,
             verbose=False):
    epoch = -1
    last_eval = "(none)"
    last_len = [0]
    def dline(progress):
        #progress = "{"+progress+"}"
        delta = max(3, last_len[0] - len(progress))
        print(progress + " "*delta + "\r", end="", flush=True)
        last_len[0] = len(progress)
    def after_batch():
        progress = "{:4d} test {} ::: train {}    ".format(
            epoch,
            last_eval,
            trainer.report_recent_short())
        dline(progress)
    def after_epoch():
        if not oneline:
            print()
    trainer.after_batch = after_batch
    trainer.after_epoch = after_epoch
    trainer.progress_prefix = ""
    trainer.verbose = not oneline
    start_epoch, name = find_save(savename, epochs)
    if restart and start_epoch >= 0:
        print("\nloading", name)
        model = torch.load(name)
        trainer.set_model(model)
    else:
        start_epoch = 0
    best_meta = None
    for epoch in range(start_epoch, epochs):
        ntrain = trainer.model.META["ntrain"]
        if callable(learning_rates):
            lr = learning_rates(ntrain)
        else:
            lr = find_lr(learning_rates, ntrain)
        trainer.set_lr(lr)
        trainer.fit_for(training, training_size)
        last_eval = trainer.test_for(testing,testing_size)
        if save_every is not None and (epoch+1) % save_every == 0:
            fname = savename.format(epoch=epoch)
            assert not os.path.exists(fname), fname
            last_eval += " [saved]"
            save_cpu(fname, trainer.model, device=trainer.device, LOGS=trainer.logs)
        if compare_metas(best_meta, trainer.model.META) > 0:
            best_meta = copy.copy(trainer.model.META)
            if save_best is not None:
                save_cpu(save_best, trainer.model, device=trainer.device, LOGS=trainer.logs)
                if "saved" not in last_eval: last_eval += " [saved]"
