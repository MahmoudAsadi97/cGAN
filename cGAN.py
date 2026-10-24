"""cGAN of Li, Dong & Qiu (2025), github.com/malvin0130/cGAN_EB (``cGAN.py``).

Portions adapted from cGAN_EB, Copyright (c) 2024 malvin0130, MIT License
(see ``LICENSE-cGAN_EB``).

``Generator``, ``Discriminator`` and ``cGAN`` are the reference model.  The
only changes to ``cGAN`` are additions that leave the reference behaviour
untouched when left at their defaults:

* ``seed``: seeds noise and shuffling (the reference sets no seed);
* ``shuffle``: reshuffle the minibatches every epoch (reference: file order);
* ``callback`` / ``callback_every``: periodic validation and checkpointing;
* ``start_epoch`` with ``training_state`` / ``load_training_state``: continue an
  interrupted run from its last checkpoint;
* the fake batch is detached in the discriminator step and ``retain_graph``
  is not used; both give the same parameter updates as the reference,
  whose generator gradients from the discriminator step are zeroed before
  the generator step.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):

    def __init__(self, input_size, label_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size+label_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, input_size, label_size, hidden_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size+label_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_size*4, 1),
            nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor, labels: torch.Tensor):
        x = torch.cat([data, labels], dim=1)
        return self.model(x)


class cGAN(object):

    def __init__(self, input_size, label_size, hidden_size, output_size, lr, seed=None, device=None):
        self.input_size = input_size
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # Initialize Generator
        self.generator = Generator(self.input_size, self.label_size, self.hidden_size, self.output_size).to(self.device)
        # Initialize Discriminator
        self.discriminator = Discriminator(self.output_size, self.label_size, self.hidden_size).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2*self.lr)
        self.history = []

    def _step(self, data_temp):
        # Train disc
        self.discriminator.zero_grad(set_to_none=False)

        real_data = data_temp[:, :-1]  # Data + Labels
        classes = data_temp[:, -1]
        onehot_classes = self._eye[classes.long()]  # = F.one_hot, without a host sync

        real_labels = torch.ones(real_data.shape[0], 1, device=self.device)
        fake_labels = torch.zeros(real_data.shape[0], 1, device=self.device)

        noise = torch.randn(real_data.shape[0], self.input_size, device=self.device)
        fake_data = self.generator(noise,
                                   onehot_classes)
        fake_outputs = self.discriminator(fake_data.detach(), onehot_classes)
        real_outputs = self.discriminator(real_data.to(torch.float32), onehot_classes)

        loss_D_real = self.loss_fn(real_outputs, real_labels)
        loss_D_fake = self.loss_fn(fake_outputs, fake_labels)
        loss_D = (loss_D_real + loss_D_fake)/2

        loss_D.backward()
        self.optimizer_D.step()

        # Train generator
        self.generator.zero_grad(set_to_none=False)
        noise = torch.randn(real_data.shape[0], self.input_size, device=self.device)
        fake_data = self.generator(noise,
                                   onehot_classes)
        fake_outputs = self.discriminator(fake_data, onehot_classes)
        loss_G = self.loss_fn(fake_outputs, real_labels)
        loss_G.backward()
        self.optimizer_G.step()

        self._acc.add_(torch.stack([loss_G.detach(), loss_D.detach(), torch.ones((), device=self.device)]))

    def train(self, data: torch.Tensor, epochs, batch_size, shuffle=False, callback=None, callback_every=1,
              cuda_graph=None, start_epoch=0, time_offset=0.0):
        """Reference training loop.  On CUDA the step of a full batch is captured once in a
        CUDA graph and replayed (identical operations, far less launch overhead)."""
        data = data.to(self.device)
        t0 = time.time() - time_offset
        self._eye = torch.eye(self.label_size, device=self.device)
        self._acc = torch.zeros(3, device=self.device)
        use_graph = self.device.type == "cuda" if cuda_graph is None else cuda_graph
        if use_graph:
            for opt in (self.optimizer_D, self.optimizer_G):
                for group in opt.param_groups:
                    group["capturable"] = True
            static = torch.zeros(batch_size, data.shape[1], dtype=data.dtype, device=self.device)
            graph, side, n_eager = None, torch.cuda.Stream(), 0

        for epoch in range(start_epoch, epochs):
            order = torch.randperm(data.shape[0], device=self.device) if shuffle else None
            for i in range(0, data.shape[0], batch_size):
                data_temp = data[order[i:i + batch_size]] if shuffle else data[i:i + batch_size]
                if not use_graph or data_temp.shape[0] != batch_size:
                    self._step(data_temp)
                elif graph is None and n_eager < 3:
                    # Warm-up on a side stream before capture.
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        self._step(data_temp)
                    torch.cuda.current_stream().wait_stream(side)
                    n_eager += 1
                else:
                    if graph is None:
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            self._step(static)
                    static.copy_(data_temp)
                    graph.replay()

            if (epoch + 1) % callback_every == 0 or epoch + 1 == epochs:
                loss_g, loss_d, n = self._acc.tolist()
                self._acc.zero_()
                self.history.append({"epoch": epoch + 1, "loss_G": loss_g / n, "loss_D": loss_d / n,
                                     "seconds": round(time.time() - t0, 1)})
                print("Epoch {} - loss_G: {:.4f}, loss_D: {:.4f} ({:.0f}s)".format(
                    epoch + 1, loss_g / n, loss_d / n, time.time() - t0), flush=True)
                if callback is not None:
                    callback(self, epoch + 1)

    def generate(self, num_samples: int, label_number: int):
        labels = [label_number]*num_samples
        labels_tensor = torch.tensor(labels).long().to(self.device)
        onehot_labels = F.one_hot(labels_tensor, num_classes=self.label_size)
        noise = torch.randn(num_samples, self.input_size).to(self.device)

        with torch.no_grad():
            generated = self.generator(noise, onehot_labels)
            return generated.cpu()

    def sample(self, labels, seed=None):
        """Generate one profile per entry of ``labels`` (any order), in model units."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(0 if seed is None else int(seed))
        labels_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=self.device)
        onehot_labels = F.one_hot(labels_tensor, num_classes=self.label_size)
        noise = torch.randn(len(labels_tensor), self.input_size, device=self.device, generator=gen)
        self.generator.eval()
        with torch.no_grad():
            out = self.generator(noise, onehot_labels).cpu().numpy()
        self.generator.train()
        return out

    def state_dict(self):
        return {"generator": self.generator.state_dict(), "discriminator": self.discriminator.state_dict()}

    def load_state_dict(self, state):
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])

    def training_state(self):
        """Everything needed to continue training: weights, optimiser states and loss history."""
        state = {"model": self.state_dict(), "optimizer_G": self.optimizer_G.state_dict(),
                 "optimizer_D": self.optimizer_D.state_dict(), "history": list(self.history)}
        return state

    def load_training_state(self, state):
        self.load_state_dict(state["model"])
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        self.history = list(state["history"])


# ---------------------------------------------------------------------------
# Improved variant: same conditioning (noise + one-hot label at the
# input of both networks), but wider networks, Wasserstein loss with gradient
# penalty (Gulrajani et al. 2017), shuffled minibatches and an exponential
# moving average (EMA) of the generator weights used for generation.
# ---------------------------------------------------------------------------
def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(a, b), nn.LeakyReLU(0.2, inplace=True)]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class GeneratorPlus(nn.Module):

    def __init__(self, input_size, label_size, hidden_size, output_size, depth=3):
        super(GeneratorPlus, self).__init__()
        self.model = _mlp([input_size + label_size] + [hidden_size] * depth + [output_size])

    def forward(self, noise, labels):
        return self.model(torch.cat([noise, labels], dim=1))


class Critic(nn.Module):
    """Discriminator without sigmoid and dropout (a Wasserstein critic).

    With ``diff`` the hour-to-hour changes x[t+1] - x[t] are appended to the input.
    They are linear in x, so this adds no information or capacity: it
    reparameterises the first layer (W1 x + W2 D x), which changes how the critic
    is initialised and optimised and gives the differences their own weights.
    """

    def __init__(self, input_size, label_size, hidden_size, depth=3, diff=False):
        super(Critic, self).__init__()
        self.diff = diff
        self.model = _mlp([input_size + (input_size - 1 if diff else 0) + label_size] + [hidden_size] * depth + [1])

    def forward(self, data, labels):
        if self.diff:
            data = torch.cat([data, data[:, 1:] - data[:, :-1]], dim=1)
        return self.model(torch.cat([data, labels], dim=1))


class RefCritic(nn.Module):
    """The reference discriminator (64-128-256) without dropout and sigmoid (a Wasserstein critic)."""

    def __init__(self, input_size, label_size, hidden_size):
        super(RefCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size+label_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size*4, 1),
        )

    def forward(self, data, labels):
        return self.model(torch.cat([data, labels], dim=1))


class cWGAN(object):

    def __init__(self, input_size, label_size, hidden_size, output_size, lr, depth=3, n_critic=5,
                 gp_lambda=10.0, betas=(0.5, 0.9), ema=0.999, critic_diff=False, ref_nets=False, seed=None,
                 device=None):
        import copy

        self.input_size = input_size
        self.label_size = label_size
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.ema_decay = ema
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if ref_nets:  # the reference networks (ablation): Generator 64-128-64, critic 64-128-256
            self.generator = Generator(input_size, label_size, hidden_size, output_size).to(self.device)
            self.discriminator = RefCritic(output_size, label_size, hidden_size).to(self.device)
        else:
            self.generator = GeneratorPlus(input_size, label_size, hidden_size, output_size, depth).to(self.device)
            self.discriminator = Critic(output_size, label_size, hidden_size, depth, critic_diff).to(self.device)
        self.generator_ema = copy.deepcopy(self.generator).eval()
        for p in self.generator_ema.parameters():
            p.requires_grad_(False)
        cap = self.device.type == "cuda"
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=betas, capturable=cap)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas, capturable=cap)
        self._eye = torch.eye(label_size, device=self.device)
        self.history = []

    def _gradient_penalty(self, real, fake, labels):
        eps = torch.rand(real.shape[0], 1, device=self.device)
        x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
        (grad,) = torch.autograd.grad(self.discriminator(x_hat, labels).sum(), x_hat, create_graph=True)
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def _step(self, batches):
        """``n_critic`` critic updates (one batch each), then one generator update."""
        for k in range(self.n_critic):
            real = batches[k, :, :-1].to(torch.float32)
            onehot = self._eye[batches[k, :, -1].long()]
            noise = torch.randn(real.shape[0], self.input_size, device=self.device)
            with torch.no_grad():
                fake = self.generator(noise, onehot)
            self.optimizer_D.zero_grad(set_to_none=False)
            loss_D = self.discriminator(fake, onehot).mean() - self.discriminator(real, onehot).mean()
            gp = self._gradient_penalty(real, fake, onehot)
            (loss_D + self.gp_lambda * gp).backward()
            self.optimizer_D.step()
        self.optimizer_G.zero_grad(set_to_none=False)
        noise = torch.randn(real.shape[0], self.input_size, device=self.device)
        loss_G = -self.discriminator(self.generator(noise, onehot), onehot).mean()
        loss_G.backward()
        self.optimizer_G.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(self.generator_ema.parameters()), list(self.generator.parameters()),
                                 1.0 - self.ema_decay)
        self._acc.add_(torch.stack([loss_G.detach(), loss_D.detach(), torch.ones((), device=self.device)]))

    def train(self, data: torch.Tensor, epochs, batch_size, shuffle=True, callback=None, callback_every=1,
              cuda_graph=None, start_epoch=0, time_offset=0.0):
        """One epoch = one pass of the critic over the data (drop_last)."""
        data = data.to(self.device)
        t0 = time.time() - time_offset
        self._acc = torch.zeros(3, device=self.device)
        use_graph = self.device.type == "cuda" if cuda_graph is None else cuda_graph
        group = self.n_critic * batch_size
        static = torch.zeros(self.n_critic, batch_size, data.shape[1], dtype=data.dtype, device=self.device)
        graph, side, n_eager = None, torch.cuda.Stream() if use_graph else None, 0
        for epoch in range(start_epoch, epochs):
            order = torch.randperm(data.shape[0], device=self.device) if shuffle else \
                torch.arange(data.shape[0], device=self.device)
            for i in range(0, data.shape[0] - group + 1, group):
                batches = data[order[i:i + group]].view(self.n_critic, batch_size, -1)
                if not use_graph:
                    self._step(batches)
                elif graph is None and n_eager < 3:
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        self._step(batches)
                    torch.cuda.current_stream().wait_stream(side)
                    n_eager += 1
                else:
                    if graph is None:
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            self._step(static)
                    static.copy_(batches)
                    graph.replay()
            if (epoch + 1) % callback_every == 0 or epoch + 1 == epochs:
                loss_g, loss_d, n = self._acc.tolist()
                self._acc.zero_()
                self.history.append({"epoch": epoch + 1, "loss_G": loss_g / n, "loss_D": loss_d / n,
                                     "seconds": round(time.time() - t0, 1)})
                print("Epoch {} - loss_G: {:.4f}, loss_D: {:.4f} ({:.0f}s)".format(
                    epoch + 1, loss_g / n, loss_d / n, time.time() - t0), flush=True)
                if callback is not None:
                    callback(self, epoch + 1)

    def sample(self, labels, seed=None, ema=True):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(0 if seed is None else int(seed))
        labels_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=self.device)
        noise = torch.randn(len(labels_tensor), self.input_size, device=self.device, generator=gen)
        net = self.generator_ema if ema else self.generator
        with torch.no_grad():
            return net(noise, self._eye[labels_tensor]).cpu().numpy()

    def state_dict(self):
        return {"generator": self.generator.state_dict(), "generator_ema": self.generator_ema.state_dict(),
                "discriminator": self.discriminator.state_dict()}

    def load_state_dict(self, state):
        self.generator.load_state_dict(state["generator"])
        self.generator_ema.load_state_dict(state["generator_ema"])
        self.discriminator.load_state_dict(state["discriminator"])

    def training_state(self):
        """Everything needed to continue training: weights, optimiser states and loss history."""
        state = {"model": self.state_dict(), "optimizer_G": self.optimizer_G.state_dict(),
                 "optimizer_D": self.optimizer_D.state_dict(), "history": list(self.history)}
        return state

    def load_training_state(self, state):
        self.load_state_dict(state["model"])
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        self.history = list(state["history"])
