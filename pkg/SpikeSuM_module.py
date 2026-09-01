# coding: utf-8
"""EI-mismatch network, prediction and error estimation"""
import os
from scipy.sparse import rand
import matplotlib.pyplot as plt
import math
import torch
import network_utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RECORD_ACTIVITY = os.environ.get("SPIKESUM_RECORD_ACTIVITY", "0") == "1"
RECORD_WEIGHTS = os.environ.get("SPIKESUM_RECORD_WEIGHTS", "0") == "1"
T_HAT_EVERY = int(os.environ.get("SPIKESUM_T_HAT_EVERY", "0"))


class SpikeSuM(object):
    """Network Class"""

    def __init__(
        self,
        params,
    ):
        """
        Initialize SpikeSuM class
        params: Dictionnary including all network params. More details in
            results/Set_SpikeSuM-M_params.ipynb
        """
        # Environment properties
        self.number_rooms = params["number_rooms"]
        self.states = params["states"]

        # Network properties
        self.n_memory = params["n_memory"]
        self.EI_neurons = params["EI_neurons"]
        self.input_neurons = params["input_neurons"]
        self.batch_size = params['batch_size']
        self.sparsity = 0.1 ## Hard coded because never change.
        self.plot = params["plot"]
        self.modulation = params["modulation"]
        # Set from outside by scripts/spikesum.py ('HD', 'PE' or None). It was
        # never initialised here, so forward() raised AttributeError whenever
        # the class was used without going through that script.
        self.tosave = None
        
        self.rooms_encoding = torch.zeros(
            (self.batch_size, self.number_rooms, self.input_neurons)).to(device)
        
        # Neuron model
        self.tau = params["tau"]
        self.eta1 = params["eta1"]
        self.eta2 = params["eta2"]
        self.theta = torch.Tensor([params["theta"]]).to(device)

        # ---- Self-calibrating surprise threshold (off by default) ---------
        #     theta = a + b * ln(K_est)
        # Additive mode: theta = EMA(A~) + c. The margin is invariant, so if
        # learning drags A~ down theta follows by the same amount and the
        # crossing rate is unchanged -- loop gain zero to first order, no
        # freezing needed. Measured drift -0.4% over a 5x longer run, against
        # -64% for a multiplicative rule.
        #
        # Hybrid: theta = EMA(A~) + a + b*ln(K_est). The mean tracks the level,
        # the histogram supplies the margin, which is the only term that
        # changes sign with K (-0.085 at K=2, +0.114 at K=8).
        self.hybrid_theta = params.get("hybrid_theta", False)
        self.mean_theta = params.get("mean_theta", False)
        if self.hybrid_theta:
            self.hyb_a = params.get("hyb_a", -0.2470)
            self.hyb_b = params.get("hyb_b", 0.1810)
        if self.mean_theta or self.hybrid_theta:
            self.mean_c = params.get("mean_c", 0.05)
            self.mean_tau = params.get("mean_tau", 300.0)
            self.mean_warmup = params.get("mean_warmup", 200)
            self.mean_lam = float(math.exp(-1.0 / self.mean_tau))
            self.mean_A = None
            self.mean_epoch = 0

        self.exo_theta = params.get("exo_theta", False)
        if self.exo_theta:
            self.exo_a = params.get("exo_a", -0.0833)
            self.exo_b = params.get("exo_b", 0.3703)
            self.exo_freeze = params.get("exo_freeze", 900)
            tau_hist = params.get("exo_tau", 300.0)
            self.exo_lam = float(math.exp(-1.0 / tau_hist))
            self.exo_C = torch.full(
                (self.number_rooms, self.number_rooms), 1e-3).to(device)
            self.exo_prev = None
            self.exo_room = None
            self.exo_epoch = 0
            self.exo_keff = float(self.number_rooms)
            self.exo_frozen = None
        self.N = params["N"]
        self.Poisson_rate = params["Poisson_rate"]
        self.tau = params["tau"]
        self.len_epsc = params["l"]
        self.decay = 0.9 ## Hard coded because never change.
        self.sign = torch.Tensor([[1], [-1]]).to(device)
        self.scale = (self.EI_neurons / 128)
        self.FB_inhib = params["FB_inhib"]
        self.decay_factor =  (1 - torch.exp(torch.Tensor([-1 / self.tau]))).to(device)
        
        # Initialiste the layer population
        self.initiate_layer()
       

        for i in range(self.number_rooms):
            self.rooms_encoding[:, i, self.states[i]] = 1.
        # Weights initialisation
        self.W = params["W"]
        if self.W is None:
            # Scaling is important for

            self.W =  torch.rand(self.batch_size,2, self.input_neurons, self.n_memory * self.EI_neurons).to(device) / (params['W_init'] * self.scale)
        self.observation_weights, self.readout_weights = self.initiate_feedback(
            params["random_projection"])
        if not params["random_projection"]:
            self.A0 = (
                .2
                * self.EI_neurons
                / self.number_rooms
                * self.phi(torch.Tensor([1]).to(device))
                * (self.input_neurons / self.number_rooms)
            )  # Expectation of maximum neuron active at the same time
        else:
            self.A0 = (
                self.sparsity
                * .2
                * self.EI_neurons
                * self.phi(torch.log(torch.cosh(torch.Tensor([1]).to(device))))
                * (self.input_neurons / self.number_rooms)
            )

        self.initiate_info()
    def initiate_layer(self):
        """
        Set all main layer dependencies
        """
         # Layer initialisation
        self.h = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.u = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.refractoriness = torch.zeros_like(self.h)
        self.EPSC_EI_decay = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.filtered_activity = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.filtered_activity_nohin = torch.zeros(
            (self.batch_size, 2, self.n_memory * self.EI_neurons)).to(device)
        self.filtered_theta = torch.zeros(
            self.batch_size, self.n_memory).to(device)
        self.filtered_EPSC = torch.zeros(
            (self.batch_size, 2, self.input_neurons)).to(device)
        self.network_activity = torch.zeros(self.n_memory)
        self.output = torch.zeros(
            self.batch_size,
            2 * self.EI_neurons).to(device)
    def initiate_info(self):
        """
        Instantiate all info to save from a simulation
        """
        # plot_network() reads info['Activity'], so recording has to be on
        # whenever this module plots, regardless of the environment variable.
        self._record_activity = RECORD_ACTIVITY or getattr(self, "plot", False)
        self.info = {}
        self.info["Activity"] = []
        self.info["Activity_full"] = [[] for _ in range(self.n_memory)]
        self.info["Activity_P1"] = []
        self.info["Activity_P2"] = []
        self.info["error"] = []
        self.info["weights"] = []
        self.info["spikes"] = []
        self.info["EPSC"] = []
        self.info["T1"] = []
        self.info["T2"] = []
        self.info["T_hat"] = torch.zeros(
            (self.number_rooms, self.number_rooms))
        self.info["Learning_rate"] = []
        self.info["readout_weights"] = self.readout_weights
        self.info["EI_spikes"] = []
        self.info["weights_evolution"] = [[],[]]
        self.info["T_hat_history"] = []
        for drives in ['PE','HD']:
            self.info['absolute error' + drives] = []
            self.info['effective update' + drives] = []
            self.info['prediction error' + drives] = []
            self.info['effective update_pos'+drives] = []
            self.info['prediction error_pos'+drives] = []
            self.info['effective update_neg'+drives] = []
            self.info['prediction error_neg'+drives] = []
            self.info['self_third' + drives] = []
            self.info['activity' + drives] = []
            self.info['activity_inh'+ drives] = []
    def initiate_feedback(self, random_projection=False):
        """
        Observation weight initialisation.

        param random_projection: True/False; declares whether room observation is one hot encoded or randomly projected

        returns: Observation weights (Projection onto the error layer); readout weights (allow memory wise decoding of Prediction weights)
        """

        readout_weights = []
        if not random_projection:
            observation_weights = torch.zeros(
                (self.batch_size, 2, self.input_neurons, self.n_memory * self.EI_neurons)).to(device)
            diff = self.input_neurons / self.EI_neurons
            for memory in range(self.n_memory):

                for room in self.rooms_encoding[0]:

                    nk = torch.sum(room) * self.Poisson_rate * self.len_epsc

                    idx = torch.nonzero(room)[0][0]
                    length = int(torch.sum(room))
                    observation_weights[:, :, idx: idx + length, int(1.0 * idx / diff) + memory * self.EI_neurons: int(
                        1.0 * (idx + length) / diff + memory * self.EI_neurons), ] = 2.0 / nk
        if random_projection:
            sparsify = rand(
                
                2 * self.batch_size * self.input_neurons,
                self.n_memory * self.EI_neurons,
                density=self.sparsity,
                format="csr",
            )
            sparsify.data[:] = 1.
            observation_weights = torch.rand(
                self.batch_size,
                2,
                self.input_neurons,
                self.n_memory *
                self.EI_neurons).float()
            observation_weights *= sparsify.toarray().reshape(self.batch_size, 2 , self.input_neurons,
                self.n_memory * self.EI_neurons)
            observation_weights = observation_weights.float()
        for i in range(2):
            self.R = self.rooms_encoding.clone()
            pop_weights = []
            for memory in range(self.n_memory):
                P = observation_weights[:, i, :, memory *
                                        self.EI_neurons:memory *
                                        self.EI_neurons +
                                        self.EI_neurons].clone().to(device)
                readout = torch.pinverse(self.R @ P)
                pop_weights += [torch.unsqueeze(readout, 1)]
            readout_weights += [
                torch.unsqueeze(
                    torch.cat(
                        pop_weights,
                        dim=1),
                    1)]
        readout_weights = torch.cat(readout_weights, dim=1).to(device)
        return observation_weights.to(device), readout_weights.to(device)

    def estimate_T(self):
        """
        Decode prediction weights

        return: The estimated transition matrix $$T=\frac{1}{2}(T_1+T_2)$$
        """
        shape = self.W.shape
        W_reshaped = self.W.view(
            shape[0],
            shape[1],
            shape[2],
            self.n_memory,
            int(shape[3]/self.n_memory)).permute(
            0,
            1,
            3,
            2,
            4)
        Transition_hidden_space = torch.einsum(
            "bijkl,bijlm->bijkm", W_reshaped, self.readout_weights)
        Transition_one_hot_space = torch.einsum(
            "bijkl,bmk->bijlm", Transition_hidden_space, self.R )
        T = torch.mean(Transition_one_hot_space, dim=1)  # T1 + T2 average
        #T = torch.transpose(T, 2, 3)
        T = torch.nn.functional.normalize(T, p = 1.0, dim = 2)
        return T

    def clear_spike_train(self):
        """
        Delete the spike train we keep in memory
        """
        self.info["EI_spikes"] = []

    def phi(self, x):
        """
        error neuron activation function

        param x: neuron membrane potential

        return $$f(x)$$
        """
        return (x > 0).float() * torch.tanh(x)

    def decode_observed_room(self, EPSC_observation):
        """Which room the observation population currently encodes.

        rooms_encoding[b, r, :] is the m-hot pattern of room r, so a dot
        product with the observation plus an argmax is what a downstream
        readout would do. Decoding from the input rather than reading the
        simulation's index matters: otherwise the mechanism would be consulting
        a variable the network never sees. Verified, 118/118 correct.
        """
        obs = EPSC_observation.detach()
        obs = obs.mean(dim=1) if obs.dim() == 3 else obs
        score = torch.einsum("brn,bn->br", self.rooms_encoding, obs)
        if float(score.max()) > 0:
            self.exo_room = int(torch.argmax(score[0]))

    def update_mean_theta(self):
        """theta = EMA(A~) + margin. Additive, so the margin is invariant.

        Only rectified activity is averaged: in SpikeSuM-C the modules the
        selector inhibits sit at A~ around -9595 and would drag the mean.
        """
        a = self.network_activity.detach()
        act = a[a > 0]
        if act.numel():
            cur = float(act.mean())
            self.mean_A = cur if self.mean_A is None else (
                self.mean_A + (1 - self.mean_lam) * (cur - self.mean_A))
        self.mean_epoch += 1
        if self.mean_A is None or self.mean_epoch <= self.mean_warmup:
            return
        if self.hybrid_theta:
            k = max(self.exo_keff, 1.05)
            margin = self.hyb_a + self.hyb_b * math.log(k)
        else:
            margin = self.mean_c
        self.theta = torch.full_like(
            torch.as_tensor(self.theta).float().reshape(-1),
            self.mean_A + margin).reshape(1, -1)

    def update_exo_theta(self):
        """Slow histogram of observed transitions -> K_est -> theta."""
        self.exo_epoch += 1
        if self.exo_room is not None and self.exo_prev is not None:
            self.exo_C *= self.exo_lam
            self.exo_C[self.exo_prev, self.exo_room] += (1 - self.exo_lam)
        self.exo_prev = self.exo_room

        if self.exo_epoch % 10 == 0:
            P = self.exo_C / self.exo_C.sum(dim=1, keepdim=True)
            self.exo_keff = float((1.0 / (P ** 2).sum(dim=1)).mean())

        if self.hybrid_theta:
            return
        if self.exo_epoch == self.exo_freeze:
            k = max(self.exo_keff, 1.05)
            self.exo_frozen = self.exo_a + self.exo_b * math.log(k)
        if self.exo_frozen is not None:
            self.theta = torch.full_like(
                torch.as_tensor(self.theta).float().reshape(-1),
                self.exo_frozen).reshape(1, -1)

    def third_factor(self, x):
        """
        Modulatory learning signal

        param x: network activity

        param self.modulation: define whether we use constant learning rate, single of full modulation

        param self.theta: Surprise level, if $$x>\theta$$ the agent is considered in a Surprised state

        return: Surprise modulatory signal signal
        """
        if self.modulation == 'full':
            return (self.eta1 * torch.tanh((x)) + self.eta2 * \
                    torch.tanh(x) * (x >= self.theta).float())  * (x > 0).float()
        elif self.modulation == 'single':
            return self.eta1 * torch.tanh((x)) * (x > 0).float()
        elif self.modulation == 'step':
            return (self.eta1 +
                    self.eta2 *
                    (x >= self.theta).float() *
                    (x > 0).float())
        elif self.modulation == 'none':
            return self.eta1 * (x > 0)

    def update_pot(self, h, I):  # noqa: E741
        """
        Layer update

        param h: input potential

        param I: input current

        return: Integrated potential
        """

        h += 1 / self.tau * (-h + I)
        return h

    def update_layer(self, I, EPSC_decay):  # noqa: E741
        """
        Full update of the error layer

        Param I: Input current receive from both prediction and observation

        Param EPSC_decay: Estimation of time since last spike of error neurons
        
        return: Spikes, EPSC,  time since last spike (in the form of decaying EPSC)
        """

        spikes = 0 * self.h
        self.h = self.update_pot(self.h, I)
        self.u = self.h.clone() - self.refractoriness
        self.ratio = torch.mean(self.u[self.u > 0])

        idx = self.phi(self.u) >  torch.Tensor(self.batch_size,
                                                    2, self.n_memory * self.EI_neurons).uniform_().to(device)
        spikes[idx] = 1.0
        EPSC, EPSC_decay = network_utils.square_EPSC(EPSC_decay, self.len_epsc, spikes)
        self.refractoriness *= self.decay
        self.refractoriness[idx] = 1
        return spikes.detach(), EPSC.detach(), EPSC_decay.detach()

    def save_prediction(self, current_T_matrix):
        """
        Saving the Matrix transition estimation error

        param current_T_matrix: True maze transition matrix to be estimaed

        Return None
        """

        if self.exo_theta or self.hybrid_theta:
            self.update_exo_theta()
        if self.mean_theta or self.hybrid_theta:
            self.update_mean_theta()

        T_hat = self.estimate_T()

        self.info["error"] += [
            torch.mean(
                (current_T_matrix - T_hat)**2,
                dim=(
                    2,
                    3)).clone().detach()]
        self.info["T_hat"] = T_hat.clone().detach()

        # T_hat above is overwritten on every call, so only the final estimate
        # survives into the saved pickle. Optionally keep a periodic history:
        # that is the time axis of Fig 6E, which cannot be recovered afterwards.
        if T_HAT_EVERY:
            self._t_hat_calls = getattr(self, "_t_hat_calls", 0) + 1
            if (self._t_hat_calls - 1) % T_HAT_EVERY == 0:
                self.info["T_hat_history"] += [
                    (self._t_hat_calls - 1, T_hat.clone().detach().cpu())]

    def record_weight_evolution(self):
        """Coarse per-module summary of the plastic weights.

        Averages the weights within each stimulus cluster, giving one
        matrix per context module and per population.

        The original version computed the postsynaptic block size from
        input_neurons, but the postsynaptic axis of W is n_memory * EI_neurons.
        With the shipped parameters (128 vs 512) the view raised a RuntimeError,
        and even with the right size it collapsed the modules into each other
        whenever n_memory > 1. The reshape below keeps the module axis separate.
        """
        rooms = self.number_rooms
        pre = self.input_neurons // rooms            # sensory neurons per stimulus
        post = self.EI_neurons // rooms              # error neurons per stimulus
        out = []
        for population in range(2):
            w = self.W[:, population].view(
                self.batch_size, rooms, pre, self.n_memory, rooms, post)
            # average over batch, and within each pre/post cluster
            out += [torch.mean(w, dim=(0, 2, 5)).permute(1, 0, 2).cpu()]
        self.info["weights_evolution"][0] += [out[0].unsqueeze(0)]
        self.info["weights_evolution"][1] += [out[1].unsqueeze(0)]


    def forward(
        self, EPSC_buffer, EPSC_observation, module_inhib, learning=False
    ):
        """
        SpikeSumNet step

        param EPSC_buffer: The active neurons in the buffer population

        param EPSC_observation: The active neurons in the observaiton  population

        module_inhib: Feedback inhibition coming from the dishinibitory modules (if multiple memories)

        param learning: switch learning off (debugging only)

        return: Average weight update and commitment modulation for disinhibitory neurons
        """
        
        if self.exo_theta or self.hybrid_theta:
            self.decode_observed_room(EPSC_observation)

        I = (self.sign * (torch.einsum("bijk,bij->bik",  # noqa: E741
                                       self.W,
                                       EPSC_buffer) - torch.einsum("bijk,bij->bik",
                                                                   self.observation_weights,
                                                                   EPSC_observation)).detach())
        (
            self.EI_spikes,
            EPSC_EI,
            self.EPSC_EI_decay,
        ) = self.update_layer(I, self.EPSC_EI_decay)
        self.filtered_activity = self.filtered_activity + (1.0 / self.N) * (
            - self.filtered_activity + (self.EI_spikes  - self.FB_inhib * module_inhib) / self.A0 
        ).detach()
        # filtered_activity_nohin is the same low-pass filter without the
        # inhibition from the other modules. It feeds the P1/P2 activity traces
        # and save_drives, and is used nowhere else, so it is only worth
        # computing when one of those is actually being recorded.
        if self._record_activity or self.tosave is not None:
            self.filtered_activity_nohin = self.filtered_activity_nohin + (1.0 / self.N) * (
                - self.filtered_activity_nohin + (self.EI_spikes) / self.A0 
            ).detach()
        memory_activity = torch.sum(
            self.filtered_activity, dim=1).reshape(
            self.batch_size, self.n_memory, -1)
        self.network_activity = torch.sum(memory_activity, dim=2).detach()
        if self._record_activity:
            self.info["Activity_P1"] += [torch.mean(torch.sum(self.filtered_activity_nohin[:,0],dim = 1)).cpu()]
            self.info["Activity_P2"] += [torch.mean(torch.sum(self.filtered_activity_nohin[:,1],dim = 1)).cpu()]
            self.info["Activity"] += [self.network_activity.detach().clone()]

        self.info["Activity_P1"] += [torch.mean(torch.sum(self.filtered_activity_nohin[:,0],dim = 1)).cpu()]
        third = self.third_factor(self.network_activity).detach()
        commitement_modulation = (third > 0 ) * (1 - 2 * (third > self.third_factor(self.theta)))
        prediction_modulation = third.detach()
        third = torch.unsqueeze(
            third.repeat_interleave(
                self.EI_neurons,
                1),
            dim=1).repeat_interleave(
            2,
            dim=1).detach()

        if learning:
            self.filtered_EPSC = self.decay_factor * self.filtered_EPSC + EPSC_buffer * (1 - self.decay_factor)
            deltaW = (
                torch.einsum(
                    "bij,bik->bijk",
                    self.filtered_EPSC,
                    third *
                    self.h)).detach()
            self.W -= torch.einsum("ij,bikl->bikl", self.sign, deltaW).detach()
            self.W[self.W < 0] = 0
            self.W = self.W.detach()
            if self.tosave is not None:
                self.save_drives(third, deltaW)
        self.output = torch.sum(EPSC_EI, dim = 1).detach() / 2
        self.input = self.network_activity .detach()
        if RECORD_WEIGHTS:
            self.record_weight_evolution()
        
        return prediction_modulation, commitement_modulation
    
    def save_drives(self,third,deltaW):
    ## Prediction error Drive 
        if self.batch_size == 1:
            for drives in ['PE','HD']:
                if drives == 'PE':
                    third_per_memory = torch.mean(third, axis = 1).reshape(-1,self.n_memory, self.EI_neurons).clone().detach()
                    third_per_memory = torch.mean(third_per_memory, axis = -1)
                    activity_per_memory = torch.sum(self.filtered_activity, axis = 1).reshape(-1,self.n_memory, self.EI_neurons)
                    activity_per_memory = torch.sum(activity_per_memory, axis = -1)
                    activity_per_memory_nohin = torch.sum(self.filtered_activity_nohin, axis = 1).reshape(-1,self.n_memory, self.EI_neurons)
                    activity_per_memory_nohin = torch.sum(activity_per_memory_nohin, axis = -1)

                    for memory in range(self.n_memory):
                        self.info['self_third' + drives] += [third_per_memory[0,memory].clone().detach()]
                        self.info['activity' + drives] += [activity_per_memory[0,memory].clone().detach()]
                        self.info['activity_inh'+ drives] += [activity_per_memory_nohin[0,memory].clone().detach()]
                    self.info['effective update'+drives] += [torch.mean((third * torch.abs(self.h)).detach().clone(),axis=1)]
                    self.info['prediction error'+drives] += [torch.mean(torch.abs(self.h).detach().clone(),axis=1)]
                ## Hebbian Drive
                if drives == 'HD':
                    Hebbian_drive = torch.einsum(
                               "bij,bik->bijk",
                               self.filtered_EPSC,
                               self.h)
                    self.info['effective update_pos'+drives] += [torch.abs(deltaW[0,0][Hebbian_drive[0,0] != 0]).detach().clone().cpu()]
                    self.info['prediction error_pos'+drives] += [torch.abs(Hebbian_drive[0,0][Hebbian_drive[0,0] != 0]).detach().clone().cpu()]
                    self.info['effective update_neg'+drives] += [torch.abs(deltaW[0,1][Hebbian_drive[0,1] != 0]).detach().clone().cpu()]
                    self.info['prediction error_neg'+drives] += [torch.abs(Hebbian_drive[0,1][Hebbian_drive[0,1] != 0]).detach().clone().cpu()]
                    
                
    def plot_network(self):
        """
        Plotting few properties of the network; the estimated transition matrix error, the activity  and finally the estimated transition matrix.
        This is shown for every memories. The transition matrix shows the estimated transition for every memory.
        """
        if self.plot:
            print("----- Plot SpikeSuM Module:-----")

            for memory in range(self.n_memory):
                plt.plot(torch.cat(
                    self.info['error']).view(-1, self.n_memory)[:, memory].cpu().detach())
            plt.title('Estimated transition error')
            plt.show()
            for memory in range(self.n_memory):
                x = torch.mean((torch.reshape(torch.cat(
                    self.info['Activity']).view(-1, self.n_memory)[:, memory], (-1, 100))), dim=1)
                plt.plot((x * (x > 0)).cpu().detach())
            plt.title('memory Activity')
            plt.show()
            T = self.info["T_hat"]
            T = T.reshape(-1, T.shape[-1])
            T = torch.nn.functional.normalize(T, p = 1.0, dim = 1)
            plt.imshow(T.cpu().detach().T, aspect=1)
            plt.colorbar()
            plt.show()
