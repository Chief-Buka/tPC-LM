import math
import numpy as numpy
import torch
from torch.nn.functional import relu, leaky_relu
from torch.nn.functional import cosine_similarity

class AutoTPC():
    def __init__(self, autoregressive, y_size, x_size, batch_size, delta_t_x, 
    delta_t_w, inf_iters, error_units, device
    ):

    self.autoregressive = autoregressive
    self.y_size = y_size
    self.x_size = x_size
    self.batch_size = batch_size
    self.error_units = error_units
    self.inf_iters = inf_iters
    self.delta_t_x = delta_t_x
    self.delta_t_w = delta_t_w
    self.device = device

    self.Wxy = torch.randn(y_size, x_size).to(device) * 0.03
    self.Wxx = torch.randn(x_size, x_size).to(device) * 0.03
    self.Wyx = torch.randn(x_size, y_size).to(device)

    self.bx = torch.zeros((x_size, 1)).to(device)
    self.by = torch.zeros((y_size, 1)).to(device)

    self.cx = torch.ones(y_size).to(device)
    self.cy = torch.ones(x_size).to(device)

    self.error_y = None
    self.error_x = None
    self.pred_y = None
    self.pred_x = None
    self.prev_y = None
    self.prev_x = None
    self.bottom_up = None
    self.top_down = None
    self.mask = None

    def f(self, x):
        return torch.tanh(x)

    def f_deriv(self, x):
        return 1.0 - torch.tanh(x)**2

    def g(self, x):
        return x

    def g_deriv(self, x):
        return torch.ones(x.size()).to(self.device)

    def h(self, x):
        return torch.tanh(x)
    
    def h_deriv(self, x):
        return 1.0 - torch.tanh(x)**2

    def step(self):

        # Observation Prediction
        self.pred_y = self.Wxy@self.g(self.x) + self.by
        
        # Observation Prediction Error
        self.error_y = self.y - self.pred_y

        # Precision Weighted Observation Predition Error 
        if not self.error_units:
            self.pw_error_y = torch.einsum("n,nb->nb", (1.0/self.cy), self.error_y)
        else:
            pass

        # State Prediction
        self.pred_x = self.Wxx@self.f(self.prev_x) + self.Wyx@self.h(self.prev_y) + self.bx

        # State Prediction Error
        self.error_x = self.x - self.pred_x

        # Precision Weighted State Prediction Error
        if not self.error_units:
            self.pw_error_x = torch.einsum("n,nb->nb", (1.0/self.cx), self.error_x)
        else:
            pass

        # Calculate the gradient
        self.top_down = -self.pw_error_x
        self.bottom_up = self.g_deriv(self)*(self.W_xy.T@self.pw_error_y)

        # Update the State
        self.delta_x = self.top_down + self.bottom_up
        self.x = self.x + self.delta_t_x * self.delta_x

    
    def update_weights(self):
        
        # Helper function for weight matrix updates
        def delta_W(code):
            if code == "xx":
                arg1 = self.pw_error_x
                arg2 = self.f(self.prev_x)
            elif code == "yx":
                arg1 = self.pw_error_x
                arg2 = self.h(self.prev_y)
            elif code == "xy":
                arg1 = self.pw_error_y
                arg2 = self.g(self.x)

            result = torch.sum(
                torch.einsum("bnm,b->bnm",
                    torch.einsum("nb,mb->bnm", arg1, arg2),
                    self.mask
                ),
                axis=0
            )

            return result

        # Helper function for bias vector updates
        def delta_b(code):
            if code == "x":
                arg = self.pw_error_x
            elif code == "y":
                arg = self.pw_error_y

            result = torch.unsqueeze(torch.sum(
                torch.einsum("nb,b->bn", arg, self.mask),
                axis=0
            ), dim=-1)

            return result

        # for averaging across the batch 
        total = torch.sum(self.mask)


        ## State Prediction Parameters ##
        # Transition Parameters
        self.delta_Wxx = delta_W(code="xx")
        self.Wxx = self.Wxx + self.delta_t_w * (self.delta_Wxx/total)

        # Autoregressive Parameters
        self.delta_Wyx = delta_W(code="yx")
        self.Wyx = self.Wyx + self.delta_t_w * (self.delta_Wyx/total)

        # Bias
        self.delta_bx = delta_b(code="x")
        self.bx = self.bx + self.delta_t_w * (self.delta_bx/total)


        ## Observation Prediction Parameters ##
        self.delta_Wxy = delta_W(code="xy")
        self.Wxy = self.Wxy + self.delta_t_w * (self.delta_Wxx/total)

        # Bias
        self.delta_by = delta_b(code="y")
        self.by = self.by + self.delta_t_w * (self.delta_by/total)

    def compute_energy(self):
        # for averaging across the batch 
        total = torch.sum(self.mask)

        # The negative energy = cost function maximized using gradient ascent
        # = the negative of the term we are minimizing
        energy = -0.5 * (
            torch.log(torch.prod(self.cx)) +
            torch.log(torch.prod(self.cy)) +
            torch.sum(torch.einsum("nb,b->bn", self.error_y*self.pw_error_y, self.mask))/total +
            torch.sum(torch.einsum("mb,b->bm", self.error_x*self.pw_error_x, self.mask))/total
        )
        return energy

    def compute_cosdis(self):
        cosine_distance = 1. - cosine_similarity(self.pred_y, self.y, dim=0)
        return torch.sum(cosine_distance * self.mask)/torch.sum(self.mask)

    def reset(self, reset_state, reset_error):
        if reset_state:
            self.x = torch.randn(self.x_size, self.batch_size).to(self.device) * 0.001
        if reset_error:
            self.pw_error_x = torch.zeros((self.x_size, self.batch_size)).to(self.device)
            self.pw_error_y = torch.zeros((self.y_size, self.batch_size)).to(self.device)

    def update_prev(self):
        self.prev_x = torch.clone(self.x)
        self.prev_y = torch.clone(self.y)

    def set_random_prev(self):
        self.prev_x = torch.randn(self.x_size, self.batch_size).to(self.device) * 0.03
        self.prev_y = torch.randn(self.x_size, self.batch_size).to(self.device) * 0.03

    def predict(self):
        # State Prediction
        self.pred_x = self.Wxx@self.f(self.x) + self.Wyx@self.h(self.prev_y) + self.bx
        # Observation Prediction
        self.pred_y = self.Wxy@self.g(self.pred_x) + self.by

    def save_parameters(self, epoch, energy):
        model_parameters = {
            "y_size": model.y_size,
            "x_size": model.x_size,
            "batch_size": model.batch_size,
            "Wxx": model.Wxx,
            "Wxy": model.Wxy,
            "Wyx": model.Wyx,
            "bx": model.bx,
            "by": model.by,
            "cy": model.cy,
            "cx": model.cx,
            "delta_t_x": model.delta_t_x,
            "delta_t_w": model.delta_t_w,
            "inf_iters": model.inf_iters,
            "energy": energy
        }

        torch.save(model_paramaters, f"{savedir}/epoch_{epoch}.pt")

    def load_parameters(self, path):
        parameters = torch.load(path)
        model.y_size = parameters["y_size"]
        model.x_size = parameters["x_size"]
        model.batch_size = parameters["batch_size"]
        model.Wxx = parameters["Wxx"]
        model.Wxy = parameters["Wxy"]
        model.Wyx = parameters["Wyx"]
        model.bx = parameters["bx"]
        model.by = parameters["by"]
        model.cy = parameters["cy"]
        model.cx = parameters["cx"]
        model.delta_t_x = parameters["delta_t_x"]
        model.delta_t_w = parameters["delta_t_w"]
        model.inf_iters = parameters["inf_iters"]

        print(f"Model Energy: {parameters["energy"]:.3f}")