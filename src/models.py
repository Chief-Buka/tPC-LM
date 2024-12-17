import math
import numpy as numpy
import torch
from torch.nn.functional import relu, leaky_relu
from torch.nn.functional import cosine_similarity
from pathlib import Path 
import os

class TPC():
    def __init__(
        self, autoregressive=None,
        y_size=None, x_size=None, batch_size=None, delta_t_x=None, 
        delta_t_w=None, inf_iters=None, error_units=None, device=None
    ):

        self.autoregressive = autoregressive
        self.y_size = y_size
        self.x_size = x_size
        self.batch_size = batch_size # dynamic batch size to accomodate leftovers
        self.error_units = error_units
        self.inf_iters = inf_iters
        self.delta_t_x = delta_t_x
        self.delta_t_w = delta_t_w

        if y_size and x_size:
            self.Wxy = torch.randn(y_size, x_size).to(device) * 0.03
            self.Wxx = torch.randn(x_size, x_size).to(device) * 0.03
            self.Wyx = torch.randn(x_size, y_size).to(device) * 0.03

            self.bx = torch.zeros((x_size, 1)).to(device)
            self.by = torch.zeros((y_size, 1)).to(device)

            self.cx = torch.ones(x_size).to(device)
            self.cy = torch.ones(y_size).to(device)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.error_y = None
        self.error_x = None
        self.pred_y = None
        self.pred_x = None
        self.prev_y = None
        self.prev_x = None
        self.bottom_up = None
        self.top_down = None
        self.mask = None

    def __str__(self):
        output = [
            f"Model device: {self.device}", 
            f"y_size: {self.y_size}",
            f"x_size = {self.x_size}",
            f"Wxx_size = {self.Wxx.shape}",
            f"Wxy_size = {self.Wxy.shape}",
            f"Wyx_size = {self.Wyx.shape}",
            f"delta_t_x = {self.delta_t_x}",
            f"delta_t_w = {self.delta_t_w}",
            f"inf_iters = {self.inf_iters}",
            f"autoregressive = {self.autoregressive}",
            f"batch_size = {self.batch_size}",
            f"error_units = {self.error_units}"
        ]
        return "\n".join(output)
        

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

    # Want to avoid doing the exact same computation more than once since this method is called many times
    def step(self, t):

        # Observation Prediction
        self.pred_y = self.Wxy@self.g(self.x) + self.by
        
        # Observation Prediction Error
        self.error_y = self.y - self.pred_y

        # Precision Weighted Observation Predition Error 
        if self.error_units:
            self.pw_error_y = self.pw_error_y + self.delta_t_x * (
                self.error_y - self.pw_error_y #torch.einsum("n,nb->nb", (1.0/self.cy), self.pw_error_y)
            )
        else:
            self.pw_error_y = self.error_y #torch.einsum("n,nb->nb", (1.0/self.cy), self.error_y)

        if t == 0:
            # State Prediction
            if self.autoregressive:
                self.pred_x = self.Wxx@self.f(self.prev_x) + self.Wyx@self.h(self.prev_y) + self.bx
            else:
                self.pred_x = self.Wxx@self.f(self.prev_x) + self.bx


        # State Prediction Error
        self.error_x = self.x - self.pred_x

        # Precision Weighted State Prediction Error
        if self.error_units:
            self.pw_error_x = self.pw_error_x + self.delta_t_x * (
                self.error_x - self.pw_error_x #torch.einsum("n,nb->nb", (1.0/self.cx), self.pw_error_x)
            )
        else:
            self.pw_error_x = self.error_x #torch.einsum("n,nb->nb", (1.0/self.cx), self.error_x)

        # Calculate the gradient
        self.top_down = -self.pw_error_x
        self.bottom_up = self.g_deriv(self.x)*(self.Wxy.T@self.pw_error_y)

        # Update the Stateself.x_norm
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

            return result/total

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

            return result/total

        # for averaging across the batch 
        total = torch.sum(self.mask)


        ## State Prediction Parameters ##
        # Transition Parameters
        self.delta_Wxx = delta_W(code="xx")
        self.Wxx = self.Wxx + self.delta_t_w * self.delta_Wxx

        # Autoregressive Parameters
        if self.autoregressive:
            self.delta_Wyx = delta_W(code="yx")
            self.Wyx = self.Wyx + self.delta_t_w * self.delta_Wyx

        # Bias
        self.delta_bx = delta_b(code="x")
        self.bx = self.bx + self.delta_t_w * self.delta_bx


        ## Observation Prediction Parameters ##
        self.delta_Wxy = delta_W(code="xy")
        self.Wxy = self.Wxy + self.delta_t_w * self.delta_Wxy

        # Bias
        self.delta_by = delta_b(code="y")
        self.by = self.by + self.delta_t_w * self.delta_by

    def compute_energy(self):
        # for averaging across the batch 
        total = torch.sum(self.mask)

        energy = 0.5 * (
            torch.log(torch.prod(self.cx)) +
            torch.log(torch.prod(self.cy)) +
            torch.sum(torch.einsum("nb,b->bn", self.error_y*self.pw_error_y, self.mask))/total +
            torch.sum(torch.einsum("mb,b->bm", self.error_x*self.pw_error_x, self.mask))/total
        )
        return energy


    def reset(self, reset_state=False, reset_error=False):
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
        self.prev_y = torch.randn(self.y_size, self.batch_size).to(self.device) * 0.03

    def predict(self):
        # State Prediction
        if self.autoregressive:
            self.pred_x = self.Wxx@self.f(self.x) + self.Wyx@self.h(self.prev_y) + self.bx
        else:
            self.pred_x = self.Wxx@self.f(self.x) + self.bx
        # Observation Prediction
        self.pred_y = self.Wxy@self.g(self.pred_x) + self.by

    def save_parameters(self, epoch, energy, savedir):
        model_parameters = {
            "y_size": self.y_size,
            "x_size": self.x_size,
            "batch_size": self.batch_size,
            "Wxx": self.Wxx,
            "Wxy": self.Wxy,
            "Wyx": self.Wyx,
            "bx": self.bx,
            "by": self.by,
            "cy": self.cy,
            "cx": self.cx,
            "delta_t_x": self.delta_t_x,
            "delta_t_w": self.delta_t_w,
            "inf_iters": self.inf_iters,
            "energy": energy,
            "autoregressive": self.autoregressive
        }

        Path(savedir).mkdir(parents=True, exist_ok=True)
        torch.save(model_parameters, f"{savedir}/epoch_{epoch}.pt")

    def load_parameters(self, path):
        parameters = torch.load(path)
        self.y_size = parameters["y_size"]
        self.x_size = parameters["x_size"]
        self.batch_size = parameters["batch_size"]
        self.Wxx = parameters["Wxx"].to(self.device)
        self.Wxy = parameters["Wxy"].to(self.device)
        self.Wyx = parameters["Wyx"].to(self.device)
        self.bx = parameters["bx"].to(self.device)
        self.by = parameters["by"].to(self.device)
        self.cy = parameters["cy"].to(self.device)
        self.cx = parameters["cx"].to(self.device)
        self.delta_t_x = parameters["delta_t_x"]
        self.delta_t_w = parameters["delta_t_w"]
        self.inf_iters = parameters["inf_iters"]
        self.autoregressive = parameters["autoregressive"]

        print(f"Model Energy: {parameters['energy']:.3f}")