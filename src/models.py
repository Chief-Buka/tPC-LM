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

            self.cx = torch.ones(x_size).to(device) * 0.5
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

        self.negative_slope = 0.01

    def __str__(self):
        output = [
            f"Model device = {self.device}", 
            f"y_size = {self.y_size}",
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
        #return torch.tanh(x)
        return leaky_relu(x, self.negative_slope)

    def f_deriv(self, x):
        #return 1.0 - torch.tanh(x)**2
        return (self.negative_slope) + (1-self.negative_slope)*(x>0)

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
                self.error_y - torch.einsum("n,nb->nb", (1.0/self.cy), self.pw_error_y)
            )
        else:
            self.pw_error_y = torch.einsum("n,nb->nb", (1.0/self.cy), self.error_y)

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
                self.error_x - torch.einsum("n,nb->nb", (1.0/self.cx), self.pw_error_x)
            )
        else:
            self.pw_error_x = torch.einsum("n,nb->nb", (1.0/self.cx), self.error_x)

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

            batch_result = torch.einsum(
                "bnm,b->bnm",
                torch.einsum("nb,mb->bnm", arg1, arg2),
                self.mask
            )

            result = torch.sum(
                batch_result,
                axis=0
            ) / total

            return result, batch_result

        # Helper function for bias vector updates
        def delta_b(code):
            if code == "x":
                arg = self.pw_error_x
            elif code == "y":
                arg = self.pw_error_y

            batch_result = torch.einsum(
                "nb,b->bn", 
                arg, 
                self.mask
            )

            result = torch.unsqueeze(torch.sum(
                batch_result,
                axis=0
            ), dim=-1) / total

            return result, batch_result

        # for averaging across the batch 
        total = torch.sum(self.mask)


        ## State Prediction Parameters ##
        # Transition Parameters
        self.delta_Wxx, self.batched_delta_Wxx = delta_W(code="xx")
        self.Wxx = self.Wxx + self.delta_t_w * self.delta_Wxx

        # Autoregressive Parameters
        if self.autoregressive:
            self.delta_Wyx, self.batched_delta_Wyx = delta_W(code="yx")
            self.Wyx = self.Wyx + self.delta_t_w * self.delta_Wyx

        # Bias
        self.delta_bx, self.batched_bx = delta_b(code="x")
        self.bx = self.bx + self.delta_t_w * self.delta_bx


        ## Observation Prediction Parameters ##
        self.delta_Wxy, self.batched_delta_Wxy = delta_W(code="xy")
        self.Wxy = self.Wxy + self.delta_t_w * self.delta_Wxy

        # Bias
        self.delta_by, self.batched_by = delta_b(code="y")
        self.by = self.by + self.delta_t_w * self.delta_by

    def compute_energy(self, batched_result=False):
        # for averaging across the batch 
        total = torch.sum(self.mask)

        # error_y and pw_error_y are nb
        # error_x and pw_error_x are mb

        if batched_result:
            energy = 0.5 * (
                #torch.log(torch.prod(self.cx)) +  <-- this is just zero since cov = I
                #torch.log(torch.prod(self.cy)) +  <-- this is just zero since cov = I
                torch.sum(self.error_y*self.pw_error_y, dim=0) * self.mask +
                torch.sum(self.error_x*self.pw_error_x, dim=0) * self.mask
            )
        else:
            # energy = 0.5 * (
            #     torch.log(torch.prod(self.cx)) +
            #     torch.log(torch.prod(self.cy)) +
            #     torch.sum(torch.einsum("nb,b->bn", self.error_y*self.pw_error_y, self.mask))/total +
            #     torch.sum(torch.einsum("mb,b->bm", self.error_x*self.pw_error_x, self.mask))/total
            # )

            energy = 0.5 * (
                #torch.log(torch.prod(self.cx)) +
                #torch.log(torch.prod(self.cy)) +
                torch.sum(torch.sum(self.error_y*self.pw_error_y, dim=0) * self.mask)/total +
                torch.sum(torch.sum(self.error_x*self.pw_error_x, dim=0) * self.mask)/total
            )

        return energy

    def compute_cosine_distance(self, batched_result=False):
        if batched_result:
            obs_cos_dist = 1 - cosine_similarity(self.Wxy@self.g(self.pred_x) + self.by, self.y, dim=0)
            obs_cos_dist = obs_cos_dist * self.mask
        return obs_cos_dist


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
            "autoregressive": self.autoregressive,
            "epoch": epoch
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

    
    def eval_mode(self, batch_size, max_inf_iters):
        self.batch_size = batch_size
        self.delta_t_w = 0.0
        self.inf_iters = max_inf_iters
        self.error_units = False


class AutoTPC():
    def __init__(
        self, K=0,
        y_size=None, x_size=None, batch_size=None, delta_t_x=None, 
        delta_t_w=None, inf_iters=None, error_units=None, device=None,
        f_type='linear', g_type='linear', h_type='linear'
    ):

        self.K = K
        self.y_size = y_size
        self.x_size = x_size
        self.batch_size = batch_size # dynamic batch size to accomodate leftovers
        self.error_units = error_units
        self.inf_iters = inf_iters
        self.delta_t_x = delta_t_x
        self.delta_t_w = delta_t_w
        self.epoch = 0

        if y_size and x_size:
            self.Wxy = torch.randn(y_size, x_size).to(device) * 0.03
            self.Wxx = torch.randn(x_size, x_size).to(device) * 0.03
            self.Wyx = []
            for k in range(self.K):
                self.Wyx.append(torch.randn(x_size, y_size).to(device) * 0.03)

            self.bx = torch.zeros((x_size, 1)).to(device)
            self.by = torch.zeros((y_size, 1)).to(device)

            self.cx = torch.ones(x_size).to(device) * 0.5
            self.cy = torch.ones(y_size).to(device)

            self.py = (1.0/self.cy.unsqueeze(dim=1))
            self.px = (1.0/self.cx.unsqueeze(dim=1))

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
        self.f_type = f_type
        self.g_type = g_type
        self.h_type = h_type

        self.negative_slope = 0.01

    def __str__(self):
        output = [
            f"Model device = {self.device}", 
            f"y_size = {self.y_size}",
            f"x_size = {self.x_size}",
            f"Wxx_size = {self.Wxx.shape}",
            f"Wxy_size = {self.Wxy.shape}",
            f"markov order = {len(self.Wyx)}",
            f"delta_t_x = {self.delta_t_x}",
            f"delta_t_w = {self.delta_t_w}",
            f"inf_iters = {self.inf_iters}",
            f"batch_size = {self.batch_size}",
            f"error_units = {self.error_units}",
            f"epoch = {self.epoch}",
            f"f_type = {self.f_type}",
            f"g_type = {self.g_type}"
        ]
        if len(self.Wyx) != 0:
            output.append(f"Wyx_size = {self.Wyx[0].shape}")
            output.append(f"h_type = {self.h_type}")
        return "\n".join(output)
        

    def f(self, x):
        if self.f_type == 'linear':
            return x
        elif self.f_type == 'leaky_relu':
            return leaky_relu(x, self.negative_slope)

    def f_deriv(self, x):
        if self.f_type == 'linear':
            return torch.ones(x.size()).to(self.device)
        elif self.f_type == 'leaky_relu':
            return (self.negative_slope) + (1-self.negative_slope)*(x>0)

    def g(self, x):
        if self.g_type == 'linear':
            return x
        elif self.g_type == 'leaky_relu':
            return leaky_relu(x, self.negative_slope)

    def g_deriv(self, x):
        if self.g_type == 'linear':
            return torch.ones(x.size()).to(self.device)
        elif self.g_type == 'leaky_relu':
            return (self.negative_slope) + (1-self.negative_slope)*(x>0)

    def h(self, x):
        if self.h_type == 'linear':
            return x
        elif self.h_type == 'leaky_relu':
            return leaky_relu(x, self.negative_slope)
    
    def h_deriv(self, x):
        if self.h_type == 'linear':
            return torch.ones(x.size()).to(self.device)
        elif self.h_type == 'leaky_relu':
            return (self.negative_slope) + (1-self.negative_slope)*(x>0)

    def solve(self):
        self.predict(y=False) # compute self.pred_x
        temp = self.Wxy.T * self.py.squeeze() #
        lhs = temp@self.Wxy + torch.diag(self.px.squeeze()) #
        rhs = temp@(self.y - self.by) + torch.diag(self.px.squeeze())@(self.pred_x)
        self.x = torch.linalg.solve(lhs, rhs)

        self.pred_y = self.Wxy@self.g(self.x) + self.by
        self.error_y = self.y - self.pred_y
        self.pw_error_y = self.error_y * self.py

        self.error_x = self.x - self.pred_x
        self.pw_error_x = self.error_x * self.px
 
        # self.top_down = -self.pw_error_x
        # self.bottom_up = self.g_deriv(self.x)*(self.Wxy.T@self.pw_error_y)
        # self.delta_x = self.top_down + self.bottom_up
        # print(torch.linalg.norm(self.delta_x,dim=0))

    def step(self, t):

        # Observation Prediction
        self.pred_y = self.Wxy@self.g(self.x) + self.by
        
        # Observation Prediction Error
        self.error_y = self.y - self.pred_y

        # Precision Weighted Observation Predition Error 
        if self.error_units:
            self.pw_error_y = self.pw_error_y + self.delta_t_x * (
                self.error_y - (self.pw_error_y * self.cy.unsqueeze(dim=1))
            )
        else:
            self.pw_error_y = self.error_y * self.py

        if t == 0:
            # State Prediction
            self.pred_x = 0.0
            for k in range(self.K):
                self.pred_x = self.pred_x + self.Wyx[k]@self.h(self.prev_y[k])
            self.pred_x = self.pred_x + self.Wxx@self.f(self.prev_x) + self.bx


        # State Prediction Error
        self.error_x = self.x - self.pred_x

        # Precision Weighted State Prediction Error
        if self.error_units:
            self.pw_error_x = self.pw_error_x + self.delta_t_x * (
                self.error_x - (self.pw_error_x * self.cx.unsqueeze(dim=1))
            )
        else:
            self.pw_error_x = self.error_x * self.px

        # Calculate the gradient
        self.top_down = -self.pw_error_x
        self.bottom_up = self.g_deriv(self.x)*(self.Wxy.T@self.pw_error_y)

        # Update the State
        self.delta_x = self.top_down + self.bottom_up
        self.x = self.x + self.delta_t_x * self.delta_x

    
    def update_weights(self):
        
        # Helper function for weight matrix updates

        def delta_W(code, num=0):
            if code == "xx":
                arg1 = self.pw_error_x
                arg2 = self.f(self.prev_x)
            elif code == "xy":
                arg1 = self.pw_error_y
                arg2 = self.g(self.x)
            elif code == "yx":
                arg1 = self.pw_error_x
                arg2 = self.h(self.prev_y[num])

            batch_result = torch.einsum(
                "bnm,b->bnm",
                torch.einsum("nb,mb->bnm", arg1, arg2),
                self.mask
            )

            result = torch.sum(
                batch_result,
                axis=0
            ) / total

            return result, batch_result

        # Helper function for bias vector updates
        def delta_b(code):
            if code == "x":
                arg = self.pw_error_x
            elif code == "y":
                arg = self.pw_error_y

            batch_result = torch.einsum(
                "nb,b->bn", 
                arg, 
                self.mask
            )

            result = torch.unsqueeze(torch.sum(
                batch_result,
                axis=0
            ), dim=-1) / total

            return result, batch_result

        # for averaging across the batch 
        total = torch.sum(self.mask)


        ## State Prediction Parameters ##
        # Transition Parameters
        self.delta_Wxx, self.batched_delta_Wxx = delta_W(code="xx")
        self.Wxx = self.Wxx + self.delta_t_w * self.delta_Wxx

        self.delta_Wyx = [0]*self.K
        self.batched_delta_Wyx = [0]*self.K
        
        # need to ensure that state associated with weight is not identically zero, if function does not map zero to zero
        for k in range(self.K):
            self.delta_Wyx[k], self.batched_delta_Wyx[k] = delta_W(code="yx", num=k)
            self.Wyx[k] = self.Wyx[k] + self.delta_t_w * self.delta_Wyx[k]


        # Bias
        self.delta_bx, self.batched_bx = delta_b(code="x")
        self.bx = self.bx + self.delta_t_w * self.delta_bx


        ## Observation Prediction Parameters ##
        self.delta_Wxy, self.batched_delta_Wxy = delta_W(code="xy")
        self.Wxy = self.Wxy + self.delta_t_w * self.delta_Wxy

        # Bias
        self.delta_by, self.batched_by = delta_b(code="y")
        self.by = self.by + self.delta_t_w * self.delta_by

    def compute_energy(self, comp_metrics=False):
        # for averaging across the batch 
        total = torch.sum(self.mask)

        # error_y and pw_error_y are nb
        # error_x and pw_error_x are mb

        energy = 0.5 * (
            torch.sum(self.error_y*self.pw_error_y, dim=0) * self.mask +
            torch.sum(self.error_x*self.pw_error_x, dim=0) * self.mask
        )
        if comp_metrics:
            energy += 0.5 * (
                torch.logdet(torch.diag(self.cx)) +  # the matrices are diagonal
                torch.logdet(torch.diag(self.cy)) + 
                self.x_size*torch.log(torch.Tensor([2*torch.pi]).to(self.device))  + 
                self.y_size*torch.log(torch.Tensor([2*torch.pi]).to(self.device)) 
            )

        return energy

    def compute_cosine_distance(self, batched_result=False):
        if batched_result:
            obs_cos_dist = 1 - cosine_similarity(self.Wxy@self.g(self.pred_x) + self.by, self.y, dim=0)
            obs_cos_dist = obs_cos_dist * self.mask
        return obs_cos_dist


    def reset(self, reset_state=False, reset_error=False):
        if reset_state:
            self.x = torch.randn(self.x_size, self.batch_size).to(self.device) * 0.03
        if reset_error:
            self.pw_error_x = torch.zeros((self.x_size, self.batch_size)).to(self.device)
            self.pw_error_y = torch.zeros((self.y_size, self.batch_size)).to(self.device)

    def update_prev(self):
        for k in reversed(range(1,self.K)):
            self.prev_y[k] = torch.clone(self.prev_y[k-1])
        if self.K > 0:
            self.prev_y[0] = torch.clone(self.y)
        self.prev_x = torch.clone(self.x)

    def set_random_prev(self):
        self.prev_y = [0]*self.K
        for k in range(self.K):
            self.prev_y[k] = torch.zeros((self.y_size, self.batch_size)).to(self.device)
        self.prev_x = torch.randn(self.x_size, self.batch_size).to(self.device) * 0.03

    def predict(self, x=True, y=True):
        # State Prediction
        if x:
            self.pred_x = 0.0
            for k in range(self.K):
                self.pred_x = self.pred_x + self.Wyx[k]@self.h(self.prev_y[k])
            self.pred_x = self.pred_x + self.Wxx@self.f(self.prev_x) + self.bx
        # Observation Prediction
        if y:
            self.pred_y = self.Wxy@self.g(self.pred_x) + self.by

    def save_parameters(self, epoch, energy, savedir):
        model_parameters = {
            "y_size": self.y_size,
            "x_size": self.x_size,
            "batch_size": self.batch_size,
            "Wxx": self.Wxx,
            "Wxy": self.Wxy,
            "bx": self.bx,
            "by": self.by,
            "cy": self.cy,
            "cx": self.cx,
            "delta_t_x": self.delta_t_x,
            "delta_t_w": self.delta_t_w,
            "inf_iters": self.inf_iters,
            "energy": energy,
            "epoch": epoch,
            "K": self.K,
            "f_type": self.f_type,
            "g_type": self.g_type,
            "h_type": self.h_type,
        }
        for k in range(self.K):
            model_parameters[f"Wyx{k}"] = self.Wyx[k]

        Path(savedir).mkdir(parents=True, exist_ok=True)
        torch.save(model_parameters, f"{savedir}/epoch_{epoch}.pt")

    def load_parameters(self, path):
        parameters = torch.load(path)
        self.y_size = parameters["y_size"]
        self.x_size = parameters["x_size"]
        self.batch_size = parameters["batch_size"]
        self.Wxx = parameters["Wxx"].to(self.device)
        self.Wxy = parameters["Wxy"].to(self.device)
        self.bx = parameters["bx"].to(self.device)
        self.by = parameters["by"].to(self.device)
        self.cy = parameters["cy"].to(self.device)
        self.cx = parameters["cx"].to(self.device)
        self.py = (1.0/self.cy.unsqueeze(dim=1))
        self.px = (1.0/self.cx.unsqueeze(dim=1))
        self.delta_t_x = parameters["delta_t_x"]
        self.delta_t_w = parameters["delta_t_w"]
        self.inf_iters = parameters["inf_iters"]
        self.epoch = parameters["epoch"]
        self.K = parameters["K"]
        self.f_type = parameters["f_type"]
        self.g_type = parameters["g_type"]
        self.h_type = parameters["h_type"]
        self.Wyx = [0]*self.K
        for k in range(self.K):
            self.Wyx[k] = parameters[f"Wyx{k}"].to(self.device)

        to_print = [
            f"Model Energy: {parameters['energy']:.3f}",
            f"Last Train Epoch: {self.epoch}"
        ]

        print("\n".join(to_print))

    
    def eval_mode(self, args):
        self.batch_size = args.batch_size
        self.delta_t_w = 0.0
        self.inf_iters = args.max_inf_iters
        self.error_units = args.error_units
        self.start_at_prediction = args.start_at_prediction
        self.delta_t_x = args.delta_t_x
        self.threshold = args.threshold