import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class SE2PoseLoss_ODE(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        # print(pose_pred.shape, pose_target.shape)
        loss_fn = nn.MSELoss()
        rg = ((self.l**2 + self.w**2)/12)**0.5
        se2_pose_loss = loss_fn(pose_target[:,0],pose_pred[:,0]) + loss_fn(pose_target[:,1],pose_pred[:,1])
        se2_pose_loss += rg*loss_fn(pose_target[:,2],pose_pred[:,2])
        # ---
        return se2_pose_loss

class SingleStepLoss_ODE(nn.Module):

    def __init__(self, loss_fn, num_t_steps=4, method='dopri5'):
        super().__init__()
        self.loss = loss_fn
        self.num_steps = num_t_steps
        self.method = method

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) 
        and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        state_action = torch.cat([state,action],dim = -1)
        t = torch.linspace(0,1,self.num_steps)
        next_state = odeint(model,state_action,t,method=self.method).permute(1,0,2)[:,1,:3]
        single_step_loss = self.loss(next_state.unsqueeze(1), target_state.unsqueeze(1))
        # ---
        return single_step_loss


class MultiStepLoss_ODE(nn.Module):

    def __init__(self, loss_fn, discount=0.99, num_t_steps=4, method='dopri5'):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount
        self.num_steps = num_t_steps
        self.method = method

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        multi_step_loss_arr = torch.zeros(actions.shape[1])
        # multi_step_loss = 0
        curr_state = state
        for i in range(actions.shape[1]):
          state_action = torch.cat([curr_state,actions[:,i]],dim = -1)

          t = torch.linspace(0,1,self.num_steps)
          
          next_state = odeint(model,state_action,t,method=self.method)
          next_state = next_state.permute(1,0,2)[:,:,:3]
         
          multi_step_loss_arr[i] = (self.discount**i) * self.loss(next_state[:,-1], target_states[:,i])
          curr_state = next_state[:,-1]

        multi_step_loss = multi_step_loss_arr.sum()
        # ---
        return multi_step_loss



class ODEDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim,num_layers=3,hidden_dim=100, method='dopri5'):
        super(ODEDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.method = method
        
        self. layer_list = []
        self.layer_list.append(nn.Linear(state_dim+action_dim, hidden_dim))
        self.layer_list.append(nn.ReLU())

        for _ in range(num_layers-2):
            self.layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer_list.append(nn.ReLU())
        
        self.layer_list.append(nn.Linear(hidden_dim, state_dim+action_dim))
        self.net = nn.Sequential(*self.layer_list)

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self, t, state_action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        next_state = self.net(state_action)
        return next_state



def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    Q = torch.tensor([[1,0,0],[0,1,0],[0,0,0.1]] ,dtype = state.dtype, device = state.device)

    cal = (state - target_pose.view(1,-1)) @ Q @ (state - target_pose.view(1,-1)).T
    cost = torch.diag(cal)     
    # ---
    return cost


# def collision_detection(state):
#     """
#     Checks if the state is in collision with the obstacle.
#     The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
#     :param state: torch tensor of shape (B, state_size)
#     :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
#     """
#     obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
#     obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
#     box_size = BOX_SIZE  # scalar for parameter w
#     in_collision = None
#     # --- Your code here
#     num_states = state.shape[0]
#     in_collision = torch.empty((num_states))
#     max_dist = torch.sqrt(2*(torch.tensor(box_size)**2))/2 + torch.sqrt(torch.sum(obstacle_dims**2))/2
#     min_dist = (torch.tensor(box_size)+ torch.min(obstacle_dims))/2
      
#     for i in range(num_states):
#       dist = (obstacle_centre - state[i,:2])**2
#       dist = torch.sqrt(torch.sum(dist))
#       # print(dist, max_dist, min_dist)
#       if dist > max_dist:
#         in_collision[i] = 0
#       elif dist <= min_dist:
#         in_collision[i] = 1
#       else:
#         half_side = box_size / 2
#         c = torch.cos(state[i,2])
#         s = torch.sin(state[i,2])
#         R = torch.tensor([[c, -s, state[i,0]], [s, c, state[i,1]], [0, 0, 1]])
#         block_vertices = torch.tensor([[half_side, half_side, 1], [-half_side, half_side, 1], [-half_side, -half_side, 1], [half_side, -half_side, 1]])
#         block_vertices = torch.matmul(R, block_vertices.T).transpose(0, 1)[:, :2]
        
#         min_dim = obstacle_centre - obstacle_dims/2
#         max_dim = obstacle_centre + obstacle_dims/2

#         result = torch.concat(((block_vertices>=min_dim).float(),(block_vertices<=max_dim).float()),dim = 1)
#         corner = torch.sum(result, dim=1)
#         corner_in_obstacle = 4.0
#         if corner_in_obstacle in corner:
#           in_collision[i] = 1
#         else:
#           in_collision[i] = 0
        
#     # ---
#     return in_collision

# def collision_detection(state):
#     """
#     Checks if the state is in collision with the obstacle.
#     The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
#     :param state: torch tensor of shape (B, state_size)
#     :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
#     """
#     obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
#     obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
#     box_size = BOX_SIZE  # scalar for parameter w
#     in_collision = None
#     # --- Your code here
#     num_states = state.shape[0]
#     in_collision = torch.empty((num_states))
#     max_dist = torch.sqrt(2*(torch.tensor(box_size)**2))/2 + torch.sqrt(torch.sum(obstacle_dims**2))/2
#     min_dist = (torch.tensor(box_size)+ torch.min(obstacle_dims))/2
      
#     for i in range(num_states):
#         dist = (obstacle_centre - state[i,:2])**2
#         dist = torch.sqrt(torch.sum(dist))
#         if dist > max_dist:
#             in_collision[i] = 0
#         elif dist <= min_dist:
#             in_collision[i] = 1
#         else:
#             half_side = box_size / 2
#             c = torch.cos(state[i,2])
#             s = torch.sin(state[i,2])
#             R = torch.tensor([[c, -s, state[i,0]], [s, c, state[i,1]], [0, 0, 1]])
#             block_vertices = torch.tensor([[half_side, half_side, 1], [-half_side, half_side, 1], [-half_side, -half_side, 1], [half_side, -half_side, 1]])
#             block_vertices = torch.matmul(R, block_vertices.T).transpose(0, 1)[:, :2]

#             min_dim = obstacle_centre - obstacle_dims/2
#             max_dim = obstacle_centre + obstacle_dims/2

#             collision_check = ((block_vertices[:,0] >= min_dim[0]) & (block_vertices[:,0] <= max_dim[0]) & (block_vertices[:,1] >= min_dim[1]) & (block_vertices[:,1] <= max_dim[1]))
            
#             if collision_check.sum() > 0:
#                 in_collision[i] = 1
#             else:
#                 in_collision[i] = 0
        
#     # ---
#     return in_collision



def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    # obstacle_corners = get_corner_coordinates(obstacle_centre.reshape(1, -1), obstacle_dims)
    # unrotated_object_corners = get_corner_coordinates(state[:, :2], torch.tensor([box_size, box_size]))
    # object_corners = rotate_corner_coordinates(unrotated_object_corners, state[:, -1])
    
    # in_collision = torch.full(size=(len(object_corners),), fill_value=-1.0, dtype=torch.float)
    # for i in range(len(object_corners)):
    #     in_collision[i] = is_polygon_intersecting(obstacle_corners.reshape(4, 2), object_corners[i])

    w = box_size/2
    obs_w = obstacle_dims/2
    corner = torch.tensor([[w,w],[-w,-w]])
    corner_obs = torch.stack((obs_w,-1*obs_w), dim=0)
    obs_dim = obstacle_centre + corner_obs
    in_collision = torch.zeros(state.shape[0])

    for i in range(state.shape[0]):
      obj_dim = corner + state[i,:2]
      if(obj_dim[0,0]<obs_dim[1,0] or
      obj_dim[1,0]>obs_dim[0,0] or
      obj_dim[0,1]<obs_dim[1,1] or
      obj_dim[1,1]>obs_dim[0,1]):
        in_collision[i] = 0
      else:
        in_collision[i] = 1

    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    Q = torch.tensor([[1,0,0],[0,1,0],[0,0,0.1]] ,dtype = state.dtype, device = state.device)

    # cal = (state - target_pose.view(1,-1)) @ Q @ (state - target_pose.view(1,-1)).T
    # cost = torch.diag(cal)     

    collide = collision_detection(state)
    cost = torch.sum((state - target_pose) @ Q * (state - target_pose), 1) + 100 * collide
    # ---
    return cost


class PushingController_ODE(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.1 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.001
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        # print(self.model.type)
        state_action = torch.cat((state, action), dim=-1)
        t = torch.linspace(0, 1, 10)
        next_state = odeint(self.model,state_action,t,method=self.model.method)[-1,:,:3]
        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state)
        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.detach().numpy()
        # ---
        return action
    
# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
def train_step_ode(model, train_loader, optimizer, loss_fn) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0. # TODO: Modify the value
    # Initialize the train loop
    # --- Your code here
    model.train()
    # ---
    for item in train_loader:
        loss = loss_fn.forward(model, item["state"], item["action"], item["next_state"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ---
        train_loss += loss.item()
    return train_loss/len(train_loader)


def val_step_ode(model, val_loader, loss_fn) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    # --- Your code here
    model.eval()
    # ---
    for item in val_loader:
        loss = loss_fn.forward(model, item["state"], item["action"], item["next_state"])
        # ---
        val_loss += loss.item()
    return val_loss/len(val_loader)


def train_model_ode(model, loss_fn, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = None
    # Initialize the optimizer
    # --- Your code here
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
  
    # ---
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = None
        val_loss_i = None
        # --- Your code here
        train_loss_i = train_step_ode(model,train_dataloader,optimizer, loss_fn)
        val_loss_i = val_step_ode(model,val_dataloader, loss_fn)
        
        # ---
        pbar.set_description(f'Epoch: {epoch_i} | Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def visualize(true_y, pred_y, odefunc, itr, t):
        makedirs('png')
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        plt.show(block=False)

        
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)