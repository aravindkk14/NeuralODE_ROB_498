from panda_pushing_env import PandaPushingEnv
from visualizers import GIFVisualizer
from loading_train_data import process_data_single_step, process_data_multiple_step
from loading_train_data import SingleStepDynamicsDataset, MultiStepDynamicsDataset

from learning_state_dynamics import ResidualDynamicsModel
from learning_state_dynamics import MultiStepLoss, SE2PoseLoss, SingleStepLoss
from learning_state_dynamics import train_model
from learning_state_dynamics import PushingController, free_pushing_cost_function, obstacle_avoidance_pushing_cost_function


from learning_ode_dynamics import ODEDynamicsModel
from learning_ode_dynamics import MultiStepLoss_ODE, SingleStepLoss_ODE, SE2PoseLoss_ODE
from learning_ode_dynamics import train_model_ode
from learning_ode_dynamics import PushingController_ODE, free_pushing_cost_function, obstacle_avoidance_pushing_cost_function
from learning_ode_dynamics import makedirs as direct_file


from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE


import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from tqdm.notebook import tqdm
import datetime
import itertools

current_time = datetime.datetime.now()
fil = str(current_time.date()) + " " + str(current_time.hour) + "_" + str(current_time.minute)


#--- Parse arguments
parser = argparse.ArgumentParser('ODE/Ressidual Dynamics Model')
parser.add_argument('--odeint_method', type=str, choices=['dopri5', 'dopri8'], default='dopri5')
parser.add_argument('--load_train_eval', type=str, choices=['load', 'train', 'eval'], default='eval')
parser.add_argument('--num_steps', type=int, default=4)
parser.add_argument('--model', type=str, choices=['ode', 'residual'], default='ode')
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_t_steps', type=int, default=4)
parser.add_argument('--unsave_model', action='store_false')
parser.add_argument('--stop_controller', action='store_true')

args = parser.parse_args()

def test_case1():
    # Create the GIF visualizer
    visualizer = GIFVisualizer()

    # Initialize the simulation environment. This will only render push motions, omitting the robot reseting motions.
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False, camera_heigh=500, camera_width=500, render_every_n_steps=5)
    env.reset()

    # Perform a sequence of 3 random actions:
    for i in tqdm(range(10)):
        action_i = env.action_space.sample()
        state, reward, done, info = env.step(action_i)
        
        if done:
            break

    # Create and visualize the gif file.
    Image(filename=visualizer.get_gif())

def no_obstacle_res(model):
    visualizer = GIFVisualizer()

    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController(env, model, free_pushing_cost_function, num_samples=100, horizon=10)
    env.reset()

    state_0 = env.reset()
    state = state_0

    # num_steps_max = 100
    num_steps_max = 20
    pbar = tqdm(range(num_steps_max))

    for i in pbar:
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        pbar.set_description(f'Iter: {i:d}')
        if done:
            break
            
    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_FREE
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')
            
    # Create and visualize the gif file.
    path = os.path.join('results',fil)
    direct_file(path)
    Image(filename=visualizer.get_gif(os.path.join(path,'no_obstacle_res.gif')))

def obstacle_res(model):
    visualizer = GIFVisualizer()

    # set up controller and environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController(env, model,
                                obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
    env.reset()

    state_0 = env.reset()
    state = state_0

    num_steps_max = 20
    pbar = tqdm(range(num_steps_max))

    for i in pbar:
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        pbar.set_description(f'Iter: {i:d}')
        if done:
            break
            
    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')

    # Create and visualize the gif file.
    path = os.path.join('results',fil)
    direct_file(path)
    Image(filename=visualizer.get_gif(os.path.join(path,'obstacle_res.gif')))


def no_obstacle_ode(model):
    visualizer = GIFVisualizer()

    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController_ODE(env, model, free_pushing_cost_function, num_samples=100, horizon=10)
    env.reset()

    state_0 = env.reset()
    state = state_0

    # num_steps_max = 100
    num_steps_max = 20
    pbar = tqdm(range(num_steps_max))

    for i in pbar:
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        pbar.set_description(f'Iter: {i:d}')
        if done:
            break
        
    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_FREE
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')
            
    # Create and visualize the gif file.
    path = os.path.join('results',fil)
    direct_file(path)
    Image(filename=visualizer.get_gif(os.path.join(path,'no_obstacle_ode.gif')))

def obstacle_ode(model):
    visualizer = GIFVisualizer()

    # set up controller and environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
    controller = PushingController_ODE(env, model,
                                obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
    env.reset()

    state_0 = env.reset()
    state = state_0

    num_steps_max = 20
    pbar = tqdm(range(num_steps_max))

    for i in pbar:
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        pbar.set_description(f'Iter: {i:d}')
        if done:
            break
            
    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')

    # Create and visualize the gif file.
    path = os.path.join('results',fil)
    direct_file(path)
    Image(filename=visualizer.get_gif(os.path.join(path,'obstacle_ode.gif')))


def multiloader_test(train_loader):
    # let's check your dataloader

    # you should return a dataloader
    print('Is the returned train_loader a DataLoader?')
    print('Yes' if isinstance(train_loader, torch.utils.data.DataLoader) else 'No')
    print('')

    # You should have used random split to split your data - 
    # this means the validation and training sets are both subsets of an original dataset
    print('Was random_split used to split the data?')
    print('Yes' if isinstance(train_loader.dataset, torch.utils.data.Subset) else 'No')
    print('')

    # The original dataset should be of a MultiStepDynamicsDataset
    print('Is the dataset a MultiStepDynamicsDataset?')
    print('Yes' if isinstance(train_loader.dataset.dataset, MultiStepDynamicsDataset) else 'No')
    print('')

    # we should see the state is shape (batch_size, 3)
    # and action, next_state are shape (batch_size, num_steps, 3)
    for item in train_loader:
        print(f'state is shape {item["state"].shape}')
        print(f'action is shape {item["action"].shape}')
        print(f'next_state is shape {item["next_state"].shape}')
        break


def singleloader_test(train_loader):
    # let's check your dataloader

    # you should return a dataloader
    print('Is the returned train_loader a DataLoader?')
    print('Yes' if isinstance(train_loader, torch.utils.data.DataLoader) else 'No')
    print('')

    # You should have used random split to split your data - 
    # this means the validation and training sets are both subsets of an original dataset
    print('Was random_split used to split the data?')
    print('Yes' if isinstance(train_loader.dataset, torch.utils.data.Subset) else 'No')
    print('')

    # The original dataset should be of a MultiStepDynamicsDataset
    print('Is the dataset a SingleStepDynamicsDataset?')
    print('Yes' if isinstance(train_loader.dataset.dataset, SingleStepDynamicsDataset) else 'No')
    print('')

    # we should see the state is shape (batch_size, 3)
    # and action, next_state are shape (batch_size, num_steps, 3)
    for item in train_loader:
        print(f'state is shape {item["state"].shape}')
        print(f'action is shape {item["action"].shape}')
        print(f'next_state is shape {item["next_state"].shape}')
        break



def plot_loss(train_losses, val_losses, name = 'loss.png', eval = False):
    # plot train loss and test loss:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    axes[0].plot(train_losses)
    axes[0].grid()
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_yscale('log')
    axes[1].plot(val_losses)
    axes[1].grid()
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_yscale('log')

    if eval:
        path = os.path.join('evals',fil)
        direct_file(path)
        plt.savefig(os.path.join(path,name))
    else:
        path = os.path.join('results',fil)
        direct_file(path)
        plt.savefig(os.path.join(path,name))
        plt.show()
    

if __name__ == "__main__":   


    if args.load_train_eval == 'train':
        #--- Collect data
        collected_data = np.load('collected_data.npy', allow_pickle=True)

        if args.model == 'residual':
            if args.num_steps == 1:
                #--- Train data for Single step residual dynamics model
                pushing_residual_dynamics_model = ResidualDynamicsModel(3,3)
                train_loader, val_loader = process_data_single_step(collected_data, batch_size=args.batch_size)
                pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
                pose_loss = SingleStepLoss(pose_loss)

                lr = args.learning_rate
                num_epochs = args.num_epoch
                train_losses = None
                val_losses = None

                train_losses, val_losses = train_model(pushing_residual_dynamics_model,pose_loss,train_loader, val_loader, num_epochs,lr)
                plot_loss(train_losses, val_losses)

                if args.unsave_model:
                    path = os.path.join('models',fil)
                    direct_file(path)
                    torch.save(pushing_residual_dynamics_model.state_dict(), os.path.join(path,'pushing_residual_dynamics_model.pt'))
            else:
                #--- Train data for Multi step residual dynamics model
                pushing_residual_dynamics_model = ResidualDynamicsModel(3,3)
                train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=args.batch_size,num_steps=args.num_steps)
                pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
                pose_loss = MultiStepLoss(pose_loss, discount=0.9)

                lr = args.learning_rate
                num_epochs = args.num_epoch
                train_losses = None
                val_losses = None

                train_losses, val_losses = train_model(pushing_residual_dynamics_model,pose_loss,train_loader, val_loader, num_epochs,lr)
                plot_loss(train_losses, val_losses)

                if args.unsave_model:
                    path = os.path.join('models',fil)
                    direct_file(path)
                    torch.save(pushing_residual_dynamics_model.state_dict(), os.path.join(path,'pushing_multi_step_residual_dynamics_model.pt'))

        elif args.model == 'ode':
            if args.num_steps == 1:
                pushing_ode_model = ODEDynamicsModel(3,3,method=args.odeint_method)
                train_loader, val_loader = process_data_single_step(collected_data, batch_size=args.batch_size)
                pose_loss = SE2PoseLoss_ODE(block_width=0.1, block_length=0.1)
                pose_loss = SingleStepLoss_ODE(pose_loss,num_t_steps=args.num_t_steps,method=args.odeint_method)

                lr = args.learning_rate
                num_epochs = args.num_epoch
                train_losses = None
                val_losses = None

                train_losses, val_losses = train_model_ode(pushing_ode_model,pose_loss,train_loader, val_loader, num_epochs,lr)
                plot_loss(train_losses, val_losses)

                if args.unsave_model:
                    path = os.path.join('models',fil)
                    direct_file(path)
                    torch.save(pushing_ode_model.state_dict(), os.path.join(path,'pushing_single_ode_dynamics_model.pt'))

            else:
                pushing_ode_model = ODEDynamicsModel(3,3,method=args.odeint_method)
                train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=args.batch_size ,num_steps=args.num_steps)

                pose_loss = SE2PoseLoss_ODE(block_width=0.1, block_length=0.1)
                pose_loss = MultiStepLoss_ODE(pose_loss, discount=0.9,num_t_steps=args.num_t_steps,method=args.odeint_method)

                lr = args.learning_rate
                num_epochs = args.num_epoch
                train_losses = None
                val_losses = None

                train_losses, val_losses = train_model_ode(pushing_ode_model,pose_loss,train_loader, val_loader, num_epochs,lr)
                plot_loss(train_losses, val_losses)

                if args.unsave_model:
                    path = os.path.join('models',fil)
                    direct_file(path)
                    torch.save(pushing_ode_model.state_dict(), os.path.join(path,'pushing_multi_ode_dynamics_model.pt'))
        

    elif args.load_train_eval == 'load':
        if args.model == 'residual':
            if args.num_steps == 1:
                #--- Load data for Single step residual dynamics model
                pushing_residual_dynamics_model = ResidualDynamicsModel(3,3)
                pushing_residual_dynamics_model.load_state_dict(torch.load('models/pushing_residual_dynamics_model.pt'))
            else:
                #--- Load data for Multi step residual dynamics model
                pushing_residual_dynamics_model = ResidualDynamicsModel(3,3)
                pushing_residual_dynamics_model.load_state_dict(torch.load('models/pushing_multi_step_residual_dynamics_model.pt'))

        elif args.model == 'ode':
            if args.num_steps == 1:
                #--- Load data for Single step ODE dynamics model
                pushing_ode_model = ODEDynamicsModel(3,3)
                pushing_ode_model.load_state_dict(torch.load('models/pushing_ode_dynamics_model.pt'))
            else:
                #--- Load data for Multi step ODE dynamics model
                pushing_ode_model = ODEDynamicsModel(3,3)
                pushing_ode_model.load_state_dict(torch.load('models/pushing_multi_ode_dynamics_model.pt'))
    
    
    elif args.load_train_eval == 'eval':
        collected_data = np.load('collected_data.npy', allow_pickle=True)

        num_layers = [3,4]
        hidden_size = [80,100,120]
        odeint_methods = ['euler','explicit_adams','dopri5', 'dopri8']
        num_t_steps = [4,6,8]

        for om, hs, ts, nl in itertools.product(odeint_methods,hidden_size,num_t_steps,num_layers):
            print('num_layers: ',nl,'hidden_size: ',hs,'odeint_method: ',om,'num_t_steps: ',ts)
            pushing_ode_model = ODEDynamicsModel(3,3,num_layers=nl,hidden_dim=hs,method=om)
            train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=args.batch_size ,num_steps=args.num_steps)

            pose_loss = SE2PoseLoss_ODE(block_width=0.1, block_length=0.1)
            pose_loss = MultiStepLoss_ODE(pose_loss, discount=0.9,num_t_steps=ts,method=om)

            lr = args.learning_rate
            num_epochs = args.num_epoch
            train_losses = None
            val_losses = None

            train_losses, val_losses = train_model_ode(pushing_ode_model,pose_loss,train_loader, val_loader, num_epochs,lr)
            name = 'ode_'+str(nl)+'_'+str(hs)+'_'+str(om)+'_'+str(ts)+'.png'
            plot_loss(train_losses, val_losses, name, True)

            path = os.path.join('evals',fil,'losses')
            direct_file(path)
            arr = np.array([train_losses,val_losses])
            loss_name = 'ode_'+str(nl)+'_'+str(hs)+'_'+str(om)+'_'+str(ts)+'_losses.npy'
            np.save(os.path.join(path,loss_name),arr)            

            path = os.path.join('evals',fil)
            direct_file(path)
            model_name = 'ode_'+str(nl)+'_'+str(hs)+'_'+str(om)+'_'+str(ts)+'_model.pt'
            torch.save(pushing_ode_model.state_dict(), os.path.join(path,model_name))


    if args.stop_controller == False:
        if args.model == 'residual':
                no_obstacle_res(pushing_residual_dynamics_model)
                obstacle_res(pushing_residual_dynamics_model)

        elif args.model == 'ode':
                #--- Trajectory planning for single step ODE dynamics model
                no_obstacle_ode(pushing_ode_model)
                obstacle_ode(pushing_ode_model)
    