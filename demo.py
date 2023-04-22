import argparse
from main import main_run

#--- Parse arguments
parser = argparse.ArgumentParser('ODE_Residual_Dynamics_Model')
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
parser.add_argument('--check_eval_model', action='store_true')
parser.add_argument('--final', action='store_false')

args = parser.parse_args()

if __name__ == '__main__':
    main_run(args)