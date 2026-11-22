from data.loader import data_name2data
from model.patch_model.multi_scale_patches import multi_scale_patches_model
from model.evaluate import evaluate, get_final_cost
import torch
from utils import seed_all
import ast
import argparse
arg = argparse.ArgumentParser()


def str2bool(s):
    if s in ["1", "true", "True"]:
        return True
    else:
        return False


def parse_list(value):
    return ast.literal_eval(value)


# Parameters Configuration
arg.add_argument("--device", type=str, required=False, default="cuda:0")
# Replace the section with the file path where you will store your data.
arg.add_argument("--data_path", type=str, required=False, default="/home/software/tsad_data")
arg.add_argument("--data_name", type=str, required=False, default="MSL")
arg.add_argument("--seed", type=int, required=False, default=42)
arg.add_argument("--validation_size", type=float, required=False, default=0.2) 
arg.add_argument("--kernel_dim", "-k_dim", type=int, required=False, default=5)
arg.add_argument("--sigma", type=float, required=False, default=0.6)
arg.add_argument("--n_feature_dim", "-d1", type=int, required=False, default=8)
arg.add_argument("--n_feature_group", "-n1", type=int, required=False, default=1)
arg.add_argument("--n_enhancement_dim", "-d2", type=int, required=False, default=9)
arg.add_argument("--n_enhancement_group", "-n2", type=int, required=False, default=1)
arg.add_argument("--n_feature_cascade_layer", "-x1", type=int, required=False, default=2)
arg.add_argument("--n_enhancement_cascade_layer", "-x2", type=int, required=False, default=5)
arg.add_argument("--enhancement_s", type=float, required=False, default=0.9)
arg.add_argument("--output_reg", type=float, required=False, default=2 ** -3)
arg.add_argument("--use_positional", type=str2bool, required=False, default=False)
arg.add_argument("--max_len", type=int, required=False, default=50000)
arg.add_argument("--patches_list", type=parse_list, required=False, default=[80, 92])
arg.add_argument("--pre_trained_weight", type=str2bool, required=False, default=0)
params = arg.parse_args().__dict__


bls_param = dict(
    kernel_dim=params["kernel_dim"],
    sigma=params["sigma"],
    n_feature_dim=params["n_feature_dim"],
    n_feature_group=params["n_feature_group"],
    n_enhancement_dim=params["n_enhancement_dim"],
    n_enhancement_group=params["n_enhancement_group"],
    n_feature_cascade_layer=params["n_feature_cascade_layer"],
    n_enhancement_cascade_layer=params["n_enhancement_cascade_layer"],
    enhancement_s=params["enhancement_s"],
    output_reg=params["output_reg"],
    use_positional=params["use_positional"],
    max_len=params["max_len"],
    device=params["device"]
)


if __name__ == "__main__":
    
    # Set the global default data type as single-precision floating-point.
    device = torch.device(params["device"])
    if device != "cpu":
        torch.set_default_dtype(torch.float32)

    # To guarantee the reproducibility of the experiments, it's advisable to set a fixed random seed.
    seed_all(params["seed"])

    # Data Preparation
    train_x, test_x, test_y = data_name2data(params["data_path"], params["data_name"])
    train_x = torch.from_numpy(train_x)

    test_x = torch.from_numpy(test_x)
    
    if device != "cpu":
        train_x = train_x.to(device).float()
        test_x = test_x.to(device).float()
    else:
        train_x = train_x.float()
        test_x = test_x.float()
    
    test_y = test_y.reshape(-1)

    data_name = params["data_name"]

    # Specify multiscale patches for the final ensemble model.
    patches_list = params["patches_list"]

    # Confirm whether to use the model weights trained in the previous round.
    pre_trained_weight = params["pre_trained_weight"]

    # Set the proportion of the validation set.
    validation_size = params["validation_size"]

    p, t_tr_sum, t_te_sum, TR_LOSS, VA_LOSS = multi_scale_patches_model(patches_list, train_x, test_x, pre_trained_weight, validation_size, data_name, **bls_param)
    
    if not pre_trained_weight:
        cost = get_final_cost(TR_LOSS, VA_LOSS, validation_size)
        print(f'\nThe value of the comprehensive cost function is: {cost:.10f}')
        print("\nThe final ensemble DPatchBLS's total training duration is: %f s" % t_tr_sum)
        print("The final ensemble DPatchBLS's total testing duration is: %f s" % t_te_sum)
    else:
        print("\nThe final ensemble DPatchBLS's total testing duration is: %f s" % t_te_sum)

    evaluate(p, test_y)
