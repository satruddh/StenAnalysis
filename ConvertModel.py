import torch
from UnetBN import UnetBN
from Enet import ENet


MODEL1_PATH = "./UnetBN_best_model_weights-dice.pth"
MODEL2_PATH = "./student_enet2_KD.pth"


SCRIPTED_MODEL1_PATH = "model1_scripted.pt"
SCRIPTED_MODEL2_PATH = "model2_scripted.pt"

def load_model(model_path,model=None):
    
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device('cpu'),
            weights_only=True 
        )
    )
    model.eval()
    return model

def convert_to_torchscript(model, save_path,dummy_input=torch.randn(1,3,256,256)):
    
    
    scripted_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(scripted_model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    
    unetbn = UnetBN(in_channels=3,out_channels=1)
    model1 = load_model(MODEL1_PATH, unetbn)
    input1 = torch.randn(1,3,256,256)
    convert_to_torchscript(model1, SCRIPTED_MODEL1_PATH,dummy_input=input1)

    
    enet= ENet(num_classes=1)
    model2 = load_model(MODEL2_PATH, enet)
    convert_to_torchscript(model2, SCRIPTED_MODEL2_PATH,dummy_input=input1)
