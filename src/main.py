import os
from datetime import datetime
import torch
from train_target import train_model, load_model, evaluate_mia
from data_loader import load_cifar10, load_svhn

def select_from_options(prompt, options):
    options_str = '/'.join(options)
    while True:
        user_input = input(f"{prompt} ({options_str}): ").strip().lower()
        if user_input in options:
            return user_input
        print(f"Please enter one of: {options_str}")

if __name__ == "__main__":
    dataset = select_from_options("Which dataset do you want to use?", ['cifar10', 'svhn'])

    while True:
        try:
            epochs = int(input("How many epochs do you want to train for? (20/50/100): ").strip())
            if epochs in [20, 50, 100]:
                break
            else:
                print("Please enter 20, 50, or 100.")
        except Exception:
            print("Please enter a valid integer (20/50/100).")

    train_or_load = select_from_options("Do you want to train a new model or load an existing model?", ['train', 'load'])

    if dataset == 'cifar10':
        trainloader, valloader, testloader = load_cifar10()
    else:
        trainloader, valloader, testloader = load_svhn()

    # Save outputs in 'outputs/' in the parent folder of src
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', f"{dataset}_epoch{epochs}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = f"{dataset}_simplecnn_epoch{epochs}.pth"

    if train_or_load == 'train':
        model, train_acc, val_acc = train_model(
            trainloader, valloader, epochs=epochs, model_path=model_path, output_dir=output_dir
        )
    else:
        if not os.path.exists(model_path):
            print(f"Trained model {model_path} does not exist. Training now...")
            model, train_acc, val_acc = train_model(
                trainloader, valloader, epochs=epochs, model_path=model_path, output_dir=output_dir
            )
        else:
            model = load_model(model_path, device)
            train_acc = val_acc = None

    evaluate_mia(
        model, trainloader, testloader, device, output_dir,
        dataset_name=dataset, epochs=epochs, train_acc=train_acc, val_acc=val_acc
    )
