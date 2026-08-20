import os
import re
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torch.cuda.amp import autocast, GradScaler
import random
import torch
from torch import nn, einsum

# Import custom network architecture and weight conversion utility
from network.S2UNet import S2UNet, convert_old_weights_to_new

# Enable cuDNN benchmark to optimize performance for fixed input sizes
torch.backends.cudnn.benchmark = True

# Global variables for network and dataset naming conventions
netsource = 'S2UNet'
datasource = 'FANVID-he'


class Config:
    """
    Configuration parameters for training, testing, and file paths.
    """
    # Hardware and training hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    lr = 1e-4
    final_lr = 8e-5
    warmup_epochs = 10
    epochs = 100
    img_size = 256
    mode = 'test'
    channels = 16

    # Training data paths
    train_hazy = "data/" + datasource + "/train/hazy"
    train_clean = "data/" + datasource + "/train/vis"
    train_nir = "data/" + datasource + "/train/nir"
    train_nir2 = "data/" + datasource + "/train/nir2"

    # Validation data paths
    val_hazy = "data/" + datasource + "/val/hazy"
    val_clean = "data/" + datasource + "/val/vis"
    val_nir = "data/" + datasource + "/val/nir"
    val_nir2 = "data/" + datasource + "/val/nir2"

    # Testing data paths
    test_hazy = "data/" + datasource + "/test/hazy"
    test_clean = "data/" + datasource + "/test/vis"
    test_nir = "data/" + datasource + "/test/nir"
    test_nir2 = "data/" + datasource + "/test/nir2"

    # Directories and files for logging and saving models/results
    save_dir = "results"
    ablation_log = "log/ablation.txt"
    model_base_dir = "trained_models"
    model_filename = netsource + ".pth"
    train_log = "log/" + netsource + ".txt"


class DehazeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multi-modal image data.
    Handles RGB images (hazy and ground truth) and two independent NIR modalities.
    """

    def __init__(self, hazy_dir, clean_dir, nir_dir, nir_dir2, is_train=True):
        self.is_train = is_train
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        self.nir_dir = nir_dir
        self.nir_dir2 = nir_dir2

        # Filter and store valid image filenames
        self.hazy_filenames = [f for f in os.listdir(hazy_dir) if f.endswith(('.png', '.tiff'))]
        self.clean_nir_filenames = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.tiff'))]

        # Transformation pipeline for 3-channel RGB images
        self.transform_rgb = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Transformation pipeline for 1-channel NIR images
        self.transform_nir = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Transformation pipeline for the second 1-channel NIR images
        self.transform_nir2 = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.hazy_filenames)

    def __getitem__(self, idx):
        # Construct paths for the target modalities based on naming conventions
        hazy_filename = self.hazy_filenames[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_filename)
        clean_path = os.path.join(self.clean_dir, hazy_filename.replace('_foggy_0.5.png', '.png'))
        nir_path = os.path.join(self.nir_dir, hazy_filename.replace('rgb_foggy_0.5.png', 'thermal_foggy_0.5.png'))
        nir_path2 = os.path.join(self.nir_dir2, hazy_filename.replace('rgb_foggy_0.5.png', 'thermal_foggy_0.5.png'))

        # Load images and convert to appropriate color spaces
        hazy = Image.open(hazy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        nir = Image.open(nir_path).convert('L')
        nir2 = Image.open(nir_path2).convert('L')

        # Apply spatial data augmentations randomly during training
        if self.is_train:
            if random.random() > 0.5:
                hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)
                clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
                nir = nir.transpose(Image.FLIP_LEFT_RIGHT)
                nir2 = nir2.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hazy = hazy.transpose(Image.FLIP_TOP_BOTTOM)
                clean = clean.transpose(Image.FLIP_TOP_BOTTOM)
                nir = nir.transpose(Image.FLIP_TOP_BOTTOM)
                nir2 = nir2.transpose(Image.FLIP_TOP_BOTTOM)

        # Return transformed tensors
        return (
            self.transform_rgb(hazy),
            self.transform_rgb(clean),
            self.transform_nir(nir),
            self.transform_nir(nir2)
        )


def validate(cfg, model):
    """
    Evaluates the model on the validation set to track generalization.
    Returns the average PSNR and SSIM across the validation dataset.
    """
    val_set = DehazeDataset(cfg.val_hazy, cfg.val_clean, cfg.val_nir, cfg.val_nir2, is_train=False)
    val_loader = DataLoader(val_set, batch_size=2 * cfg.batch_size, shuffle=False, num_workers=4)

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for hazy, clean, nir, nir2 in val_loader:
            hazy = hazy.to(cfg.device)
            clean = clean.to(cfg.device)
            nir = nir.to(cfg.device)
            nir2 = nir2.to(cfg.device)

            # Perform mixed precision inference
            with torch.amp.autocast(cfg.device):
                output = model([hazy, nir, nir2])

            # Denormalize outputs and ground truths from [-1, 1] to [0, 255]
            output = (output.clamp(-1, 1) + 1) / 2.0 * 255.0
            clean = (clean + 1) / 2.0 * 255.0

            # Convert tensors to numpy format (H, W, C) for metric calculations
            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            clean_np = clean.cpu().numpy().transpose(0, 2, 3, 1)

            batch_size = output.size(0)
            for i in range(batch_size):
                pred = output_np[i].astype(np.uint8)
                gt = clean_np[i].astype(np.uint8)

                # Accumulate evaluation metrics
                current_psnr = psnr(gt, pred, data_range=255)
                current_ssim = ssim(gt, pred, data_range=255, channel_axis=2, multichannel=True)
                total_psnr += current_psnr
                total_ssim += current_ssim

    # Calculate dataset averages
    avg_psnr = total_psnr / len(val_set)
    avg_ssim = total_ssim / len(val_set)

    # Revert model back to training mode
    model.train()
    return avg_psnr, avg_ssim


def add_text_to_image(image_np, text_to_add):
    """
    Utility function to overlay text onto a numpy image array.
    Used for appending metric values directly onto the output images.
    """
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    # Attempt to load a readable font, fallback to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Determine bounding box and dimensions for the text block
    if hasattr(draw, 'textbbox'):
        text_box = draw.textbbox((0, 0), text_to_add, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
    else:
        text_size = draw.textsize(text_to_add, font=font)
        text_width = text_size[0]
        text_height = text_size[1]

    # Calculate coordinates to place text in the bottom right corner
    margin = 10
    image_width, image_height = img.size
    x = image_width - text_width - margin
    y = image_height - text_height - margin
    position = (x, y)

    # Create a text outline for better visibility against varying backgrounds
    outline_color = "black"
    draw.text((position[0] - 1, position[1] - 1), text_to_add, font=font, fill=outline_color)
    draw.text((position[0] + 1, position[1] - 1), text_to_add, font=font, fill=outline_color)
    draw.text((position[0] - 1, position[1] + 1), text_to_add, font=font, fill=outline_color)
    draw.text((position[0] + 1, position[1] + 1), text_to_add, font=font, fill=outline_color)

    # Main text drawing logic is intentionally disabled in this configuration
    main_color = "white"
    draw.text(position, text_to_add, font=font, fill=main_color)

    return img


def train(cfg, model):
    """
    Executes the main training loop, handles learning rate scheduling,
    logging, mixed precision backpropagation, and checkpoint saving.
    """
    model_path = os.path.join(cfg.model_base_dir, cfg.model_filename)

    # Initialize dataloader, optimizer, loss function, and gradient scaler
    train_set = DehazeDataset(cfg.train_hazy, cfg.train_clean, cfg.train_nir, cfg.train_nir2)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler(cfg.device)

    best_loss = float('inf')
    best_val_psnr = 0.0
    val_frequency = 2

    for epoch in range(cfg.epochs):
        start_time = time.time()

        # Update learning rate: Linear warmup followed by linear decay
        if epoch < cfg.warmup_epochs:
            lr = (epoch + 1) / cfg.warmup_epochs * cfg.lr
        else:
            progress = (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)
            lr = cfg.lr - (cfg.lr - cfg.final_lr) * progress

        for param_group in opt.param_groups:
            param_group['lr'] = lr

        model.train()
        epoch_loss = 0.0

        for i, (hazy, clean, nir, nir2) in enumerate(train_loader):
            hazy = hazy.to(cfg.device)
            clean = clean.to(cfg.device)
            nir = nir.to(cfg.device)
            nir2 = nir2.to(cfg.device)

            opt.zero_grad()

            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.amp.autocast(cfg.device):
                pred = model([hazy, nir, nir2])
                loss = criterion(pred, clean)

            # Gradient scaling, unscaling, clipping, and optimizer step
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()

            # Periodically log batch metrics
            if i % 200 == 0:
                log_info = (f"Epoch [{epoch + 1}/{cfg.epochs}] Batch [{i}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f} LR: {lr:.2e}")
                print(log_info)
                with open(cfg.train_log, "a") as f:
                    f.write(log_info + "\n")

        avg_loss = epoch_loss / len(train_loader)

        # Calculate and format the duration of the current epoch
        epoch_time = time.time() - start_time
        minutes, seconds = divmod(epoch_time, 60)
        time_str = f"{int(minutes)}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"

        # Log epoch summary
        log_info = f"Epoch [{epoch + 1}/{cfg.epochs}] Avg Loss: {avg_loss:.4f} LR: {lr:.2e} Time: {time_str}"
        print(log_info)
        with open(cfg.train_log, "a") as f:
            f.write(log_info + "\n")

        # Increase validation frequency during late training stages
        if epoch > 80:
            val_frequency = 1

        # Validation and checkpoint saving logic
        if (epoch + 1) % val_frequency == 0:
            val_psnr, val_ssim = validate(cfg, model)
            val_log = f"----------Validation @ Epoch {epoch + 1} - PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}----------"
            print(val_log)
            with open(cfg.train_log, "a") as f:
                f.write(val_log + "\n")

            # Update latest model weights
            torch.save(model.state_dict(), model_path)

            # Conditionally save best weights based on validation PSNR
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_path_val = model_path.replace(".pth", "_val.pth")
                torch.save(model.state_dict(), save_path_val)
                sv_log = f"Saved best validation model to {save_path_val} with PSNR: {best_val_psnr:.2f}"
                print(sv_log)
                with open(cfg.train_log, "a") as f:
                    f.write(sv_log + "\n")

        # Track absolute lowest training loss
        if avg_loss < best_loss:
            best_loss = avg_loss


def test(cfg, model):
    """
    Evaluates the trained model on the test dataset. Generates restored images,
    calculates metrics, resizes outputs, and saves them to the configured directory.
    """
    # Verify the existence of the best validation checkpoint
    val_name = cfg.model_filename
    model_path_to_load = os.path.join(cfg.model_base_dir, val_name.replace(".pth", "_val.pth"))

    if not os.path.exists(model_path_to_load):
        print(f"Error: Model not found at {model_path_to_load}. Please train the model first.")
        return

    # Load and map legacy weights to the current model architecture
    print(f"Loading and converting legacy weights from: {model_path_to_load}")
    old_state_dict = torch.load(model_path_to_load, map_location=cfg.device, weights_only=True)
    new_state_dict = convert_old_weights_to_new(old_state_dict)

    model.load_state_dict(new_state_dict)
    model.eval()

    # Initialize dataloader and ensure output directory structure exists
    test_set = DehazeDataset(cfg.test_hazy, cfg.test_clean, cfg.test_nir, cfg.test_nir2, is_train=False)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=5)
    os.makedirs(cfg.save_dir, exist_ok=True)
    print(f"Saving output images to: {cfg.save_dir}")

    total_psnr = 0.0
    total_ssim = 0.0

    # Execute inference loop without tracking gradients
    with torch.no_grad():
        count = 1
        for batch_idx, (hazy, clean, nir, nir2) in enumerate(test_loader):
            hazy = hazy.to(cfg.device)
            clean = clean.to(cfg.device)
            nir = nir.to(cfg.device)
            nir2 = nir2.to(cfg.device)

            with torch.amp.autocast(cfg.device):
                output = model([hazy, nir, nir2])

            # Denormalize outputs and ground truths
            output = (output.clamp(-1, 1) + 1) / 2.0 * 255.0
            clean = (clean + 1) / 2.0 * 255.0

            # Convert to standard numpy format for processing and saving
            output_np = output.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
            clean_np = clean.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

            for i in range(output.size(0)):
                print(f"\rProcessing image {count}/{len(test_set)}...", end="", flush=True)

                fn = test_set.hazy_filenames[batch_idx * test_loader.batch_size + i]
                pred = output_np[i]
                gt = clean_np[i]

                # Calculate individual image metrics and accumulate
                current_psnr = psnr(gt, pred, data_range=255)
                current_ssim = ssim(gt, pred, data_range=255, channel_axis=2, multichannel=True)
                total_psnr += current_psnr
                total_ssim += current_ssim

                # Prepare metrics text (currently empty by configuration)
                metrics_text = (f"PSNR: {current_psnr:.2f}\nSSIM: {current_ssim:.4f}")
                #metrics_text = (f"")

                # Convert to PIL for resizing and annotation
                pred_pil = Image.fromarray(pred)
                pred_resized = pred_pil.resize((256, 192), Image.BILINEAR)
                pred_resized_np = np.array(pred_resized)

                # Overlay text and prepare for file system saving
                img_with_text = add_text_to_image(pred_resized_np, metrics_text)

                # Construct dynamic path based on datasets and networks
                save_path = os.path.join(cfg.save_dir, datasource, netsource, fn)
                save_dir = os.path.dirname(save_path)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                img_with_text.save(save_path)

                count += 1

    print("\nProcessing complete.")

    # Calculate and log final global metrics
    overall_avg_psnr = total_psnr / len(test_set)
    overall_avg_ssim = total_ssim / len(test_set)
    log_info = f"{datasource} {netsource} Overall PSNR: {overall_avg_psnr:.2f} dB Overall SSIM: {overall_avg_ssim:.4f}"
    print(log_info)

    with open(cfg.ablation_log, "a") as f:
        f.write(log_info + "\n")


if __name__ == "__main__":
    cfg = Config()

    # Initialize the scalable dynamic network architecture
    # The list [3, 1, 1] defines the input channel dimensions corresponding to RGB, NIR1, and NIR2
    model = S2UNet(
        in_channels_list=[3, 1, 1],
        dim=cfg.channels,
        dim_mults=(1, 2, 4, 5),
        num_blocks_encoder=[1, 1, 1, 1],
        num_blocks_decoder=[1, 1, 1, 1],
        share_aux_init_conv=True
    ).to(cfg.device)

    # Route execution based on configuration mode
    if cfg.mode == 'train':
        print("Starting training...")
        train(cfg, model)
        print("Training finished.")
    else:
        print("Starting testing...")
        test(cfg, model)
        print("Testing finished.")