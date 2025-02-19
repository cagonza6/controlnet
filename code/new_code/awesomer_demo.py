
import yaml
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import seed_everything

from share import disable_verbosity   # disables some verbosity

# internal dependencies
import config
import einops
import random

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.ddim_hacked import DDIMSampler
from cldm.cldm import ControlLDM
import imageio


disable_verbosity()
# force CPU, I do not have nvidia cards and ROCm is not working right
device = torch.device('cpu')


relative_path = Path(__file__)
root_path =  relative_path.parent
modelconfig_path = root_path / 'models/cldm_v15.yaml'
ckpt_path = root_path /  'models/control_sd15_canny.pth'

input_image = imageio.imread(root_path / 'test_imgs/mri_brain.jpg')


def load_conf(conf_path: str) -> dict:
    """Helper to read yaml files. in particular the configuration.

    :param conf_path: path to yaml file
    :return: dict
    """
    with open(conf_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_config_path: str, model_file_path: str) -> ControlLDM:
    """Loads the given model with the given configuration
    It forces the usage of the CPU.

    :param model_config_path: path to the model configuration path
    :param model_file_path: path to the model file 
    :return:
    """
    model_config_path = load_conf(model_config_path)
    m_config = model_config_path['model']
    params = m_config['params']
    model = ControlLDM(**params)
    state_dict = torch.load(model_file_path,
                            map_location=torch.device(device))
    model.load_state_dict(state_dict)
    return model.cpu()


def process(model, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
            scale, seed, eta, low_threshold, high_threshold):
    ddim_sampler = DDIMSampler(model)
    apply_canny = CannyDetector()

    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        #  Here I'm forcing to CPU
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

if __name__ == "__main__":
	# this ensures the requirements besides the model are downloaded
	load_model(modelconfig_path, ckpt_path)
