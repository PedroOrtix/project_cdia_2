# Fuente: 
# https://huggingface.co/spaces/JingyeChen22/TextDiffuser-2-Text-Inpainting

import torch

import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler,UNet2DConditionModel
from tqdm import tqdm
from PIL import Image, ImageDraw
from src.inpaint_functions import format_prompt, to_tensor, add_tokens


#### importar modelos de difusión
text_encoder = CLIPTextModel.from_pretrained('JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="text_encoder").cuda().half()

tokenizer = CLIPTokenizer.from_pretrained('pt-sk/stable-diffusion-1.5', subfolder="tokenizer")

vae = AutoencoderKL.from_pretrained('pt-sk/stable-diffusion-1.5', subfolder="vae").half().cuda()

unet = UNet2DConditionModel.from_pretrained('JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="unet", low_cpu_mem_usage=False, ignore_mismatched_sizes=True).half().cuda()

scheduler = DDPMScheduler.from_pretrained('pt-sk/stable-diffusion-1.5', subfolder="scheduler") 


def inpaint(orig_i, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_natural, global_dict):
    """
    Realiza el proceso de inpainting en una imagen utilizando un modelo de difusión de texto.
    
    Args:
        orig_i (PIL.Image): Imagen original sobre la que se realizará el inpainting.
        prompt (str): Texto descriptivo para guiar la generación.
        keywords (str): Palabras clave adicionales para el proceso.
        positive_prompt (str): Prompt positivo adicional.
        radio (int): Radio para el proceso de inpainting.
        slider_step (int): Número de pasos de difusión.
        slider_guidance (float): Factor de guía para el proceso de difusión.
        slider_batch (int): Tamaño del lote para procesamiento por batches.
        slider_natural (bool): Si se debe usar procesamiento de lenguaje natural.
        global_dict (dict): Diccionario con configuraciones globales.
        
    Returns:
        tuple: Tupla conteniendo las imágenes generadas y el prompt compuesto utilizado.
    """

    # print(type(i))
    # exit(0)
    add_tokens(tokenizer, text_encoder)

    print(f'[info] Prompt: {prompt} | Keywords: {keywords} | Radio: {radio} | Steps: {slider_step} | Guidance: {slider_guidance} | Natural: {slider_natural}')
    print(f'Global Stack: {global_dict["stack"]}')

    # global stack
    # global state

    if len(positive_prompt.strip()) != 0:
        prompt += positive_prompt

    with torch.no_grad():
        image_mask = Image.new('L', (512,512), 0)
        draw = ImageDraw.Draw(image_mask)


        ### Tokenizador CLIP
        if slider_natural:
            user_prompt = f'{prompt}'
            composed_prompt = user_prompt
            prompt = tokenizer.encode(user_prompt)
            
        else:
            user_prompt = format_prompt(draw, prompt, global_dict['stack'])
                
            prompt = tokenizer.encode(user_prompt)
            
            composed_prompt = tokenizer.decode(prompt)

            print("Prompt Compuesto:",composed_prompt)
        
         
        prompt = prompt[:77]
        while len(prompt) < 77: 
            prompt.append(tokenizer.pad_token_id) 

        prompts_cond = prompt
        prompts_nocond = [tokenizer.pad_token_id]*77
        
        ### Codificador CLIP
        
        prompts_cond = [prompts_cond] * slider_batch
        prompts_nocond = [prompts_nocond] * slider_batch

        prompts_cond = torch.Tensor(prompts_cond).long().cuda()
        prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()
        
        encoder_hidden_states_cond = text_encoder(prompts_cond)[0].half()
        encoder_hidden_states_nocond = text_encoder(prompts_nocond)[0].half()

        ### Aplicar máscara
                
        # image_mask convertido a tensor float16
        image_mask = torch.Tensor(np.array(image_mask)).float().half().cuda()
        
        # (H, W) -> (1, H, W) -> (1, 1, H, W) * (B, 1, 1, 1) = (B, 1, H, W)
        # El resultado es el tensor de máscara de imagen repetido B veces a lo largo de la dimensión del lote
        image_mask = image_mask.unsqueeze(0).unsqueeze(0).repeat(slider_batch, 1, 1, 1)

        # Redimensionar la imagen original a 512x512
        image = orig_i.resize((512,512))
        
        # convertir valores del tensor de imagen a distribución de [-1,1]
        image_tensor = to_tensor(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)   # (1, 3, 512, 512)
        
        # establecer área enmascarada en tensor a 0
        masked_image = image_tensor * (1-image_mask)
        
        ### Codificador VAE
        
        # vector de características latentes muestreado de la distribución codificada del VAE
        masked_feature = vae.encode(masked_image.half()).latent_dist.sample()
        
        # Escalar la característica latente muestreada por un factor de escala especificado en la configuración VAE
        masked_feature = (masked_feature * vae.config.scaling_factor).half() # (4, 4, 64, 64)
        
        print(f'forma de masked_feature {masked_feature.shape}')

        ## Muestreador DDPM
        
        # Redimensionar la máscara de imagen a 64x64 usando interpolación de vecino más cercano
        feature_mask = torch.nn.functional.interpolate(image_mask, size=(64,64), mode='nearest').cuda()
        noise = torch.randn((slider_batch, 4, 64, 64)).to("cuda").half()
        scheduler.set_timesteps(slider_step) 
        
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():  # guía libre de clasificador

                noise_pred_cond = unet(sample=noise, timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:slider_batch],feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
                noise_pred_uncond = unet(sample=noise, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:slider_batch],feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
                noisy_residual = noise_pred_uncond + slider_guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
                noise = scheduler.step(noisy_residual, t, noise).prev_sample
                del noise_pred_cond
                del noise_pred_uncond

                torch.cuda.empty_cache()

        ## Decodificador VAE
        
        noise = 1 / vae.config.scaling_factor * noise 
        images = vae.decode(noise, return_dict=False)[0] 
        width, height = 512, 512
        results = []
        new_image = Image.new('RGB', (2*width, 2*height))
        for index, image in enumerate(images.cpu().float()):
            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
            results.append(image)
            row = index // 2
            col = index % 2
            new_image.paste(image, (col*width, row*height))
            
        # Liberar memoria de la GPU
        torch.cuda.empty_cache()
        return tuple(results), composed_prompt
