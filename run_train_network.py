import argparse, configparser, subprocess

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    kohya_directory = config.get('Directories', 'kohya_directory')
    pretrained_model_name = config.get('Directories', 'pretrained_model_name')
    
    parser = argparse.ArgumentParser(description="Runs Kohya training on target directory")
    
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--output', required=True, type=str, help='Output name for LORA')
    parser.add_argument('-m', '--model', required=True, type=str, help='Name of pretrained model')
    
    args = parser.parse_args()
        
    ffmpeg_command = [
        f'{kohya_directory}venv/Scripts/accelerate', 'launch', '--num_cpu_threads_per_process=2', f'{kohya_directory}train_network.py',
        '--enable_bucket', '--min_bucket_reso=256', '--max_bucket_reso=2048', 
        f'--pretrained_model_name_or_path={pretrained_model_name + args.model}',
        f'--train_data_dir={args.input}img', 
        f'--reg_data_dir={args.input}reg', 
        '--resolution=512,512',
        f'--output_dir={args.input}model', 
        f'--logging_dir={args.input}log', 
        '--network_alpha=1', '--save_model_as=safetensors', '--network_module=networks.lora', '--text_encoder_lr=0.0004', '--unet_lr=0.0004', '--network_dim=256', 
        f'--output_name={args.output}',
        '--lr_scheduler_num_cycles=10', '--no_half_vae', 
        '--learning_rate=0.0004', '--lr_scheduler=constant', 
        '--train_batch_size=1', '--max_train_steps=240', 
        '--save_every_n_epochs=1', 
        '--mixed_precision=bf16', '--save_precision=bf16', 
        '--cache_latents', '--cache_latents_to_disk', 
        '--optimizer_type=Adafactor', 
        '--optimizer_args', 'scale_parameter=False', 'relative_step=False', 'warmup_init=False', 
        '--max_data_loader_n_workers=0', 
        '--bucket_reso_steps=64',
        '--xformers', 
        '--bucket_no_upscale', 
        '--noise_offset=0.0'
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f'Conversion complete.')
    except subprocess.CalledProcessError as e:
        print(f'Error during conversion: {e}')
    
if __name__ == '__main__':
    main()