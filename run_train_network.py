import argparse, configparser, subprocess, os, shutil

def run_training(input, output, model, repeat, instance, class_name, reg=''):
    config = configparser.ConfigParser()
    config.read('config.ini')
    kohya_directory = config.get('Directories', 'kohya_directory')
    pretrained_model_name = config.get('Directories', 'pretrained_model_name')
    
    image_files = os.listdir(input)
    
    os.mkdir(f'{input}lora')
    os.mkdir(f'{input}lora/img')
    os.mkdir(f'{input}lora/img/{repeat}_{instance} {class_name}')
    os.mkdir(f'{input}lora/model')
    os.mkdir(f'{input}lora/log')
    
    for file in image_files:
        source_file = os.path.join(input, file)
        destination_file = os.path.join(f'{input}lora/img/{repeat}_{instance} {class_name}/', file)
        
        # Use shutil.move() to move the file
        shutil.copy(source_file, destination_file)
        print(f"Moved: {source_file} -> {destination_file}")
    
    
    ffmpeg_command = [
        f'{kohya_directory}venv/Scripts/accelerate', 'launch', '--num_cpu_threads_per_process=2', f'{kohya_directory}train_network.py',
        '--enable_bucket', '--min_bucket_reso=256', '--max_bucket_reso=2048', 
        f'--pretrained_model_name_or_path={pretrained_model_name + model}',
        f'--train_data_dir={input}lora/img', 
        f'--reg_data_dir={reg}', 
        '--resolution=512,512',
        f'--output_dir={input}lora/model', 
        f'--logging_dir={input}lora/log', 
        '--network_alpha=1', '--save_model_as=safetensors', '--network_module=networks.lora', 
        '--text_encoder_lr=0.0004', '--unet_lr=0.0004', '--network_dim=256', 
        f'--output_name={output}',
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

def main():
    parser = argparse.ArgumentParser(description="Runs Kohya training on target directory")
    
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--output', required=True, type=str, help='Output name for LORA')
    parser.add_argument('-m', '--model', required=True, type=str, help='Name of pretrained model')
    parser.add_argument('-r', '--repeat', default=3, type=str, help='Number of times to repeat each training image')
    parser.add_argument('-n', '--instance', required=True, type=str, help='Name of the instance. This is the trigger word')
    parser.add_argument('-c', '--class_name', required=True, type=str, help='Class name of the LORA')
    
    args = parser.parse_args()
    
    run_training(args.input, args.output, args.model, args.repeat, args.instance, args.class_name)
        
if __name__ == '__main__':
    main()