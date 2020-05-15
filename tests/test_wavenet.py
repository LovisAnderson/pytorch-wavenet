import time
from wavenet_model import WaveNetModel, load_latest_model_from
from model_logging import TensorboardLogger
from wavenet_training import *
from torch.autograd import Variable
import torch
import numpy as np
from pathlib import Path
from audio_data import *

sample_dir = Path(__file__).parents[1] / 'train_samples'
model = WaveNetModel(layers=2,
                     blocks=2,
                     dilation_channels=2,
                     residual_channels=2,
                     skip_channels=2,
                     end_channels=4, 
                     output_length=16,
                     bias=True)

#print("model: ", model)
print("scope: ", model.receptive_field)

def delete_folder(pth) :
    for sub in pth.iterdir() :
        if sub.is_dir() :
                delete_folder(sub)
        else :
                sub.unlink()
    pth.rmdir()
test_folder = Path(__file__).parent / 'test_output'
if test_folder.exists():
    delete_folder(test_folder)
test_folder.mkdir()
in_path = Path(__file__).parents[1] / 'train_samples'
out_path = Path(__file__).parents[1] / 'train_samples/test_dataset.npz'
dataset = WavenetDataset(
    dataset_file=str(out_path),
    item_length=model.receptive_field + model.output_length - 1,
    target_length=model.output_length,
    file_location=str(in_path),
    test_stride=300)

print('Length dataset', len(dataset))

def generate_and_log_samples(step):
    sample_length=320
    gen_model = load_latest_model_from('snapshots')
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    
    logger.audio_summary('temperature_0.5', samples, step, sr=16000)
    print("audio clips generated")


logger = TensorboardLogger(log_interval=200,
                           validation_interval=200,
                           generate_interval=200,
                           generate_function=generate_and_log_samples,
                           log_dir=str(test_folder / 'logs'))
                           
trainer = WavenetTrainer(model=model,
                         dataset=dataset,
                         lr=0.001,
                         snapshot_path='snapshots',
                         snapshot_name='test_model',
                         snapshot_interval=100,
                         logger=logger
                         )
print('start training...')
tic = time.time()

trainer.train(batch_size=16,
              epochs=2)

toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

generated = model.generate_fast(500)
print(generated)
delete_folder(test_folder)