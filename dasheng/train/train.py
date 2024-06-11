from loguru import logger
from fire import Fire
import numpy as np
from audiowebdataset import create_dataloader

import models
import utils
import torch
import sys
import ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events)
from ignite.handlers import (Checkpoint, DiskSaver, global_step_from_engine,
                             create_lr_scheduler_with_warmup)
from accelerate import Accelerator

logger.configure(handlers=[{
    "sink": sys.stderr,
    "format": "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}",
    'level': 'DEBUG',
}])


def transfer_to_device(batch, device):
    return (x.to(device, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def create_engine(engine_function,
                  local_rank: int = 0,
                  output_transform=lambda x: x):
    engine = Engine(engine_function)
    if local_rank == 0:
        ProgressBar().attach(engine, output_transform=output_transform)
    return engine


class Runner(object):

    def __init__(self, seed: int = 42, nthreads: int = 1):
        super().__init__()
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(nthreads)
        logger.info(f"Using seed {seed}")

    def __create_dir(self, config: utils.MAEConfig):
        config.outputdir.mkdir(exist_ok=True, parents=True)
        logger.add(
            config.outputdir / config.logfile,
            enqueue=True,
            level='INFO',
            format=
            "[<red>{level}</red> <green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
        )

    def log_basic_info(self, config_parameters: utils.MAEConfig, device):
        logger.info(f"Running on device {device}")
        logger.info(f"Storing output in {config_parameters.outputdir}")
        logger.info(f"- PyTorch version: {torch.__version__}")
        logger.info(f"- Ignite version: {ignite.__version__}")
        if torch.cuda.is_available():
            logger.info(f"- GPU Device: {torch.cuda.current_device()}")
            logger.info(f"- CUDA version: {torch.version.cuda}")
        for k, v in config_parameters.to_dict().items():
            logger.info(f"{k} : {v}")

    def train(self, config, **overwrite_kwargs):
        config_parameters = utils.MAEConfig.from_config_file(
            config, **overwrite_kwargs)
        accelerator = Accelerator()

        def log(message: str):
            if accelerator.is_main_process:
                logger.info(message)

        if accelerator.is_main_process:
            self.__create_dir(config_parameters)
            self.log_basic_info(config_parameters, device=accelerator.device)
        train_dataloader = create_dataloader(
            config_parameters.train_data,
            crop_size=int(config_parameters.chunk_length *
                          config_parameters.sample_rate),
            batch_size=config_parameters.batch_size,
            crop_shuffle=config_parameters.crop_shuffle,
            resampled=True)

        test_dataloader = create_dataloader(
            config_parameters.cv_data,
            crop_size=int(config_parameters.chunk_length *
                          config_parameters.sample_rate),
            batch_size=config_parameters.batch_size)

        model = getattr(
            models,
            config_parameters.model)(**config_parameters.model_args).train()
        log(model)

        if '8bit' in config_parameters.optimizer:
            import bitsandbytes as bnb
            optimizer = getattr(bnb.optim, config_parameters.optimizer)(
                model.parameters(),
                **config_parameters.optimizer_args)  # add bnb optimizer
        else:
            optimizer = getattr(torch.optim, config_parameters.optimizer)(
                model.parameters(), **config_parameters.optimizer_args)

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                with accelerator.autocast():
                    x, *_ = transfer_to_device(batch, accelerator.device)
                    loss = model(x,
                                 mask_ratio=config_parameters.mask_ratio,
                                 return_loss=True)
                    return loss

        def train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                x, *_ = transfer_to_device(batch, accelerator.device)
                optimizer.zero_grad()
                with accelerator.autocast():
                    loss = model(x,
                                 mask_ratio=config_parameters.mask_ratio,
                                 return_loss=True)
                    accelerator.backward(loss)
                    optimizer.step()
                    return {
                        'loss': loss.item(),
                        'lr': optimizer.param_groups[0]['lr']
                    }

        def run_validation(engine, title=None):
            if accelerator.is_main_process:
                results = engine.state.metrics
                output_str_list = [
                    f"{title:<10} Results - Epoch : {train_engine.state.epoch:<4}"
                ] + [
                    f"{metric} {results[metric]:<5.4f}" for metric in results
                ] + [f"LR: {optimizer.param_groups[0]['lr']:.2e}"]
                log(" ".join(output_str_list))

        train_engine = create_engine(train_batch)
        inference_engine = create_engine(_inference, output_transform=None)
        ignite.metrics.Average().attach(inference_engine, 'Loss')

        score_function = Checkpoint.get_default_score_fn(
            *config_parameters.score_function)
        checkpoint_saver = Checkpoint(
            {
                'model': model.encoder,
                'config': config_parameters,
            },
            DiskSaver(config_parameters.outputdir),
            n_saved=config_parameters.n_saved,
            global_step_transform=global_step_from_engine(train_engine),
            filename_prefix='best',
            score_function=score_function)
        last_checkpoint_saver = Checkpoint(
            {
                'model': model.encoder,
                'config': config_parameters
            },
            DiskSaver(config_parameters.outputdir),
            n_saved=1,
            global_step_transform=global_step_from_engine(train_engine))

        train_length = config_parameters.epoch_length * config_parameters.epochs
        decay_steps = train_length

        if config_parameters.use_scheduler:
            scheduler = ignite.handlers.param_scheduler.CosineAnnealingScheduler(
                optimizer, 'lr', optimizer.param_groups[0]['lr'],
                optimizer.param_groups[0]['lr'] * config_parameters.decay_frac,
                decay_steps)
            warmup_time_in_iters = None
            if config_parameters.warmup_iters is not None:
                warmup_time_in_iters = config_parameters.warmup_iters
            elif config_parameters.warmup_epochs is not None:
                warmup_time_in_iters = config_parameters.epoch_length * config_parameters.warmup_epochs
            if warmup_time_in_iters is not None:
                log(f"Using warmup with {warmup_time_in_iters} iters")
                scheduler = create_lr_scheduler_with_warmup(
                    scheduler,
                    warmup_start_value=0.0,
                    warmup_duration=warmup_time_in_iters)

            train_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
        inference_engine.add_event_handler(Events.COMPLETED, checkpoint_saver)
        inference_engine.add_event_handler(Events.COMPLETED,
                                           last_checkpoint_saver)

        @train_engine.on(
            Events.EPOCH_COMPLETED(every=config_parameters.valid_every))
        def valid_eval(train_engine):
            with inference_engine.add_event_handler(Events.COMPLETED,
                                                    run_validation,
                                                    "Validation"):
                inference_engine.run(test_dataloader)

        model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader)

        train_engine.run(
            train_dataloader,
            max_epochs=config_parameters.epochs,
            epoch_length=config_parameters.epoch_length,
        )
        output_model = config_parameters.outputdir / checkpoint_saver.last_checkpoint
        if config_parameters.average_final_model:
            log("Averaging best models ...")
            output_model = config_parameters.outputdir / 'averaged.pt'

            averaged_state_dict = utils.average_models([
                config_parameters.outputdir / f.filename
                for f in checkpoint_saver._saved
            ])
            torch.save(averaged_state_dict, output_model)


if __name__ == "__main__":
    Fire(Runner().train)
