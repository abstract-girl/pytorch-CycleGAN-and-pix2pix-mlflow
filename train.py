"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import mlflow
import os
import numpy as np
import io
import torch


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    
    # MLflow setup
    if opt.use_mlflow:
        # Set MLflow authentication through environment variables if credentials are provided
        if opt.mlflow_username and opt.mlflow_password:
            os.environ['MLFLOW_TRACKING_USERNAME'] = opt.mlflow_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = opt.mlflow_password
            print("Using MLflow with Basic authentication via environment variables")
            
        mlflow.set_tracking_uri(opt.mlflow_tracking_uri)
        mlflow.set_experiment(opt.mlflow_experiment_name)
        
        # Enable autolog if specified
        if opt.mlflow_autolog:
            print("Enabling MLflow autolog")
            # Enable pytorch autolog with a few customizations
            mlflow.pytorch.autolog(
                log_every_n_epoch=1,
                log_models=True,
                disable_for_unsupported_versions=False,
            )
            
        # Start MLflow run
        with mlflow.start_run(run_name=opt.name) as run:
            # Log parameters
            for key, value in vars(opt).items():
                if key != 'gpu_ids':  # Skip complex parameters
                    mlflow.log_param(key, value)
            
            dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
            dataset_size = len(dataset)    # get the number of images in the dataset.
            print('The number of training images = %d' % dataset_size)

            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
            total_iters = 0                # the total number of training iterations

            for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
                epoch_start_time = time.time()  # timer for entire epoch
                iter_data_time = time.time()    # timer for data loading per iteration
                epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
                visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
                model.update_learning_rate()    # update learning rates in the beginning of every epoch.
                
                # Log learning rate at the beginning of each epoch
                if opt.use_mlflow:
                    mlflow.log_metric("learning_rate", model.optimizers[0].param_groups[0]['lr'], step=epoch)
                
                for i, data in enumerate(dataset):  # inner loop within one epoch
                    iter_start_time = time.time()  # timer for computation per iteration
                    if total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    total_iters += opt.batch_size
                    epoch_iter += opt.batch_size
                    model.set_input(data)         # unpack data from dataset and apply preprocessing
                    model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                    if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                        save_result = total_iters % opt.update_html_freq == 0
                        model.compute_visuals()
                        visuals = model.get_current_visuals()
                        visualizer.display_current_results(visuals, epoch, save_result)
                        

                    if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                        losses = model.get_current_losses()
                        t_comp = (time.time() - iter_start_time) / opt.batch_size
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                        
                        # Always log metrics to MLflow at each print_freq, regardless of autolog setting
                        if opt.use_mlflow:
                            for loss_name, loss_value in losses.items():
                                mlflow.log_metric(f"loss/{loss_name}", loss_value, step=total_iters)
                            mlflow.log_metric("time/data", t_data, step=total_iters)
                            mlflow.log_metric("time/compute", t_comp, step=total_iters)
                            mlflow.log_metric("progress/epoch", epoch, step=total_iters)
                            mlflow.log_metric("progress/iteration", total_iters, step=total_iters)

                    if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                        model.save_networks(save_suffix)
                        
                        # Log latest model checkpoint to MLflow
                        if opt.use_mlflow:
                            checkpoint_files = []
                            for net_name in model.model_names:
                                save_filename = '%s_net_%s.pth' % (save_suffix, net_name)
                                save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
                                if os.path.exists(save_path):
                                    checkpoint_files.append(save_path)
                            
                            # Log checkpoints as artifacts
                            for ckpt_file in checkpoint_files:
                                artifact_path = f"checkpoints/iter_{total_iters}"
                                mlflow.log_artifact(ckpt_file, artifact_path=artifact_path)

                    iter_data_time = time.time()
                if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                    model.save_networks('latest')
                    model.save_networks(epoch)
                    
                    # Log epoch model to MLflow
                    if opt.use_mlflow:
                        # Save checkpoints directory to MLflow
                        checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
                        for net_name in model.model_names:
                            save_filename = '%s_net_%s.pth' % (epoch, net_name)
                            save_path = os.path.join(checkpoint_dir, save_filename)
                            if os.path.exists(save_path):
                                mlflow.log_artifact(save_path, artifact_path=f"checkpoints/epoch_{epoch}")

                epoch_time = time.time() - epoch_start_time
                print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, epoch_time))
                # Log epoch metrics to MLflow
                if opt.use_mlflow:
                    mlflow.log_metric("epoch/duration_seconds", epoch_time, step=epoch)
                    mlflow.log_metric("epoch/completed", epoch, step=epoch)
    else:
        # Original training code without MLflow
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        total_iters = 0                # the total number of training iterations

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            model.update_learning_rate()    # update learning rates in the beginning of every epoch.
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
