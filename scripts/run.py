#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

import csv

# import torch
# import torch.nn as nn
# import torch.optim as optim

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")
	parser.add_argument("--profile", default="", help="Path to memory and time profiling data")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video.")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")


	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	args.mode = args.mode or mode_from_scene(args.scene) or mode_from_scene(args.load_snapshot)
	if not args.mode:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	if args.mode == "sdf":
		mode = ngp.TestbedMode.Sdf
		configs_dir = os.path.join(ROOT_DIR, "configs", "sdf")
		scenes = scenes_sdf
	elif args.mode == "nerf":
		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
	elif args.mode == "image":
		mode = ngp.TestbedMode.Image
		configs_dir = os.path.join(ROOT_DIR, "configs", "image")
		scenes = scenes_image
	elif args.mode == "volume":
		mode = ngp.TestbedMode.Volume
		configs_dir = os.path.join(ROOT_DIR, "configs", "volume")
		scenes = scenes_volume
	else:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	base_network = os.path.join(configs_dir, "base.json")
	if args.scene in scenes:
		network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
		base_network = os.path.join(configs_dir, network+".json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene = args.scene
		if not os.path.exists(args.scene) and args.scene in scenes:
			scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
		testbed.load_training_data(scene)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw*sh > 1920*1080*4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window = args.second_window or False)


	if args.load_snapshot:
		snapshot = args.load_snapshot
		if not os.path.exists(snapshot) and snapshot in scenes:
			snapshot = default_snapshot_filename(scenes[snapshot])
		print("Loading snapshot ", snapshot)
		testbed.load_snapshot(snapshot)
	else:
		testbed.reload_network_from_file(network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]
	if args.mode == "sdf":
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Optionally match nerf paper behaviour and train on a
		# fixed white bg. We prefer training on random BG colors.
		# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		# testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = 35000

	# # dummy loss, optimizer
	# criterion = nn.CrossEntropyLoss()
	# optimizer = optim.Adam(model.parameters(), lr=lr)

	tqdm_last_update = 0
	profiling_data = []
	t_train_time = 0
	t_train_time_start = time.monotonic()
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			testbed.set_max_access_window_size(size=0)
			while testbed.frame():
				# # GPU warmup
				# optimizer.zero_grad()
				# output = global_model.local_list[0](noise_input)
				# f_loss = criterion(output, noise_label)
				# if i == 1000:
				# 	print("Total CUDA Memory Allocated for training: %.2f MB"%(torch.cuda.memory_allocated(0)/1024/1024))
				# f_loss.backward()
				# optimizer.step()
				if args.profile:
					profiling_data.append([
						testbed.m_timer.m_nerf_reset_accumulation_start,
						testbed.m_timer.m_nerf_reset_accumulation_end,
						testbed.m_timer.m_nerf_reset_accumulation_end - testbed.m_timer.m_nerf_reset_accumulation_start,
						testbed.m_timer.m_nerf_train_prep_start,
						testbed.m_timer.m_nerf_train_prep_end,
						testbed.m_timer.m_nerf_train_prep_end - testbed.m_timer.m_nerf_train_prep_start,
						testbed.m_timer.m_nerf_update_hyperparam_start,
						testbed.m_timer.m_nerf_update_hyperparam_end,
						testbed.m_timer.m_nerf_update_hyperparam_end - testbed.m_timer.m_nerf_update_hyperparam_start,
						testbed.m_timer.m_nerf_i_dont_know1_start,
						testbed.m_timer.m_nerf_i_dont_know1_end,
						testbed.m_timer.m_nerf_i_dont_know1_end - testbed.m_timer.m_nerf_i_dont_know1_start,
						testbed.m_timer.m_nerf_train_start,
						testbed.m_timer.m_nerf_train_sampling_start,
						testbed.m_timer.m_nerf_train_sampling_end,
						testbed.m_timer.m_nerf_train_sampling_end - testbed.m_timer.m_nerf_train_sampling_start,
						testbed.m_timer.m_nerf_train_inference_start,
						testbed.m_timer.m_nerf_train_inference_end,
						testbed.m_timer.m_nerf_train_inference_end - testbed.m_timer.m_nerf_train_inference_start,
						testbed.m_timer.m_nerf_train_loss_start,
						testbed.m_timer.m_nerf_train_loss_end,
						testbed.m_timer.m_nerf_train_loss_end - testbed.m_timer.m_nerf_train_loss_start,
						testbed.m_timer.m_nerf_train_forward,
						testbed.m_timer.m_nerf_train_backward,
						testbed.m_timer.m_nerf_train_backward - testbed.m_timer.m_nerf_train_forward,
						testbed.m_timer.m_nerf_train_backward_end,
						testbed.m_timer.m_nerf_train_backward_end - testbed.m_timer.m_nerf_train_backward,
						testbed.m_timer.m_nerf_train_end,
						testbed.m_timer.m_nerf_optimizer_start,
						testbed.m_timer.m_nerf_optimizer_end,
						testbed.m_timer.m_nerf_optimizer_end - testbed.m_timer.m_nerf_optimizer_start,
						testbed.m_timer.m_nerf_envmap_start,
						testbed.m_timer.m_nerf_envmap_end,
						testbed.m_timer.m_nerf_envmap_end - testbed.m_timer.m_nerf_envmap_start,
						testbed.m_timer.m_nerf_rgb_loss_scalar_start,
						testbed.m_timer.m_nerf_start_numsteps_counter,
						testbed.m_timer.m_nerf_start_numsteps_counter_compacted,
						testbed.m_timer.m_nerf_start_numsteps_counter_compacted - testbed.m_timer.m_nerf_start_numsteps_counter,
						testbed.m_timer.m_nerf_end_numsteps_counter_compacted,
						testbed.m_timer.m_nerf_end_numsteps_counter_compacted - testbed.m_timer.m_nerf_start_numsteps_counter_compacted,
						testbed.m_timer.m_nerf_start_reduce_sum,
						testbed.m_timer.m_nerf_end_reduce_sum,
						testbed.m_timer.m_nerf_end_reduce_sum - testbed.m_timer.m_nerf_start_reduce_sum,
						testbed.m_timer.m_nerf_end_rays_per_batch,
						testbed.m_timer.m_nerf_end_rays_per_batch - testbed.m_timer.m_nerf_end_reduce_sum,
						testbed.m_timer.m_nerf_rgb_loss_scalar_end,
						testbed.m_timer.m_nerf_rgb_loss_scalar_end - testbed.m_timer.m_nerf_rgb_loss_scalar_start,
						testbed.m_timer.m_nerf_compute_cdf_start,
						testbed.m_timer.m_nerf_compute_cdf_end,
						testbed.m_timer.m_nerf_compute_cdf_end - testbed.m_timer.m_nerf_compute_cdf_start,
						testbed.m_timer.m_nerf_train_extra_dims_start,
						testbed.m_timer.m_nerf_train_extra_dims_end,
						testbed.m_timer.m_nerf_train_extra_dims_end - testbed.m_timer.m_nerf_train_extra_dims_start,
						testbed.m_timer.m_nerf_train_camera_start,
						testbed.m_timer.m_nerf_train_camera_end,
						testbed.m_timer.m_nerf_train_camera_end - testbed.m_timer.m_nerf_train_camera_start,
						testbed.m_timer.m_nerf_update_loss_graph_start,
						testbed.m_timer.m_nerf_update_loss_graph_end,
						testbed.m_timer.m_nerf_update_loss_graph_end - testbed.m_timer.m_nerf_update_loss_graph_start,
						testbed.m_mem_tracker.m_base,
						testbed.m_mem_tracker.m_nerf_train_prep_end,
						testbed.m_mem_tracker.m_nerf_train_sampling_end,
						testbed.m_mem_tracker.m_nerf_train_inference_end,
						testbed.m_mem_tracker.m_nerf_train_loss_end,
						testbed.m_mem_tracker.m_nerf_train_forward_end,
						testbed.m_mem_tracker.m_nerf_train_backward_end,
						testbed.m_mem_tracker.m_nerf_optimizer_end
						])

				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now
			
			t_train_time = time.monotonic() - t_train_time_start

	print(t_train_time, testbed.loss)

	if args.save_snapshot:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(args.save_snapshot, False)

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)
		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.rendering_min_transmittance = 1e-4

		testbed.fov_axis = 0
		testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
		testbed.shall_train = False

		with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
			for i, frame in t:
				p = frame["file_path"]
				if "." not in p:
					p = p + ".png"
				ref_fname = os.path.join(data_dir, p)
				if not os.path.isfile(ref_fname):
					ref_fname = os.path.join(data_dir, p + ".png")
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".jpg")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".jpeg")
							if not os.path.isfile(ref_fname):
								ref_fname = os.path.join(data_dir, p + ".exr")

				try:
					ref_image = read_image(ref_fname)
				except:
					continue

				# NeRF blends with background colors in sRGB space, rather than first
				# transforming to linear space, blending there, and then converting back.
				# (See e.g. the PNG spec for more information on how the `alpha` channel
				# is always a linear quantity.)
				# The following lines of code reproduce NeRF's behavior (if enabled in
				# testbed) in order to make the numbers comparable.
				if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
					# Since sRGB conversion is non-linear, alpha must be factored out of it
					ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
					ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
					ref_image[...,:3] *= ref_image[...,3:4]
					ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
					ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

				if i == 0:
					write_image("ref.png", ref_image)

				testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
				image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

				if i == 0:
					write_image("out.png", image)

				if ref_image.shape[2] == 3 and image.shape[2] == 4:
					image = np.delete(image, 3, 2)
				diffimg = np.absolute(image - ref_image)
				diffimg[...,3:4] = 1.0
				if i == 0:
					write_image("diff.png", diffimg)

				A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				totssim += ssim
				totmse += mse
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr<minpsnr else minpsnr
				maxpsnr = psnr if psnr>maxpsnr else maxpsnr
				totcount = totcount+1
				t.set_postfix(psnr = totpsnr/(totcount or 1))

		psnr_avgmse = mse2psnr(totmse/(totcount or 1))
		psnr = totpsnr/(totcount or 1)
		ssim = totssim/(totcount or 1)
		print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

	if args.profile:
		with open(os.path.join(args.profile, 'profiles.csv'), 'w', newline='') as f:
			w = csv.writer(f)
			w.writerow(['T_m_nerf_reset_accumulation_start', 	# 0
						'T_m_nerf_reset_accumulation_end',		# 1
						'Accumulation',							# 1 - 0
						'T_m_nerf_train_prep_start',			# 2
						'T_m_nerf_train_prep_end',				# 3
						'Train_prep',							# 3 - 2
						'T_m_nerf_update_hyperparam_start',		# 4
						'T_m_nerf_update_hyperparam_end',		# 5
						'Update_hyparam',						# 5 - 4
						'T_m_nerf_i_dont_know1_start',			# 6
						'T_m_nerf_i_dont_know1_end',			# 7
						'I_dont_know',							# 7 - 6
						'T_m_nerf_train_start',					# 8
						'T_m_nerf_train_sampling_start',		# 9
						'T_m_nerf_train_sampling_end',			# 10
						'Smapling',								# 10 - 9
						'T_m_nerf_train_inference_start',		# 11
						'T_m_nerf_train_inference_end',			# 12
						'Inference',							# 12 - 11
						'T_m_nerf_train_loss_start',			# 13
						'T_m_nerf_train_loss_end',				# 14
						'Loss', 								# 14 - 13
						'T_m_nerf_train_forward',				# 15
						'T_m_nerf_train_backward',				# 16
						'Forward',								# 16 - 15
						'T_m_nerf_train_backward_end',			# 17
						'Backward',								# 17 - 16
						'T_m_nerf_train_end',					# 18
						'T_m_nerf_optimizer_start',				# 19
						'T_m_nerf_optimizer_end',				# 20
						'Optimizer',							# 20 - 19
						'T_m_nerf_envmap_start',				# 21
						'T_m_nerf_envmap_end',					# 22
						'Envmap',								# 22 - 21
						'T_m_nerf_rgb_loss_scalar_start',		# 23
						'T_m_nerf_start_numsteps_counter',		# 23
						'T_m_nerf_start_numsteps_counter_compacted',		# 23
						'numsteps_counter',
						'T_m_nerf_end_numsteps_counter_compacted',
						'numsteps_counter_compacted',
						'T_m_nerf_start_reduce_sum',
						'T_m_nerf_end_reduce_sum',
						'reduce_sum',
						'T_m_nerf_end_rays_per_batch',
						'rays_per_batch',
						'T_m_nerf_rgb_loss_scalar_end',			# 24
						'Loss_scalar',							# 24 - 23
						'T_m_nerf_compute_cdf_start',			# 25
						'T_m_nerf_compute_cdf_end',				# 26
						'Compute_cdf',							# 26 - 25
						'T_m_nerf_train_extra_dims_start',		# 27
						'T_m_nerf_train_extra_dims_end',		# 28
						'Extra_dims',							# 28 - 27
						'T_m_nerf_train_camera_start',			# 29
						'T_m_nerf_train_camera_end',			# 30
						'Train_camera',							# 30 - 29
						'T_m_nerf_update_loss_graph_start',		# 31
						'T_m_nerf_update_loss_graph_end',		# 32
						'Loss_graph',							# 32 - 31
						'M_m_base',								# 33
						'M_m_nerf_train_prep_end',				# 34
						'M_m_nerf_train_sampling_end',			# 35
						'M_m_nerf_train_inference_end',			# 36
						'M_m_nerf_train_loss_end',				# 37
						'M_m_nerf_train_forward_end',			# 38
						'M_m_nerf_train_backward_end',			# 39
						'M_m_nerf_optimizer_end',				# 40
						])
			w.writerows(profiling_data)

	if args.save_mesh:
		res = args.marching_cubes_res or 256
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res])

	if ref_transforms:
		testbed.fov_axis = 0
		testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
		if not args.screenshot_frames:
			args.screenshot_frames = range(len(ref_transforms["frames"]))
		print(args.screenshot_frames)
		for idx in args.screenshot_frames:
			f = ref_transforms["frames"][int(idx)]
			cam_matrix = f["transform_matrix"]
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
			outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".png"

			print(f"rendering {outname}")
			image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)
	elif args.screenshot_dir:
		outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
		print(f"Rendering {outname}.png")
		image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
		if os.path.dirname(outname) != "":
			os.makedirs(os.path.dirname(outname), exist_ok=True)
		write_image(outname + ".png", image)

	if args.video_camera_path:
		testbed.load_camera_path(args.video_camera_path)

		resolution = [args.width or 1920, args.height or 1080]
		n_frames = args.video_n_seconds * args.video_fps

		if "tmp" in os.listdir():
			shutil.rmtree("tmp")
		os.makedirs("tmp")

		for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
			testbed.camera_smoothing = args.video_camera_smoothing and i > 0
			frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
			write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

		os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")
		shutil.rmtree("tmp")
