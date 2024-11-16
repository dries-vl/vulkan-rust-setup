use std::default::Default;
use std::error::Error;

use ash::vk;

mod helper;
mod win;
mod target;


fn main() -> Result<(), Box<dyn Error>> {
    unsafe {
        let (mut base, mut target, mut sync) = helper::vk_create(1920, 1080)?;
        let process = helper::vk_create_process(&base, &target);
        
        // region MAIN LOOP
        let mut current_frame = 0;
        let _ = helper::render_loop(|| {
            let frame = &sync.frame_data[current_frame];
            let command_buffer = frame.command_buffer;
            let present_complete_semaphore = frame.present_complete_semaphore;
            let rendering_complete_semaphore = frame.rendering_complete_semaphore;
            let fence = frame.fence;

            // Wait for previous frame to complete
            base.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();

            // Get next image before resetting fence
            let (present_index, _) = match target.swapchain_loader.acquire_next_image(
                target.swapchain,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    helper::recreate_swapchain(
                        &mut base, 
                        &mut target
                    ).expect("Failed to recreate swapchain");
                    println!("Need to recreate swapchain after acquire_next_image");
                    return true;
                }
                Err(e) => {
                    println!("Failed to acquire next image: {:?}", e);
                    return false;
                }
            };

            // Reset fence only after acquiring image
            base.device.reset_fences(&[fence]).unwrap();

            // Get elapsed time in seconds as a high-precision float
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();;
            let r = ((time * 1.0).cos()); // Red channel
            let g = ((time * 1.5).cos()); // Green channel
            let b = ((time * 2.0).cos()); // Blue channel
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [r as f32, g as f32, b as f32, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(target.renderpass)
                .framebuffer(target.framebuffers[present_index as usize])
                .render_area(target.surface_resolution.into())
                .clear_values(&clear_values);

            base.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .expect("Reset command buffer failed.");

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            base.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Begin commandbuffer");

            base.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            base.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                process.pipelines[0], // first pipeline
            );
            base.device.cmd_set_viewport(command_buffer, 0, &target.viewports);
            base.device.cmd_set_scissor(command_buffer, 0, &target.scissors);
            base.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[process.vertex_buffer],
                &[0],
            );
            base.device.cmd_bind_index_buffer(
                command_buffer,
                process.index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            base.device.cmd_draw_indexed(
                command_buffer,
                process.index_buffer_data.len() as u32,
                1,
                0,
                0,
                1,
            );
            base.device.cmd_end_render_pass(command_buffer);

            base.device
                .end_command_buffer(command_buffer)
                .expect("End commandbuffer");

            let wait_semaphores = [present_complete_semaphore];
            let command_buffers = [command_buffer];
            let signal_semaphores = [rendering_complete_semaphore];
            let swapchains = [target.swapchain];
            let present_indices = [present_index];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            // Submit with fence
            base.device
                .queue_submit(sync.present_queue, &[submit_info], fence)
                .expect("queue submit failed.");

            // Present
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&present_indices);

            match target.swapchain_loader.queue_present(sync.present_queue, &present_info) {
                Ok(_) => println!("Present succeeded"),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    helper::recreate_swapchain(
                        &mut base,
                        &mut target
                    ).expect("Failed to recreate swapchain");
                    println!("Need to recreate swapchain after queue_present");
                    return true;
                }
                Err(e) => {
                    println!("Failed to present queue: {:?}", e);
                    return false;
                }
            }

            current_frame = (current_frame + 1) % sync.frames_in_flight;
            true
        });
        // endregion MAIN LOOP

        // region CLEANUP
        // todo: no longer complete, but is really needed anyway when process ends (?)
        base.device.device_wait_idle().unwrap();
        for pipeline in process.pipelines {
            base.device.destroy_pipeline(pipeline, None);
        }
        base.device.destroy_pipeline_layout(process.pipeline_layout, None);
        base.device
            .destroy_shader_module(process.vertex_shader_module, None);
        base.device
            .destroy_shader_module(process.fragment_shader_module, None);
        base.device.free_memory(process.index_buffer_memory, None);
        base.device.destroy_buffer(process.index_buffer, None);
        base.device.free_memory(process.vertex_buffer_memory, None);
        base.device.destroy_buffer(process.vertex_buffer, None);
        for framebuffer in target.framebuffers {
            base.device.destroy_framebuffer(framebuffer, None);
        }
        base.device.destroy_render_pass(target.renderpass, None);
        // endregion CLEANUP
    }

    Ok(())
}
