use std::default::Default;
use std::error::Error;

use ash::vk;
use windows_sys::Win32::UI::WindowsAndMessaging::{SetWindowLongPtrA, GWLP_USERDATA};

mod helper;
mod win;
mod example_model;


fn main() -> Result<(), Box<dyn Error>> {
    unsafe {
        let window_width = 1920;
        let window_height = 1080;
        let hwnd = win::get_hwnd(window_width as i32, window_height as i32);
        let base = helper::vk_create_base(hwnd);
        let mut target = helper::setup_swapchain(&base, None);
        let sync = helper::vk_create_sync(&base, &target);
        let mut process = helper::vk_create_process(&base, &target);

        SetWindowLongPtrA(
            hwnd,
            GWLP_USERDATA,
            Box::into_raw(Box::new(process.camera)) as isize
        );
        
        // region MAIN LOOP
        let mut current_frame = 0; // which image in swapchain currently
        // todo: turn around, make win module be main() and call this instead of other way round
        let _ = win::render_loop(|| {
            // main loop stuff
            win::handle_input(&mut process.camera);
            // win::center_cursor(hwnd);   // Keep cursor centered for continuous rotation
            process.update_uniform_buffer(&base, &target);  // Add this line
            
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
            let (present_index, _) = match base.swapchain_loader.acquire_next_image(
                target.swapchain,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    // recreate the target after a window resize
                    target = helper::setup_swapchain(&base, Some(target.swapchain));
                    println!("Recreated the target after resize/...");
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

            // todo: minimize window -> res = 0 -> error
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

            // region RENDER PASS
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
            base.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                process.pipeline_layout,  // Make sure this is stored in your Process struct
                0, // first set
                &[process.descriptor_sets[0]], // descriptor sets to bind
                &[], // dynamic offsets
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
                vk::IndexType::UINT16, // todo: u16 here depends on indices type
            );
            // draw call
            base.device.cmd_draw_indexed(
                command_buffer,
                process.index_buffer_data.len() as u32,
                1,
                0,
                0,
                1,
            );
            base.device.cmd_end_render_pass(command_buffer);
            // endregion RENDER PASS

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

            match base.swapchain_loader.queue_present(sync.present_queue, &present_info) {
                Ok(_) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    // recreate the target after a window resize
                    target = helper::setup_swapchain(&base, Some(target.swapchain));
                    println!("Recreated the target after resize/...");
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

        // todo: buffers for vertices/indices etc. are not dropped by rust, need to do so to avoid leak
        // todo: need a way to cleanup all the vulkan stuff if we want to separately stop vulkan
    }

    Ok(())
}
